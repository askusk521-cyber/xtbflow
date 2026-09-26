# XTBFlow v0.1 architecture contract

This is the canonical interface specification for issues #9, #12, #13, #14 and #20. It freezes the scientific boundary and data views before model implementation. It does not claim that the architecture is trained, optimal, chemically valid, or novel by composition alone.

## Responsibilities

XTBFlow v0.1 has one proposal model and one independent physical model. The proposal model receives a fixed CHNOS atom pool, confirmed charge and singlet multiplicity, a reactant electronic bookkeeping state, an independently prepared reactant geometry, an explicit environment pool, and a finite reaction intent. It proposes conserved event candidates, geometry initial guesses, and optionally a sign-invariant reaction direction. The physical model evaluates the same geometry through `U_hat(R) = U_GFN2(R) + DeltaU_phi(R)` and returns energy and force information. A physical energy cannot depend on which event a proposal model happens to search.

The first implementation uses a shared scalar/vector equivariant representation with six message-passing blocks, 128 scalar channels, 32 vector channels and 128 pair channels. These are reproducible starting values, not evidence of optimality. The model keeps event and coordinate state available to one another at each declared flow update:

```text
db/dtau = f_theta(b_tau, R_tau, tau, c)
dR/dtau = g_theta(b_tau, R_tau, tau, c)
```

A serial event-then-geometry baseline and a joint bidirectional mode are separate registered controls. Turning coupling off must reproduce the corresponding baseline. The normalized flow parameter is a generation coordinate, not physical time or a reaction probability.

## Input and supervision boundary

Inference inputs contain reactant-side state and independently declared environment atoms. They never contain the true product, reference transition state, reference mode, target-derived active water, or sealed test labels. pH can influence an upstream microstate distribution; it cannot silently add or remove atoms during sampling. Training may use product or transition-state labels only in a separate supervision view. Reference energies, forces, connectivity and endpoint evidence remain downstream validation views.

Explicit hydrogens and a fixed atom pool are part of v0.1. Metals, radicals, excited states and open electron exchange are out of scope. Missing charge or multiplicity is `unknown` and quarantines a record rather than defaulting to neutral singlet.

## Constructive conservation

Electronic bookkeeping uses a linear constraint `A b = c_e`. A null-space or projection parameterization keeps both initial noise and the continuous velocity in the conservation subspace:

```text
Pi = I - A^T (A A^T)^+ A
b_0 = b_R + Pi noise
db/dtau = Pi f_theta
```

When a symmetric bond-electron matrix is stored by its upper triangle, diagonal entries count once and off-diagonal entries count twice. The discrete decoder must project, integerize and perform a legal-state check; rounding entries independently is not a conservation proof. Global electron bookkeeping is necessary and does not certify valence, spin or a feasible pathway.

## Physical guidance boundary

The candidate saddle correction is an explicitly metered downstream operation:

```text
G_sad = (I - 2 u u^T) F_hat,  ||u|| = 1
```

It is called only on clean candidates or late denoising states, with a declared coordinate metric, trust region, force clipping and SCF-failure policy. xTB nuclear forces are not gradients of the event bookkeeping variables. The physical module updates geometry and direction; the joint representation may then update event scores. v0.1 does not backpropagate through an electronic-structure solver. Finite differences and HVPs, if later enabled, are charged as calculator work in the shared runtime ledger.

Zero guidance and zero coupling are mandatory exact baselines. Causal gains must be compared with the same candidate, physical-call and final-validation budgets; a terminal optimizer is a distinct control from guidance inside generation.

## Three freeze decisions

1. **Interface freeze now:** scope, units, atom pool, input/output views, conservation definition, forbidden inference fields, and physical interface are fixed by [`configs/models/xtbflow_v0.1.yaml`](../../configs/models/xtbflow_v0.1.yaml).
2. **Development freeze after selection:** #26 may choose the smallest effective capacity and coupling mode using development groups only. It freezes the checkpoint, correction potential, candidate budget and run protocol.
3. **Confirmatory freeze before testing:** #28 freezes the code commit, weights, physical protocol, finite budget, stopping rule and an unseen confirmation batch. Any later change converts that batch back to development evidence.

## Shared acceptance contract

The JSON schema in [`schemas/model_contract.schema.json`](../../schemas/model_contract.schema.json) and `tests/test_architecture_contract.py` cover the interface-level invariants. #12 owns event flow and decoder implementation; #13 owns equivariant geometry and direction; #9/#14 own the correction potential and guidance interface; #20 owns serial/joint wiring. This document prevents incompatible contracts; it does not duplicate those implementations.

Open implementation choices are listed explicitly in the config. They must be selected from development evidence and recorded in a run manifest. No MoE, reinforcement learning, learned full Hessian or unbounded planning is part of v0.1.
