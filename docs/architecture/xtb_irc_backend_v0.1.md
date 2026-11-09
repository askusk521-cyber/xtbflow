# Bounded xTB TS/IRC backend v0.1

This adapter is the next certification slice after the live physical-feedback
loop. It is deliberately a downstream validator and does not generate a
candidate or read a product, reference TS, sealed label, or target-selected
environment.

The input is one explicit `MolecularSystem` snapshot plus a durable
`CalculatorCallToken`. Coordinates are Angstrom, energies are Hartree, forces
are Hartree/Angstrom, and the finite-difference Cartesian Hessian is reported
as unweighted Hartree/Angstrom². Every energy/force evaluation consumes one
token before entering the calculator. The result retains the protocol and
calculator build identity, input hash, call count, endpoint gradients, and
failure reason.

The bounded sequence is:

```text
current candidate snapshot
→ xTB energy/forces
→ finite-difference Cartesian Hessian
→ exactly-one-negative-mode validator
→ ± bounded gradient-following path
→ ASE endpoint minimization using the same calculator protocol
→ observed endpoint bond graphs
→ strict TS/path validator
```

Endpoint optimization is an optional ASE boundary. If ASE is unavailable, the
result is `blocked_backend`; calculator errors and exhausted tokens remain
`execution_failed` or `budget_exhausted` with consumed calls retained. The
runner records wall seconds, CPU seconds/hours, GPU hours, source identity,
configuration hash, input hash, and replay identity. OpenMP/BLAS threads are
fixed to one in the smoke runner so CPU accounting is reproducible.

The validator's `strict_validation` label means that the frozen software
protocol accepted the supplied stationary-point, mode, endpoint, and path
evidence. It does not mean the candidate was discovered by xtbflow and does
not support a barrier, mechanism, chemical-accuracy, generalization, or model
superiority claim. The initial real smoke uses an external fixed candidate on
n2; target-free integration into the live generator remains a separate gate.

The current live water fixture therefore stays `blocked_certification` until
its candidate-generation contract supplies a chemically meaningful candidate
without importing reference TS or endpoint labels. This module can be wired
through that gate without changing the existing calculator protocol or
matched-budget accounting.
