# Workflow B model and baseline protocol

Status: software protocol frozen; scientific execution remains blocked by issue #58.

## Scope

Workflow B implements and tests the model/control boundary owned by issues #20,
#44, and #56.  It does not admit energy/force-only records as event or
transition-state supervision, does not call a quantum-chemistry calculator, and
does not decide that the joint model is better than a serial control.

The canonical scientific interface remains
[`architecture_v0.1.md`](architecture_v0.1.md).  The exact runnable prototype is
recorded separately in
[`joint_flow_runtime_v0.1.json`](../../configs/models/joint_flow_runtime_v0.1.json).
That separation is intentional: the current bounded prototype uses one packed
pair-event block and a pairwise equivariant coordinate field; it does **not**
yet claim to implement or validate the canonical six-block capacity.

## Event and geometry representation

The event state is the packed upper triangle of a symmetric bond/electron
matrix.  Under an atom permutation `P`, the unpacked event state transforms as
`B -> P B P^T`.  Shared pair networks consume symmetric atom-pair features,
and event velocities are projected inside the active conservation subspace.
Padding is excluded before projection, so inactive event entries remain exactly
zero rather than being masked after a global projection.

Geometry velocities use invariant pair scalars multiplied by relative
coordinate vectors.  This preserves translation invariance and proper-rotation
and atom-permutation equivariance.  The joint tests transform both event and
coordinate states together; testing only the geometry branch is insufficient.

## Registered learned controls

All learned controls are evaluated through one
`JointEventGeometryFlow` parameter set:

| Mode | Geometry -> event | Event -> geometry | Integrator requirement |
|---|---:|---:|---|
| `both_off` | no | no | explicit `tau`, ordinary Euler `dt` |
| `serial_independent` | no | yes, after an explicit event step | explicit positive `dt` |
| `joint_bidirectional` | yes | yes | explicit `tau`, ordinary Euler `dt` |

`SerialEventGeometryFlow(joint_model=...)` is a view over the same model; it
adds no serial-only parameters.  A run may claim equal generation weights only
when the recorded state hashes and parameter counts agree.  The control
manifest reports `null` equality when no concrete model or budget evidence is
provided; a Boolean declaration in configuration is not evidence.

## First scientific matrix

The frozen first matrix is stored in
[`workflow_b_baseline_ladder_v0.1.json`](../../configs/runs/workflow_b_baseline_ladder_v0.1.json):

1. reactant-only strong rules;
2. conserved event plus independent geometry (`both_off`);
3. serial conserved event-to-geometry (`serial_independent`);
4. joint conserved event-geometry (`joint_bidirectional`).

Physics guidance is deferred until a viable generator exists.  The first
selection stage uses zero calculator calls; any terminal refinement is a later,
separately frozen matched-budget comparison.  The headline joint-coupling claim
requires the joint arm to exceed the serial arm on a frozen parent-group test
split under the same candidate and downstream calculator budgets.

## Current acceptance and evidence boundary

The software checks cover finite coordinate and parameter gradients, missing
labels masked before arithmetic, explicit time and reactant conditions,
padding, atom permutation, local bidirectional state dependence, complete
training checkpoint recovery, and finite Euler rollouts for all controls.
Synthetic velocity targets are deliberately nonzero and time dependent.

These checks establish runnable software only.  Real training still requires
sample-level paired event and TS-geometry labels admitted under issue #58.
SPICE energy/force labels cannot satisfy that requirement.

The committed reports are
[`workflow_b_software_acceptance_20260929.json`](../evidence/workflow_b_software_acceptance_20260929.json)
and
[`joint_flow_smoke_20260929_v3.json`](../evidence/joint_flow_smoke_20260929_v3.json).
Both were produced from clean source commit
`7a4fe88c436dbcde3bb50bbc5cc8d7bf6669166a`; the checkpoint itself remains a
local smoke artefact and is not published as a scientific model.

Run the bounded checks with:

```bash
PYTHONPATH=src:vendor/mechai_reusable \
  python scripts/validate_workflow_b.py --output /tmp/workflow-b-acceptance.json
PYTHONPATH=src:vendor/mechai_reusable \
  python scripts/joint_flow_smoke.py \
    --output /tmp/workflow-b-smoke.json \
    --checkpoint /tmp/workflow-b-smoke.pt
PYTHONPATH=src:vendor/mechai_reusable \
  python -m pytest -q tests/test_joint_flow.py tests/test_workflow_b_protocol.py
```

For committed evidence, run the acceptance command with `--require-clean`
before writing a tracked report.  Neither command uses a calculator.
