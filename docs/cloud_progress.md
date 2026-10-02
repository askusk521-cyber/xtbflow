# Cloud-first progress slice

This slice makes the cloud-only part of the roadmap executable without claiming
that a quantum-chemistry calculation has run.

## Task routing

`xtbflow.training.task_routing` turns one provenance-bearing source row into
zero or more explicit task views:

- `event_only`: event labels only;
- `geometry_only`: geometry labels only;
- `energy_force`: energy and force labels only when the adapter attests that
  they belong to the same geometry;
- `paired_joint`: event and geometry labels only when the adapter attests that
  they belong to the same primitive step.

The availability mask is authoritative. A row with event and geometry labels
but no pairing attestation is split into two independent views; it is never
randomly paired. Missing labels are not represented by synthetic zero targets.

## Blind evaluation input

`xtbflow.evaluation.blind` snapshots a reactant-only input, rejects
product/TS/reference-derived fields recursively, and records a reproducible
fingerprint. The fingerprint can be compared across controls to show that two
methods received the same deployment-visible input.

These checks cover routing and information-flow contracts. They do not qualify
an adapter, a calculator, a TS, a force field, or a chemical mechanism.
Those claims still require the real data and physics gates in the project plan.

## Task-level availability audit

`xtbflow.data.availability` and `scripts/audit_task_availability.py` produce a
stable JSON audit from a public-record JSONL manifest. The report separates
source-declared labels from formally admitted training rows, counts identified
parent reactions, and records quarantine reasons. It requires explicit
same-geometry and event/geometry identity bits before counting `energy_force` or
`paired_joint`; it does not infer either relation from co-occurring fields.
