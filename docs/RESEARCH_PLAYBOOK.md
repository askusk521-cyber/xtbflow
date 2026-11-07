# XTBFlow Research Playbook

This document fixes the research procedure for issue #129. It is a governance
contract, not a chemistry result. `AGENTS.md` contains the repository rules that
apply to every change; this playbook defines how a scientific comparison is
registered, run, interpreted, and handed off.

## 1. Roles and claims

The method is decomposed into four roles:

- **J — joint event/geometry kernel:** event hypotheses and 3-D candidates can
  revise one another.
- **P — physics feedback:** measured energy, forces, curvature, modes, or other
  permitted physical signals enter candidate generation.
- **C — calculation-decision Agent:** under a fixed budget, selects the next
  family, geometry, retry, certification, or exploration action.
- **V — common certification:** applies the same TS, mode, bidirectional IRC,
  endpoint, identity, and accounting rules to every method arm.

A report must state which role a result tests. A combined result is not assigned
to J, P, or C until the corresponding control is present.

Evidence levels are separate:

1. `software_available`: executable code and contracts pass checks.
2. `real_closed_loop`: real reactants, physical calls, candidates,
   certification, and accounting can be replayed.
3. `development_comparison`: frozen inputs, controls, protocols, and budgets
   support a reproducible comparison on the development panel.
4. `independent_confirmation`: a new source or reaction family, untouched by
   training, selection, tuning, or method-arm choice, supports the claim after
   the protocol is frozen.

A lower level never substitutes for a higher level.

## 2. Pre-registration

Before an experiment starts, record one immutable manifest containing:

- source commit, model/checkpoint hashes, data split and overlap audit;
- allowed input packet, charge, multiplicity, units, atom mapping, parent and
  family identity, and quarantine rules;
- physical calculator, protocol hash, executable and environment versions;
- method arms, random settings, candidate and stopping rules;
- the full operational budget vector;
- primary metric, mechanism diagnostics, failure classes, and planned analysis;
- retry and diagnostic-attempt policy, including what cannot be used for tuning;
- output paths and the run/attempt identity.

The primary metric is:

```text
unique_strict_certified_channels
per independent parent, method arm, and fixed operational budget
```

Also report the fraction of parents with at least one channel, first-certification
cost, discovery-versus-cost curves, failures, retries, duplicates, censoring,
and unknown cost. Candidate count, event-change rate, physical-window size, and
model loss are mechanism diagnostics only.

## 3. Budget and accounting

Budgets are a vector, not a post-hoc weighted scalar:

```text
max_model_forward_evaluations
max_candidates
max_calculator_primitives
max_gpu_hours
max_cpu_core_hours
max_wall_seconds
max_retries
max_diagnostic_attempts
```

All arms use the same declared ceilings and record actual model, QC, CPU, GPU,
wall, storage, failure, timeout, retry, and cache values. Failed and unresolved
attempts retain their consumed cost. A cache replay has zero new cost, cannot
create a second physical convergence, and cannot increase channel count.

A 526-call historical run may inform capacity planning, but it is not the next
comparison budget until a new panel manifest freezes that limit.

## 4. Certification and channel identity

A strict channel requires all of the following for the same candidate and
protocol:

1. TS stationary-point or gradient-convergence evidence;
2. one clearly identified target imaginary mode;
3. successful IRC in both directions;
4. endpoint charge, spin, atom-map, and endpoint-identity contract;
5. a recorded actual `observed_event`;
6. complete evidence lineage and ledger settlement.

`planned_event` is a generation intent and does not certify a channel.

A `valid_alternative` is counted only when it passes the same certification and
has a different observed event or endpoint connectivity. The deduplication key
is:

```text
parent_identity
+ normalized_observed_event
+ endpoint_connectivity
+ stereochemical_or_state_identity
```

Conformers, seeds, repeated certifications, attempt IDs, and protocol versions
are lineage or comparison strata, not additional chemical channels.

## 5. Failure classification and retries

`PROTOCOL_FAILURE` with `attempted_unresolved` is an unresolved certification or
execution state, not a chemical negative. The structured record must contain:

```text
failure_classification: certification_unresolved
failure_stage: <specific stage or unknown>
failure_reason: <specific reason or unknown>
chemical_negative: false
strict_channel_increment: 0
diagnostic_retry_allowed: true
diagnostic_retry_requires_new_attempt: true
excluded_from_same_panel_tuning: true
```

It remains in attempted counts, failure denominators, and the actual ledger.
`pipeline_completed` only means that the bounded service settled its execution;
it does not mean the candidate was chemically certified. Missing evidence stays
`unknown`; it is never inferred from a terminal status.

A diagnostic retry must use a new `attempt_id`, independent budget, and preserved
original evidence. It cannot change the current panel's model, threshold,
strategy, stopping rule, or parent selection. If a global protocol defect is
found, every affected arm is rerun under the repaired protocol with a new
experiment identity.

## 6. W08 generation comparison

After one real feedback loop is complete, freeze a 2×2 comparison:

| Generation | No in-generation P / terminal physical handling | In-generation P |
|---|---|---|
| Joint J | J + terminal/no-in-generation P | J + in-generation P |
| Strong serial | strong serial/backtracking + terminal/no-in-generation P | strong serial/backtracking + in-generation P |

The strong serial control gets reasonable restart, resampling, backtracking,
candidate filtering, and tuning opportunities. All four arms share the same
reactant information, tool set, randomization policy, certification service, and
budget vector. This design estimates J, P, and J×P interaction rather than
assigning a combined gain to a single component.

The next panel must be frozen from the audited #146 development-test split:
45 records in 9 parent/family groups from the same Diels–Alder archive. Apply only declared
input filters (CHNO atoms, legal charge/multiplicity, complete reactant-only
packet and mapping, complete identity/protocol fields, and no quarantine).
Publish the full eligible pool and its manifest hash, sort deterministically by
`manifest_sha256 + parent_id`, select five parents, and exclude `ID63623` from
4101. If fewer than five are eligible, use all and label the run `pilot`.
Never replace a parent after seeing method results. This panel is a same-source,
family-holdout **development comparison**, not independent confirmation.

## 7. W09/W10 decision-policy comparison

Only after a real feedback-to-certification loop and the W08 generation controls,
compare:

- fixed scheduling;
- a simple feedback heuristic;
- a feedback-driven C Agent.

The three policies use the same generator, tools, inputs, and vector budget. The
Agent may read only feedback already produced during the active run or a faithful
stepwise replay. Each action records the visible-state hash, action and reason,
target candidate, estimated and actual cost, later certification result, and
whether it created a new, duplicate, failed, or unresolved outcome. C's result
must be reported separately from J and P.

## 8. Dual-clock analysis

Register two levels of outcomes:

**Mechanism indicators**

- whether feedback arrived while the event was still editable;
- intervention-window fraction;
- editable steps before event freeze;
- event-change fraction after feedback.

**Primary scientific outcomes**

- strict certified channels per parent and arm;
- fraction of parents with at least one strict channel;
- first strict-certification cost and full ledger.

A larger window or a changed event demonstrates a timing mechanism. It does not
by itself demonstrate improved reaction discovery.

## 9. Change record and handoff templates

Every new module has a short change record:

```text
Observed bottleneck:
Existing evidence:
Capability added:
Simplest alternative:
Finite experiment budget:
Close/replace condition:
Adoption criterion:
```

Every run handoff records:

```text
Completed:
Not completed:
Source/config/protocol/input hashes:
Run and attempt IDs:
Evidence files and report hashes:
Actual model/QC/CPU/GPU/wall/storage cost:
Failure and unresolved states:
Remaining budget:
Claim limits:
Next executable action:
```

Resuming an experiment keeps its run identity and appends an attempt. A new
experiment receives a new identity. Documentation progress never closes a
scientific issue or upgrades an evidence level.
