# Versioned reference certification service (W07 / #137)

`CertificationService` accepts one mapped, generated candidate and returns layered
observations and durable costs. `ReferenceBackend` extracts the numerical chain
from PR #119 at `e84e330390a519bf4ac12048b297403c09717646`; this is an explicitly
versioned PySCF/gpu4pyscf reference protocol, not a CP2K result or qualification.
All generator and scheduler arms must call the same service and frozen protocol.

## Input boundary

`CertificationCandidate` requires coordinates in Angstrom, atomic numbers, stable
atom identities, parent identity, reactant and generated product bond/electron
(BE) matrices, reactant automorphisms, charge, multiplicity, environment and source
provenance. BE includes diagonal nonbonding electrons and conserves the total
valence electron inventory. Matrices, geometry and mappings share the same fixed
atom pool. Unknown state/mapping/environment is rejected; there are no defaults.
The initial scope is 2–64 CHNO atoms, gas phase, neutral singlet, nonperiodic.

The adapter consumes no target products, reference TS, test labels, archived
energies or reference-selected environments. A generated proposed product BE is
an input hypothesis; only the actual optimized endpoints determine the observed
event. The archived replay panel is a numerical regression, not blind discovery
or Track-B admission. It does not change `ReactionQMRecord` quarantine.

## Frozen numerical chain

Protocol `xtbflow-reference-certification/1`: RKS wB97X/6-31G*, spherical basis,
grid 5, SCF tolerance 1e-10, at most 300 SCF cycles, no density fitting,
dispersion, solvent or free-energy correction. Each SCF, gradient and analytic
Hessian invocation consumes a durable token **before** entering the kernel.

Initial Hessian → geomeTRIC OptTS (200 iterations) → final gradient and Hessian →
mass-weighted internal modes → both IRC directions → endpoint minimization and
frequency/gradient checks → endpoint BE perception. Exact translation/rotation
span removal is retained. The TS has exactly one frequency below −30 cm⁻¹ and
no other negative mode. Maximum gradient is 1e-4 and RMS gradient 3e-5 Ha/bohr.
IRC allows 300 iterations per direction with trust/tmax 0.2; the same final TS
Hessian seeds both directions. Numerical thresholds and bounded internal retries
are included in backend identity. A frequency-gray TS can receive one VeryTight
retry; a minimum with a clear negative mode can receive one 0.10 Å mode displacement
and VeryTight optimization. Every retry shares the original call/time budget.

Compared with PR119, final gradients are now retained and explicitly checked,
and gray-zone endpoint frequencies no longer qualify. Raw TS/endpoint Hessians,
gradients, TS mode and convention, complete IRC coordinates/energies, geometry
optimization logs and endpoint graphs stay in the private raw directory. The
original starting coordinates and mapping remain in `request.json`.

## Feedback and downstream contract

`record.feedback` separates `stationary_point`, `first_order_saddle`,
`bidirectional_minima`, `planned_event_realized`, `observed_event_id` and
`observed_product_be`. It preserves the two actual endpoint BE matrices.

| Evidence status | Interpretation |
| --- | --- |
| `reference_ts_connected` | Full reference chain and actual event match the generated hypothesis. |
| `valid_alternative` | Full reference chain connects the input reactant to a different actual event. |
| `same_endpoint` | Both verified ends return to the input reactant; no new channel. |
| `reactant_mismatch` | Neither endpoint matches the declared input reactant. |
| `attempted_unresolved` | Missing or failing evidence; retain the completed lower layers and reason. |

#136/#139 may use these diagnostics to propose a new candidate. #142 may consume
both connected statuses, using the **observed** product and event; it must not
rewrite them from the planned event. A distinct geometry for the same parent,
observed event and protocol is extra evidence, not a second independent channel.
Event IDs preserve PR119's complete BE delta modulo reactant automorphisms.
The fixed mapped atom universe remains part of the parent contract.

## Budget, cache and interruption semantics

One service root is serialized with an OS lock. The stage has finite call, CPU,
GPU, storage, concurrency, retry and job-time limits. Each new attempt checks a
complete resource envelope before reserving calls. A fresh worker interpreter
runs the backend with bounded CPU affinity, CPU/file limits and captured logs;
the parent terminates it on wall timeout. Raw output size is checked every 0.1 s
and on completion. This is a monitored storage cap, **not** a filesystem quota;
transient overshoot remains possible and is recorded as a failed attempt.
Production jobs additionally require a finite Slurm allocation. Only this service
may write its ledger; unrelated writers do not share its process lock.

Cache identity includes mapped candidate content, explicit electronic/environment
state, provenance, protocol settings, package versions and installed RECORD
manifest hashes, GPU/CUDA build observations, and service/adapter/numerical/ledger
source hashes. Cosmetic candidate IDs do not force recalculation. Cached requests,
results and raw artifacts are checked against hashes before reuse. Installed
package RECORDs identify distribution builds; they are not a security attestation
against local tampering of installed binaries.

Result writing precedes settlement. Call reservations survive worker death and
report-write failure. Settlement uses deterministic token/event IDs and can resume
after partial failure without double charging. A completed result is never
silently recomputed. An interrupted attempt without a durable result is finalized
as `execution_interrupted`, with durable consumed calls and a conservative full
resource-envelope charge; this is labeled separately from measured usage.
Unsettled attempts for other keys are reconciled before new work. Retry requires
`retry=True` and stays within the stage cap; connected channels are always reused.

Execution states distinguish `not_started_budget_exhausted`, `not_started_error`,
`completed`, `execution_failed`, `execution_timeout`, `budget_exhausted` and
`execution_interrupted`. `report-failure.json` and `settlement.json` separately
record report-writing and settlement errors. If the filesystem cannot accept even
an error marker, the durable request/reservation still identifies an unresolved
interruption; no result is invented. Calls that entered a failing SCF are billed.
CPU time comes from child process accounting, GPU cost is allocated wall time,
and raw artifact sizes are recorded. Scheduler/setup/metadata overhead is reported
separately by the runner and must be included in future full-cost comparisons.

## Reference replay and limits

`scripts/replay_certification_panel.py` consumes the exact source plan and the
two hashed archived verdicts in `configs/validation/certification_replay_20261107.json`.
One prior matched event and one alternative event were deliberately selected;
their past costs were 1378 and 1048 GPU seconds. This justifies the frozen finite
stage: 2 GPU-hours, 16 CPU-core-hours, 6000 primitives, 1 GiB, one job, zero external
retries, at most 3300 seconds per candidate and 120 minutes of Slurm allocation.
No training allowance is created. Source commit and protocol/config/input hashes
must be recorded before execution, with exact commands in the dated evidence.

The runner reconstructs only whitelisted generated input fields, performs the new
chain, then compares TS energies/frequencies/aligned geometries, endpoint energies
and actual events against archived outputs. Tolerances were frozen before execution.
A second call must hit cache, consume zero new primitives and add no channel.
Raw coordinates stay on n2. Failure reports are retained. This panel establishes
numerical regression evidence only; it cannot establish discovery success,
reaction-family generalization, method superiority or solution chemistry.
