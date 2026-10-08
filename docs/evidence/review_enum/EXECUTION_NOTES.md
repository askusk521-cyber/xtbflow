# Execution deviations and bounded autonomous decisions

All analyses are exploratory. The owner explicitly amended the handoff to permit direct CPU execution and requested autonomous progress with records; GPU computations remain Slurm-only. No merge, DFT, gated-license acceptance or existing environment modification was authorized.

## Exact domain and selection

The training-only audit chose the first >=95%-coverage domain: b3f3 covers 2547/2603 unique training events. Exact legal-event counts on the five frozen training parents were 65130, 15888, 25953, 35696 and 31670; mean 34867.4, below the frozen feasibility threshold. Therefore b3f3 was selected, not relaxed after seeing evaluation performance.

The initial evaluation launcher mistakenly used split_manifest.screen_reserve (344 IDs). An explicit scope check caught this before any inference; those processes were stopped and the entire pilot directory `enum-screen-7c28436` excluded. Correct main runs use the 342 IDs in formal/freeze.json at source9403d6c. Development uses its 62 frozen IDs. This correction implements the handoff definition rather than changing it.

## Performance-only changes

- Capacity-aware recursive formation enumeration avoids impossible saturated atoms.
- Charged-valence prefilters were tested against the installed RDKit strict sanitizer and Lewis hydrogen bound. Initial over-tight assumptions for positive carbon/hydrogen were caught by unit tests and corrected before those optimized versions were used for acquisition.
- Atom/degree diagonal options are cached without changing permissible edits.
- Reactant automorphisms quotient broken-bond multisets before formation. This is exact because the full enumeration domain is invariant under the same automorphisms used by canonical_event. A symmetry-free versus symmetry-quotiented tiny-domain test passes; the real training parent 027539295e76c5842421 retained exactly 35696 events before/after pruning.
- Completed per-parent results and recovery checkpoints were resumed, not replaced. Source provenance records the original and resumed commits. No model/scorer/metric was tuned on dev or screen.

## Interrupted runs and resource accounting

Slurm CPU feasibility job2691 hit its ten-minute limit. CPU test job2692 completed. Several owned CPU feasibility/enumeration processes were explicitly stopped to deploy tested performance fixes; their costs remain part of the resource ledger rather than being hidden. No other session's processes or jobs were canceled.

GPU scoring job2700 hit its ten-minute limit after writing completed per-parent score archives. Job2707 resumed from those archives through Slurm with a twenty-minute cap. Other session job2702 was left untouched; the queue was empty before job2707 submission. Future scoring passes must recheck the queue and obey dependencies.

CPU acquisitions are one OpenMP/BLAS thread each and every process is capped at two hours. Scoring batches remain fixed at256 to avoid numerical changes from batch-size tuning. Formal output directories are `enum-screen-9403d6c`, `enum-dev-7c28436`, and `training-b3-6f8ab54` beneath the owned run root. File directory names identify initial acquisition commits; the manifest lists resumed source commits as well.

This file is an execution record, not proof that enumeration, training recovery, inference or final validation has finished. Final evidence must include exact completeness checks and resource totals.
