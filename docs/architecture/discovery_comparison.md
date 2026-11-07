# Discovery comparison and cost accounting (W08 / #138)

The semantic matrix in `configs/evaluation/comparison_matrix_20261107.json`
separates three questions: event/geometry coupling, physical-feedback timing,
and adaptive decisions (owned by W10). It retains strong serial backtracking,
one-way coupling, common reactant-only physics search and known-product search
as explicit controls. A required but unimplemented method remains unavailable;
it is never converted into a failed chemical trial. Independent native branches
must receive a declared, billed completion/search adapter.

`comparison_matrix.validate_matrix` verifies dispatch paths and matched input
permissions for primary pairs. `functional_dispatch` exposes the existing frozen
joint/independently-trained-serial batch and rejects unsupported selections.
That batch is runnable through the generation bridge, but is **not** an
equal-total-budget experiment. The full matrix remains incomplete until strong
controls, independent per-method execution, reasonable development tuning and
the physical feedback methods are implemented. Shared-weight masks are mechanism
diagnostics, not separately trained baselines.

## Billing and authoritative observations

`DiscoveryAttempt` and `DiscoveryTrial` require every runtime cost field to be
explicit. `null` means unknown; omitted fields are rejected. Runtime axes include
raw proposals, joint/event/geometry forwards, energy, force, combined energy-force,
HVP and Hessian calls, reference primitives/attempts, retries, cache hits, storage
bytes and wall/CPU/GPU seconds. Training and tuning CPU/GPU costs are separate
required fields; unknown historical costs prevent a full-cost claim.

`reference_primitive_calls` is an aggregate, not an extra primitive to add to
energy/force/Hessian counters. Under the current reference backend, each SCF is
one energy primitive, each gradient one force primitive, and each Hessian one
Hessian primitive. Started but unfinished calls remain charged. No exchange rate
between candidate counts, model calls, physics calls and hardware time is invented.
Report candidate-axis and resource-axis results separately. GPU cost represents
the stated allocated-worker wall time; process CPU time differs from reserved
CPU-core time. Include scheduler, setup, model loading and cache-lookup overhead
in a complete experiment, with a frozen attribution rule for shared work.

`certified_trace.read_reference_attempt` verifies the stored request, result,
all raw artifacts, settlement, ledger event and committed token. It recomputes
layered feedback from retained evidence, and reconciles pre-call journal entries.
Only connected reference statuses supply observed channel IDs; predictions and
proxy event hits do not create certificates. A valid alternative counts as its
actual endpoint event, not the originally planned one. The adapter emits a
**reference-attempt-only** cost vector and lists excluded stages; its zeros for
generator counters do not imply zero full-pipeline generation cost.

Execution failures, timeouts, interrupted attempts, budget exhaustion and
not-started states stay distinct. Conservative interrupted resource bounds retain
their basis in adapter provenance. A missing durable result/settlement cannot
enter certified statistics. A later cache hit adds no physical primitive or
independent channel; its actual lookup resources belong to a separate attempt.

## Discovery metrics and grouping

Within a parent/seed trial, unique channels use actual event identity under one
declared reference protocol. Multiple geometries or cache reads of that event
count once. Discovery curves use cumulative costs including unsuccessful attempts;
a certificate arriving after **any** finite budget cap is excluded from in-budget
discovery while its costs and out-of-budget observation remain recorded.

An unknown capped cost makes the corresponding comparison unresolved. Failed,
cancelled and not-started cells remain explicit planned cells. The first
in-budget certificate records its entire cumulative cost vector. If none exists,
the first-certification observation is right-censored with observed spent cost,
not assigned an artificial zero-time success. Native incomplete outputs remain
distinct from chemical failures. Only terminal trials enter terminal comparisons.

`paired_comparison` requires identical parent/seed cells, input identity,
information condition, protocol, complete budget vector and group declarations.
Missing cells are errors; callers must provide their actual failed/not-started
state. A known-product method cannot silently enter a reactant-only pair.
Seed differences are averaged within each parent before whole independent groups
are resampled. Family grouping requires real family labels; formula groups cannot
be relabeled as families. Fewer than two independent groups yields descriptive
results with no interval. Unknown paired costs suppress the estimate rather than
dropping those cells.

The prospective development defaults are seeds 0/1/2, 2000 bootstrap draws,
bootstrap seed 138, a two-sided 95% percentile interval and a minimum practical
gain of 0.1 certified channel per parent at the same runtime budget. These values
do not retrofit an independent-confirmation claim onto old exploratory results.
The complete finite batch, inputs and per-method budgets must be frozen before
execution; no significance-driven extension. W13 owns independent confirmation.

## Historical replay and real evidence audit

`scripts/replay_comparison_assets.py` replays PR113 and PR120 at the exact heads
in the W01 asset registry. It verifies all archived enumeration files, the
physical acquisition manifest/outputs, scorer inputs and additionally pinned
catalogue/split/freeze/stream snapshots before calling the original analyzers.
Their hard-coded stream locator is checked explicitly on n2. Both clean research
checkouts stay unchanged. Original per-k censoring, formula-cluster intervals,
label use and metric definitions are preserved, including exclusions; they are
not substituted for the new failure-inclusive discovery statistics.

The replay additionally reconstructs the two settled job-4096 reference attempts,
including the interrupted case, from original raw artifacts and the common
ledger. This demonstrates real cost reconstruction and certification separation,
not a paired method experiment. Historical candidate-axis proxy results and
observed reference costs are reported in separate fields. Original model, raw
coordinates, acquired energies and full archived analyses remain on n2; the
repository contains hashes, summaries and claim limits only.

Freeze the clean integration source as `XTBFlow_COMPARISON_COMMIT`; set
`XTBFlow_COMPARISON_SOURCE`, both clean pinned research checkout paths,
`XTBFlow_RUN_ROOT`, `XTBFlow_ORIGINAL_PANEL` and a new output path. Run via ssh n2:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate xtbflow
cd "$XTBFlow_COMPARISON_SOURCE"
test "$(git rev-parse HEAD)" = "$XTBFlow_COMPARISON_COMMIT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --partition=main --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G \
  --time=00:12:00 timeout 700 python scripts/replay_comparison_assets.py \
  --enum-source "$XTBFlow_ENUM_SOURCE" --thermo-source "$XTBFlow_THERMO_SOURCE" \
  --runs-root "$XTBFlow_RUN_ROOT" --reference-panel "$XTBFlow_ORIGINAL_PANEL" \
  --out "$XTBFlow_COMPARISON_OUTPUT"
```

The script caps parsing/replay at 600 wall seconds, 2 GiB hashed input and 64 MiB
output, with one CPU thread, no GPU and zero new calculator calls. The outer
12-minute allocation bounds interpreter/setup/reporting overhead. Each analyzer
also has CPU/file limits. Every output directory is fresh; partial and failed
reports remain. Compare output hashes with the original artifacts, and record
actual environment/version differences if exact replay fails.

本入口把旧报告的参考事件命中率与真实参考认证分开，避免把候选数当作完整费用。
新的统计会保留失败费用、未执行母组和未知费用，并按真实端点事件去重。
强串联回溯、流内反馈、同总预算的新配对实验和独立确认仍须完成；本次重放不能代替这些交付。
