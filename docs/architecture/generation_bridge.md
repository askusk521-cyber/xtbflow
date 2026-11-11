# Research generator integration (W01 / #131)

The bridge consumes the pinned PR119 numerical generator at
`e84e330390a519bf4ac12048b297403c09717646` and three existing 60,000-step seed-0
EMA checkpoints: joint, independently trained event, and independently trained
geometry. Checkpoint and completed-training status hashes are both checked.
It does not copy or modify the historical source or weights.
`configs/research/generation_bridge_20261107.json` freezes hashes, input selection,
methods, clocks, seed, proposal counts and finite resource caps before execution.

## Input and process separation

`run_research_generation.py` reconstructs the existing admitted development
source, chooses the smallest training atom pool (ties by record ID) and builds
an explicit mapped reactant-only query. This selection never uses products,
TSs, reference energies or successful labels. The reactant's mapped SMILES/XYZ
audit is reused. Reactant-only Kekulization, formal charges and complete BE
electron counts are explicit; automorphisms preserve the declared graph and are
capped at 2048. The gas-phase environment is a new predeclared modeling condition,
not an assertion about the unresolved source geometry environment or a
matched-geometry multi-fidelity pair.

The worker is a fresh interpreter receiving only query ID, atomic numbers,
reactant coordinates/BE, charge and multiplicity. It checks that exact whitelist.
Parent/family provenance and mappings stay outside model input and later bind the
generated output to `CertificationCandidate`. No missing electronic state is
filled from the legacy `Query` defaults: validation occurs before worker launch.

The frozen trunk imports element constants from a legacy training batching
module that also imports the target cache. A small inference adapter supplies
only that exact CHNO vocabulary and padding count. Dataset classes are unavailable
in this interpreter. The numerical model, noise and rollout implementations stay
at the pinned revision. The worker records and rejects imports of the reference
catalogue or target-cache modules; it is never given their paths or label fields.

The optional `--reactants` input replaces source-cache preparation with an
explicit, bounded, hashed reactant packet under the same field contract.
`--config` selects a recorded bounded protocol; method, seed, dimension, time,
thread, storage and complete-forward envelopes are checked before launch.
`conditioned_geometry` additionally requires `--conditioned-events`, keyed by
query ID, containing earlier generated BE hypotheses with the same electron
inventory. It runs only the trained geometry model and checks that the fixed
event remains unchanged. Reference products/TS coordinates are not model inputs.
The live action adapter supplies these inputs from verified source records;
see [feedback agent](feedback_agent.md). Runner and worker lifetime are bound to
their parents on Linux, so controller failure cannot leave a model worker running.

## Methods and costs

Joint generation uses the trained dual-clock model (`event_lead2`, 50 steps).
Serial generation uses the independently trained event model for 50 steps,
decodes its event, and then uses the independently trained geometry model for
50 steps with that fixed event. An invalid event prevents geometry execution and
remains an explicit failed proposal. Both methods start from the same addressed
reactant-only noise, using the same query/proposal/seed namespace.

Every model forward is journaled and fsynced before invocation. Failed/unfinished
forwards remain consumed. The frozen one-query, one-proposal-per-method stage has
at most 150 per-system forward evaluations, one GPU, 600 worker wall seconds,
eight CPU threads, 64 MiB raw output, no retries, training or QC calls. A 12-minute
Slurm allocation bounds setup overhead. Stage caps are 0.2 GPU-hours, 1.6 CPU-core
hours and 0.25 GiB. The current `RunLedger` records measured aggregate cost with
`system_evaluations` denoting generator forwards and `calculator_calls=0`;
the raw journal provides role/candidate detail. This is a separate generation
phase from physical certification and must be summed with that phase for end-to-end
comparisons. Same proposal count does not mean same compute; no advantage is claimed.

`raw/certification-plan.json` contains only finite decoded candidates that pass
the common fixed-pool/state/environment/BE contract. It preserves the generated
product as a hypothesis and never invents observed endpoints. The protocol,
checkpoint, query, journal and output hashes link the candidate to its cost.
Generation failures, timeout and settlement failure are distinct; existing output
directories are never reused silently. Interrupted generation requires explicit
reconciliation of its retained call journal before a separately budgeted replay.

## Execution and claim limits

Set `XTBFlow_BRIDGE_SOURCE`, `XTBFlow_BRIDGE_COMMIT`, the three asset paths and
`NEW_RUN_OUTPUT` to the recorded clean named checkout, exact commit, n2 assets
and a new output directory. Within a 12-minute Slurm `main` allocation with
`gpu:pro6000:1`, eight CPUs and 64 GiB RAM, run:

```bash
cd "$XTBFlow_BRIDGE_SOURCE"
test "$(git rev-parse HEAD)" = "$XTBFlow_BRIDGE_COMMIT"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate xtbflow
timeout 700 python scripts/run_research_generation.py \
  --cache "$XTBFlow_DFT_DA_CACHE" --research-source "$XTBFlow_RESEARCH_SOURCE" \
  --model-root "$XTBFlow_MODEL_ROOT" --out "$NEW_RUN_OUTPUT"
```

Use `main`, `gpu:pro6000:1`, eight CPUs, finite 12-minute wall time and a new output
directory. Record the exact bridge/research commits, environment, Slurm ID and
hashes in dated evidence. The resulting certification plan is private on n2.
Reference calculations use `certify_generated_candidates.py` and the separately
frozen `configs/validation/generated_candidates_20261107.json`. This runner
validates the completed and settled generation report, every artifact, call
journal, plan and candidate identities before making any reference call. Every
eligible candidate is included in generation order. Cache replay must consume
zero calls and add no channel. Per-candidate chemical evidence remains separate
from `pipeline_completed`; zero certified channels is a valid recorded outcome.

In a separate 120-minute `main`, `gpu:pro6000:1`, eight-CPU, 64-GiB allocation,
with `GENERATION_OUTPUT` pointing to the completed prior output and
`NEW_CERTIFICATION_OUTPUT` naming a fresh directory:

```bash
cd "$XTBFlow_BRIDGE_SOURCE"
test "$(git rev-parse HEAD)" = "$XTBFlow_BRIDGE_COMMIT"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate xtbflow-qc
timeout 7000 python scripts/certify_generated_candidates.py \
  --generation "$GENERATION_OUTPUT" --out "$NEW_CERTIFICATION_OUTPUT"
```

This phase allows at most two candidates, 3300 seconds/3000 primitives each,
6000 primitives, two GPU-hours, 16 CPU-core-hours and one GiB in total, with no
external retry. The recorded 9/13-atom reference-chain costs justify this finite
integration allowance; they do not guarantee success for the predetermined
16-atom input. Raw geometries and graph matrices remain on n2. Reference results
do not promote a development input or erase unresolved historical overlap.

This is a functional integration of existing learned assets on a declared
development input, not a trained-model comparison, independent-reactant benchmark,
physical validation or generalization claim. Historical weight/source overlap
with the Diels–Alder chemistry is not resolved by this run. Strong serial
backtracking, matched total budgets, training comparisons and confirmation remain
the responsibility of the later work packages.
