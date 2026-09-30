# Diels--Alder generation evaluation

This workflow evaluates the checkpoints from the bounded real-data training
matrix without giving the generator an event or transition-state label.  The
three stages are intentionally separate:

1. `export_dft_da_reactants.py` audits the public archive and writes a
   target-free JSONL manifest.  It may read product/TS files as part of that
   source audit, but those fields are not written to the manifest.
2. `run_reactant_generation.py` reads only that manifest and a frozen
   checkpoint.  It integrates from the centered reactant geometry and reactant
   bond/electron state, decodes the endpoint, and records every attempted and
   rejected trajectory.  The current deterministic vector field produces at
   most one candidate per record; `candidate_cap` is an upper bound, not a
   promise to produce 64 candidates.
3. `score_reactant_generation.py` joins the frozen candidates to product/TS
   labels and reports event exact-match, event bond MSE, and aligned geometry
   endpoint MSE.  It also reports a zero-change baseline, the existing
   reactant-only rule baseline, record-weighted means, parent-equal means, and
   paired parent bootstrap intervals.

The generated evidence contains `target_fields_read=false`.  The scoring
report is label-assisted by design and is therefore a development diagnostic;
it does not certify the geometry level of the XYZ files, physical refinement,
or reactant-side discovery quality.  A candidate fingerprint is computed before
scoring and must remain unchanged when reference labels are replaced.

The exported JSONL uses schema `xtbflow-reactant-input/v1`. Each row contains
`record_id`, `parent_reaction_id`, `family_id`, ordered `symbols`, audited
`map_ids`, `reactant_coordinates_angstrom`, the symmetric `reactant_bonds`
matrix, and explicit `charge` and `multiplicity`. The exporter derives these
fields from the audited source row; hand-built rows must preserve the same atom
order and map IDs. Product bonds, event edits, TS coordinates, reference
energies, and target labels are rejected recursively by the input firewall.

## Reproducible n2 invocation

Run from a clean checkout of branch `codex/generation-eval-20260930` (or record
the exact commit substituted below).  Set `DA_CACHE` to the private n2 cache,
`DA_CHECKPOINT` to one existing checkpoint, and `RUN_ROOT` to a private,
single-use n2 evidence directory.  The cache, checkpoint, coordinates and
candidate bytes stay outside the repository.  Each output name is single-use
and must not be reused.

```bash
cd /home/lhshen/xtbflow-rework/generation-eval
git rev-parse HEAD
source ~/miniconda3/etc/profile.d/conda.sh
conda activate xtbflow
python --version
python -c 'import torch; print(torch.__version__)'
python scripts/export_dft_da_reactants.py \
  --cache-root "$DA_CACHE" \
  --output "$RUN_ROOT/dft_da_reactants_test_20260930.jsonl" \
  --split test
python scripts/run_reactant_generation.py \
  --inputs "$RUN_ROOT/dft_da_reactants_test_20260930.jsonl" \
  --checkpoint "$DA_CHECKPOINT" \
  --runtime-config configs/models/joint_flow_runtime_v0.1.json \
  --output "$RUN_ROOT/dft_da_generation_seed11_joint_20260930.json" \
  --mode joint_bidirectional \
  --steps 16 \
  --candidate-cap 1 \
  --device cuda
python scripts/score_reactant_generation.py \
  --cache-root "$DA_CACHE" \
  --generated "$RUN_ROOT/dft_da_generation_seed11_joint_20260930.json" \
  --output "$RUN_ROOT/dft_da_generation_score_seed11_joint_20260930.json" \
  --split test \
  --candidate-cap 64 \
  --bootstrap-samples 2000 \
  --bootstrap-seed 20260930
```

The generation stage has a one-job, one-GPU, 10-minute wall budget and performs
no quantum-chemistry calculator calls.  The source audit and post-hoc scorer are
CPU-only, each with a 10-minute wall budget and no calculator calls.  Record
the actual host, environment versions, checkpoint SHA-256, source archive and
CSV hashes, commit, output hashes, and any failed stage in the evidence report.
Repeat the generation stage for all 15 existing checkpoints only after the
single-checkpoint contract and label-replacement test pass; do not start a new
training matrix from this workflow.
