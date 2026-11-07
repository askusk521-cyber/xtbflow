# Trained generation bridge — 2026-11-07, job 4100

**Generation completed:** the existing joint model produced one candidate and
the independently trained event-then-geometry models produced one candidate for
the same reactant input. Both candidates passed the explicit mapped atom-pool
contract and were handed to reference job 4101. This report records generation
only; neither generated geometry is certified by this result.

Execution source `e07ac8512fa987e0fad2e37b14c576c534a7df87` contains PR #147's
generation bridge and PR #148's monitor repair. The clean legacy numerical source
was `e84e330390a519bf4ac12048b297403c09717646`, with nine source files checked.
Host `node2`, Conda `xtbflow`, Python 3.10.19, PyTorch 2.10.0+cu128,
NumPy 2.2.6, RDKit 2024.3.3, CUDA 12.8 and RTX PRO 6000 Blackwell were observed.
The [full report](trained_generation_20261107.json) records all model checkpoint,
training-status, protocol and raw artifact hashes. The [artifact audit](trained_generation_20261107_artifacts.json)
rechecked all six artifacts, both candidate identities, the model-call journal,
shared ledger and frozen certification plan with zero calculator calls.

Input `dft-da:R0`, parent `ID63623`, 16 atoms, was chosen by smallest mapped atom
pool then record ID among 147 admitted training rows. Its input SHA256 is
`7ffaa060a33da76edae9870ca4be2c6a43c13865a5c71c464ba7ad244d30535c`.
No product, TS or energy selected the input. A new gas-phase condition was
declared explicitly; the original geometry's environment remains unknown.

| Method | Model forwards | Candidate loop wall (seconds) | Result |
| --- | ---: | ---: | --- |
| Joint, event-lead2 dual clock | 50 | 1.006543 | generated |
| Independent event then fixed-event geometry | 50 + 50 | 1.049363 | generated |

Total: **150 forwards, 150 kernel invocations, zero unfinished calls**, zero
training steps and zero new calculator calls. Runner wall including worker setup
was 7.231239 s, measured process CPU 8.250481 s, retained raw bytes 144,567.
Slurm finished `COMPLETED`, exit `0:0`, in eight seconds with one GPU and eight
CPUs allocated. Full method cost cannot be inferred from the candidate-loop
timings; model loading, setup and subsequent reference work are separate costs.
Equal proposal count here is not equal total computation.

Config SHA256: `50d7d9c80cb0dba82e4e797e5cc2456e0ca14c21646b37d5a3dc31c3eeeac11e`.
Report SHA256: `482ae6c426d71b1218cb104f7530b1d6c2755ddc8bf2bd309a58f6a8783acaa6`.
Plan SHA256: `baf51f5fbc4961fb3f80aa71363edc8ec35fe45d535dd483e629031d80264dd6`.
Job script SHA256: `d8ebef95cceb60dfc4dc14e1a584c7c3c86b41f61e645f7cbed7331117950eea`.
Frozen joint/event/geometry checkpoints were all step 60000, seed 0, with their
common split hash recorded in the report. No retraining or weight modification
occurred. No forbidden target-cache module was imported by the isolated worker.

## Bounded execution

Job 4100 followed successful job 4099 through `afterok`. The preserved script
`xtbflow-generation-e07ac85-job.sh` requested main, gpu:pro6000:1, eight CPUs,
64 GiB, 12 minutes and no requeue. The worker was bounded to 600 seconds,
150 forwards and 64 MiB; the stage capped GPU at 0.2 hours, CPU at 1.6 core-hours
and retained storage at 0.25 GiB. The command inside that allocation was:

```bash
ssh n2 '
  cd "$HOME/xtbflow-runs/w137fix-20261107-e07ac85/source"
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate xtbflow
  test "$(git rev-parse HEAD)" = e07ac8512fa987e0fad2e37b14c576c534a7df87
  export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
  export PYTHONPATH=src:vendor/mechai_reusable PYTHONDONTWRITEBYTECODE=1
  timeout 700 python scripts/run_research_generation.py \
    --research-source "$HOME/xtbflow-runs/followup-20261009/source-e84e330" \
    --model-root "$HOME/xtbflow-runs/v1a-20261007" \
    --cache "$HOME/.cache/xtbflow/dft-da-full-29118509-v1" \
    --out "$HOME/xtbflow-runs/w137fix-20261107-e07ac85/generation"
'
```

This records the original invocation. Reproduction must use a new clean named
source/output directory and finite Slurm allocation; the output already exists
and is immutable. Full software checks at the execution source passed 328
tests/68 subtests, 56 reusable tests and bootstrap 354 files. The generation
ledger is settled, but job 4101's separate reference ledger is still active at
the time of this report. Its eventual evidence must be published separately.

The development endpoint is not an independently prepared confirmatory input.
Historical model overlap is unresolved. This is runnable integration of existing
research assets, not independent generalization, a method ranking, a new training
result or a chemical mechanism claim. Weights, raw inputs and generated
coordinates remain on n2. The broader W01 asset matrix remains incomplete.
