# Reference certification replay — 2026-11-07, job 4096

**Panel status: `replay_failed`. One case passed; the second was interrupted by
a storage-monitor race after a valid first-order saddle.** Both results and all
467 consumed primitives were settled and retained. The failure is not a chemical
rejection or an unexecuted calculation. No numerical thresholds were changed.

Source: `931d81f53a85669e137600c62b343120ff0de3d0`, branch
`codex/w137-certification-service-20261107`, PR #145 based on #144. Host `node2`,
Conda `xtbflow-qc`, Python 3.11.17. The [machine-readable report](reference_certification_20261107.json)
contains all package/build/protocol hashes; the [artifact audit](reference_certification_20261107_artifacts.json)
verifies requests, results, raw artifacts, primitive journals and settlements
without new calculator calls. Raw coordinates and logs remain on n2.

| Archived item | Evidence | Primitive calls | Measured wall / CPU seconds |
| --- | --- | ---: | ---: |
| `08abfd11be911cc5` | `valid_alternative`, complete chain, numeric regression passed | 202 | 1077.873 / 1174.847 |
| `98a2bcd101825aa9` | `attempted_unresolved`, execution failed during IRC | 265 | 1300.684 / 1471.143 |

The first case preserved observed event `8382f00b0f4144f3b208275d`, distinct from
planned event `554a88f8a1149e46f48966fd`. Relative to the archived result:
TS energy difference 2.274e-13 Ha; maximum endpoint-energy difference 1.705e-13 Ha;
maximum frequency difference 3.336e-8 cm⁻¹; aligned TS RMSD 1.721e-11 Å.
Both immediate cache replays returned identical records, zero new primitives
and no additional channel. Exactly **one** unique certified channel was recorded.

The second case retained a stationary point and first-order saddle, but no
completed endpoint identity. The monitor scanned a `.writing-*` temporary file
while an atomic journal replacement removed it. `_directory_bytes` performed
separate existence and `stat()` calls, so `FileNotFoundError` escaped and the
service terminated the worker. Its 132 SCFs, 131 gradients and two Hessians remain
charged; no pending call reservation remains. A deterministic software regression
reproduced this failure in the original source. A follow-up must tolerate only
vanished entries, preserve other filesystem errors, and use a separately frozen
retry budget and fresh execution directory. Jobs 4097/4098 were cancelled before
execution after their success dependency could no longer be satisfied.

Service totals: 0.660710289 allocated GPU-hours, 0.734997171 measured process
CPU-core-hours, 1,047,238 raw bytes. Runner wall time 2379.425 s; Slurm wall time
2380 s, `FAILED`, exit `1:0`. The eight-CPU allocation occupied 5.288889 core-hours;
this differs from measured CPU use. Scheduler/setup overhead is not silently
treated as zero. Limits were 2 GPU-hours, 16 CPU-core-hours, 6000 primitives,
1 GiB, two candidates, no external retry, 3300 seconds/3000 primitives per case.

## Identities and execution

- Configuration SHA256: `7e932c5f60a8a74374029535f5087946b2d1df3f84c54b0dc07a8b4a5eecd4d3`.
- Protocol SHA256: `637e18f276d78daa7b9ed5155139dae6cb93943a9a1b5ad03e3516ff574e14d6`.
- Source plan SHA256: `0993c387fba18cd7761542099b1acb8146b9d78f2293f068d9ea872fe5d50fc8`.
- Final report SHA256: `0365ceceaae80a5f2bc724dc38b30ff080c17902814c6212a381c9086f5dac9b`.
- Job script SHA256: `8e847576675b2b285039288c8fdfad30040a2c71742df6ae0e40ad2f56ad03de`.

The job ran from the fresh named `w137-20261107-931d81f` execution directory.
The preserved job script activated `xtbflow-qc`, checked the exact clean source,
set OMP/OpenBLAS/MKL to eight threads, verified bootstrap, and executed:

```bash
ssh n2 '
  cd "$HOME/xtbflow-runs/w137-20261107-931d81f/source"
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate xtbflow-qc
  test "$(git rev-parse HEAD)" = 931d81f53a85669e137600c62b343120ff0de3d0
  export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
  export PYTHONPATH=src:vendor/mechai_reusable PYTHONDONTWRITEBYTECODE=1
  # Run inside the recorded finite Slurm allocation: main, gpu:pro6000:1,
  # one task, 8 CPUs, 64 GiB, 120 minutes; the original job ID was 4096.
  python scripts/replay_certification_panel.py \
    --plan "$HOME/xtbflow-runs/followup-20261009/o2-plan.json" \
    --archive "$HOME/xtbflow-runs/followup-20261009/o2-run" \
    --out "$HOME/xtbflow-runs/w137-20261107-931d81f/panel"
'
```

This records the original invocation, not a command to overwrite it: the runner
rejects an existing output directory. A replay must first create another clean
named checkout and new output path, with its own finite Slurm allocation. The
preserved scheduler script captures source/environment, start/final scheduler
metadata and exit status; `sacct` is disabled on n2.

Software validation at this source was separately **301 tests and 68 subtests**,
56 reusable unittest tests and bootstrap 338 files. Those checks did not expose
the real concurrent temporary-file race. The outcome-selected two-case panel is
numerical regression only, not an estimate of discovery success, superiority,
independent generalization, solution chemistry or experimental validity. The
second reference chain and the new generated-candidate chain remain unfinished.
