# Reference monitor repair replay — 2026-11-07, job 4099

**Passed:** the previously interrupted case `98a2bcd101825aa9` completed the
reference TS, mode, two-way IRC and endpoint chain after the monitor repair.
Observed and planned event `5065259568e669c1b5f2a2f8` agree. Frozen numerical
tolerances were unchanged; the comparison with the original archived verdict
passed. This is a selected numerical regression, not a new discovery-rate result.

Execution source: `e07ac8512fa987e0fad2e37b14c576c534a7df87`, branch
`codex/w137-monitor-race-20261107`, PR #148 based on #147. Host `node2`, Conda
`xtbflow-qc`, Python 3.11.17, PySCF 2.14.0, gpu4pyscf-cuda12x 1.8.1,
CuPy 14.2.0, NumPy 2.4.6, SciPy 1.17.1, geomeTRIC 1.1.1 and RDKit 2026.3.6.
The allocated device was RTX PRO 6000 Blackwell, CUDA driver 13020/runtime 12090.
The [full report](reference_monitor_retry_20261107.json) includes package RECORD,
source, method and threshold hashes. The [independent artifact audit](reference_monitor_retry_20261107_artifacts.json)
verified the request, result, all raw artifacts, call journal, settlement and
ledger totals with zero new calculator calls. No reservation remains.

- New retry: **290 primitives** = 143 SCF + 143 gradients + 4 Hessians; no unfinished primitive.
- Measured reference wall: 1462.991242 s; process CPU: 1634.645271 s; raw output: 1,201,998 bytes.
- Runner wall: 1463.714610 s. Slurm: `COMPLETED`, exit `0:0`, 1464 s.
- Reserved allocation: one GPU and eight CPUs, so 0.406667 GPU-hours and 3.253333 allocated core-hours; measured CPU use is 0.454068 core-hours.
- Identical cache replay: zero new calls and no new channel; one channel in this retry stage.
- TS energy difference: 2.274e-13 Ha; endpoint-energy maximum difference: 2.282e-8 Ha.
- Maximum frequency difference: 9.758e-8 cm^-1; aligned TS RMSD: 3.217e-10 Angstrom.

Prior job 4096 is preserved in PR #145: the first case passed with 202 calls and
the second was interrupted after 265 calls. This new retry adds 290, so the two
original cases plus repair now cost **757 primitives**. The repaired case alone
costs **555 primitives** including its failed attempt. Total measured reference
wall across all three attempts is 3841.548282 s and process CPU is 4280.635088 s.
There are two distinct certified original panel channels across the separate
runs. The original panel report remains failed; it has not been rewritten as a
successful run. The new result has a new service cache identity.

## Frozen identities and reproduction

- Config SHA256: `2688bfeb7295990c42e233bcd023c3a7ea04be6212022691b684adae9fa44490`.
- Protocol SHA256: `637e18f276d78daa7b9ed5155139dae6cb93943a9a1b5ad03e3516ff574e14d6`.
- Input plan SHA256: `0993c387fba18cd7761542099b1acb8146b9d78f2293f068d9ea872fe5d50fc8`.
- Report SHA256: `a859a201c98c0126b6a03f1b725c87c3090ebf540cac7662356132b62d4783ad`.
- Job script SHA256: `be95e4297f7d02af78f03d9bbed17fa111fc9979843fd7c9c7ce016110d4eec1`.

The preserved script `xtbflow-monitor-fix-e07ac85-job.sh` ran the following in a
fresh named execution directory. The stage allowed 1 GPU-hour, 8 CPU-core-hours,
3000 primitives, 0.5 GiB, one candidate, 3300 seconds per candidate and no external
retry. Slurm requested main, gpu:pro6000:1, eight CPUs, 64 GiB and 60 minutes.

```bash
ssh n2 '
  cd "$HOME/xtbflow-runs/w137fix-20261107-e07ac85/source"
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate xtbflow-qc
  test "$(git rev-parse HEAD)" = e07ac8512fa987e0fad2e37b14c576c534a7df87
  export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
  export PYTHONPATH=src:vendor/mechai_reusable PYTHONDONTWRITEBYTECODE=1
  # Original invocation inside the finite Slurm job 4099; choose a new output to rerun.
  python scripts/replay_certification_panel.py \
    --config configs/validation/certification_replay_retry_20261107.json \
    --plan "$HOME/xtbflow-runs/followup-20261009/o2-plan.json" \
    --archive "$HOME/xtbflow-runs/followup-20261009/o2-run" \
    --out "$HOME/xtbflow-runs/w137fix-20261107-e07ac85/panel"
'
```

The runner rejects an existing output. Reproduction requires another clean
named checkout and output plus a separately recorded finite allocation. Software
checks at this execution source passed 328 tests/68 subtests, 56 reusable tests
and bootstrap 354 files. This evidence adds real concurrent-monitor and numerical
regression validation. It does not establish method superiority, confirmatory
generalization, solution chemistry, CP2K/xTB results or experimental agreement.
Generated-candidate certification is an independent stage and is not implied by
this passed replay. Raw geometries and logs remain on n2.
