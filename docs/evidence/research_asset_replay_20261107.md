# W01 pinned assets and archival replay — 2026-11-07

The O2 analysis reproduced the archived output byte for byte from all 159
hash-verified inputs on node2. Ten candidate links retain planned and observed
events separately, geometry identities, endpoints and historical costs.
Three joint-generator checkpoints are available with the original hashes.
This is an integration increment for #131, not completion of #131 or #137.

中文说明：本次从真实原始记录重新计算了旧 DFT 报告，159 个输入文件全部通过哈希核验，
重算结果与原报告逐字节相同。三个联合生成模型的权重也仍可取得，文件内容与原记录一致。
这些结果说明后续开发可以复用这些资产，但本次没有重新生成候选或重新做量化计算。
下一步需要把新候选生成、公共认证工具和预算账本接成同一条执行链。

## Source and artifacts

- Base: main `d1aee8f7e18bfa30fd53308064f55b2a34fa42b7`.
- Implementation and full tests: `8f9f988fdbc039027d9d123f70a01fbf1ec107c5`.
- Routing change and focused tests: `35842c851b808fded088955cdf1faabc092bdeb7`.
- Pinned O2 research source: `e84e330390a519bf4ac12048b297403c09717646`, clean.
- Registry SHA256: `40b685778303ca06e4dd6be8c886277c09e0498eb01eeddf066ab93d4c6a183c`.
- Original and replay result SHA256:
  `eb99b097a8161c3a1ed93287eccaf979a6407fecbb962d0610e7b5d2be0f0043`.
- [Replay report](research_asset_replay_20261107.json): all raw identities,
  environment, source/protocol identity and per-candidate links.
- [Weight availability](research_weight_audit_20261107.json): three original
  checkpoint hashes and measured audit time; weights remain on n2.

The registry's 35 pinned repository files were independently hashed from Git
objects at the five declared PR heads. This is repository-file integrity only;
it does not mean all raw PR #111/#113/#114/#120 outputs have been rechecked.
Their unverified raw assets remain explicit handoff work.

## Execution and resources

Use the private environment variables and exact command in
[the replay contract](../architecture/research_asset_replay.md#reproduction-on-n2).
For this run `XTBFlow_REPLAY_COMMIT` was the full `8f9f988` SHA above;
the unique run key was `w131-20261107-8f9f988`, source subdirectory `source`,
output subdirectory `replay`. The n2 runs root is supplied through
`XTBFlow_RUN_ROOT`. O2 inputs use logical root `followup-20261009`; its clean
source checkout is `source-e84e330`. No historical file was changed.

The command ran through `ssh n2`, activated `xtbflow`, and used `srun
--partition=main --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G
--time=00:12:00`. Caps: 600 s replay wall time, 2 GiB hashed inputs,
64 MiB new output, no new GPU or calculator calls. Actual hashed input size
was 44,468,955 bytes; process wall time 1.079885 s and CPU time 0.949045 s.
The result is `replayed_byte_identical`. CPU measurement includes the analyzer
child. Scheduler/job setup and shell activation are outside that measurement.
`sacct` storage is disabled, so scheduler settlement and the unrecorded Slurm
job IDs are unavailable; no scheduler time is inferred from process time.

The original DFT result is unchanged: 10/10 anchors, 9/10 reference chains
and 4/10 candidate chains passed the archived rules. Candidate failures remain
2 event mismatches, 2 protocol failures, 1 unresolved and 1 SCF exception.
The original ledger accounts for 37,395 GPU seconds including pre-QC failures,
and excludes the previously disclosed GPU diagnostic checks. This is neither
new resource consumption nor the full generation-to-certification budget.

The initial source-bundle clone requested a branch not present in a HEAD-only
bundle and failed before any tests or data work. The same bundle was then
cloned with `--no-checkout`, and a named execution branch was created at the
exact SHA. No running job was restarted and no existing checkout was reset.

## Validation

From the clean integration source, on node2 with the conda environment active:

```bash
export PYTHONPATH=src:vendor/mechai_reusable
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/validate_bootstrap.py
python -m pytest -q
python -m unittest discover -s tests/reusable -p 'test_*.py'
```

The full test Slurm envelope was 10 minutes, 1 CPU, 4 GiB, no GPU and zero
new calculator calls. At `8f9f988`: bootstrap 326 files match; pytest
278 passed / 68 subtests passed (12.51 s, one existing tensor warning);
unittest 56 tests OK. The private log SHA256 is
`b27b6d16de06b80bd48b63b62a8ceb0aca07d57c7c871ac028e5e1e42da9865a`.
Only fixture/software contracts were exercised by these suites.

At `35842c8`, the same envelope ran `python -m pytest -q
tests/test_run_status_routing.py tests/test_research_replay.py`: 15 passed
in 0.11 s; bootstrap 328 files match. Log SHA256:
`7f35353907757ddbc863a36aaefb7b7a1445640721bb5d8bfa0a3921739217cd`.
The three added routing tests verify the persistent #129 link, #130 pin and
unchanged machine-readable status payload. Local checks were JSON/syntax and
`git diff --check`, not scientific execution.

## Weight audit reproduction

Run on n2 from the clean `35842c8` source, activating `xtbflow`. Set
`XTBFlow_RUN_ROOT` and an unused `XTBFlow_WEIGHT_AUDIT` output file first.
The read-only audit cap is 120 seconds / 2 GiB, zero QC calls / zero GPU.

```bash
PYTHONPATH=src python - <<'PY'
import json, os, time
from pathlib import Path
from xtbflow.governance.research_replay import audit_inputs
config = json.loads(Path('configs/research/assets_20261107.json').read_text())
asset = next(a for a in config['assets'] if a['pr'] == 111)
manifest = {'files': [
    {'path': '/xtbflow-runs/' + row['locator'].removeprefix('runs:'),
     'sha256': row['sha256']} for row in asset['model_assets']]}
rows = []
audit_inputs(manifest, Path(os.environ['XTBFlow_RUN_ROOT']), rows,
             max_bytes=2147483648, deadline=time.monotonic() + 120)
with Path(os.environ['XTBFlow_WEIGHT_AUDIT']).open('x') as stream:
    json.dump(rows, stream, indent=2)
PY
```

All three files are 3,901,150 bytes; total audit wall time 0.025844 s and
CPU time 0.014648 s. No checkpoint was deserialized or used for inference.

## Remaining gates

Fresh reactant-only joint and serial generation, common live certification,
hard shared-ledger reservations, independent task-data admission, full costs,
strong controls and scientific comparisons remain required. Two observed
event mismatches are preserved as source records; this increment does not
promote them to new `valid_alternative` certificates. Website publication
still uses the separate `site` branch. No issue is closed and no PR is merged.
