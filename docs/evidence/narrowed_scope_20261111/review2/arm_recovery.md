# Review #193: complete arm recovery and certification stop

At clean source `5419b365c1969245b53c031128da4213f2e64dcd`, node2 passed
17 focused tests, 344 full pytest tests/68 subtests, 56 reusable unittests,
bootstrap 412 files and diff check. The
[immutable receipt](arm_recovery_validation.json) records exact commands,
package versions, hashes, per-command wall/process CPU usage and limits.

The earlier adapter-level tests missed two outer-run defects. New `_run_arm`
tests execute a successful physical-feedback arm twice and verify backend calls
stay one, certification entry stays one, accepted geometry/feedback are identical,
the public cumulative call count stays one, new cost is zero and every existing
file hash is unchanged. Duplicate incremental cost remains zero; it is not
converted to historical total cost.

A two-second fake provider measurement against a one-second wall budget now
returns `GUIDANCE_BUDGET_EXHAUSTED` and never enters certification. A pre-existing
started journal with unknown backend termination returns
`BACKEND_TERMINATION_UNKNOWN`, retaining its conservative charge and leaving
files untouched. A public-ledger settlement failure returns
`COST_SETTLEMENT_INCOMPLETE` with the unsettled event, and zero certification
entries. An interruption during certification retains the prior candidate and
blocks re-entry. Changed config/snapshot/artifact identities are also blocked.

Completed replay requires a matching durable result/artifact identity. Partial
or legacy attempts stop for inspection; this patch does not reconstruct a
missing candidate, automatically finish an interrupted backend, or rewrite a
historical run. The top-level CLI rejects nonempty output directories. The
individual arm's immutable start marker prevents a second launch before a
completed result exists. No branch was merged and no evidence ties these bugs
to the old chemical outcomes.

Reproduction uses this exact source and a fresh log directory. Software cap:
one CPU thread, 600 wall/CPU seconds per command, 64 MiB per log, concurrency
one and no retry. The actual collection wrapper has a 1900-second overall wall
limit and its hash is in the receipt. No real calculator, trained checkpoint,
GPU allocation or research training is used.

```bash
ssh n2 '
  cd "$XTB_REVIEW193_CODE"
  test "$(git rev-parse HEAD)" = 5419b365c1969245b53c031128da4213f2e64dcd
  test -z "$(git status --porcelain)"
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate xtbflow
  export PYTHONPATH=src:vendor/mechai_reusable CUDA_VISIBLE_DEVICES=
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  ulimit -f 131072
  timeout 600 python -m pytest -q tests/test_live_arm_recovery.py tests/test_failed_provider_accounting.py > "$XTB_NEW_REVIEW_LOGS/focused.log" 2>&1
  timeout 600 python -m pytest -q > "$XTB_NEW_REVIEW_LOGS/pytest.log" 2>&1
  timeout 600 python -m unittest discover -s tests/reusable -p "test_*.py" > "$XTB_NEW_REVIEW_LOGS/unittest.log" 2>&1
  python scripts/validate_bootstrap.py > "$XTB_NEW_REVIEW_LOGS/bootstrap.log"
  git diff --check
'
```
