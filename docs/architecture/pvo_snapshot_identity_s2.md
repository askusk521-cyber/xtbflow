# S2 old PVO snapshot identity

Related #185; stack base is `a93eeb4b30301f7792f1952c947e30b0a9d837e6`.
The reused provider now sends current snapshot coordinates to the calculator
with the template's original elements/order/charge/spin/environment. It verifies
the returned input hash and keeps original-reactant hash, calculator-input hash
and immutable snapshot ID separate. Two recording-fake snapshots prove both
actual input consumption and stale-result rejection. Historical one-step data
are unchanged. No vectors or Hessians are transported by this interface.

At exact source `fa42e48ec8c419c2e44a2b715696ea86821ed3be`, node2/xtbflow
passed 303 pytest tests/68 subtests, 56 reusable tests and bootstrap 377 files.
See [receipt](../evidence/narrowed_scope_20261111/s2_pvo_software.json).

Reproduction uses a clean checkout at that source, a new log directory and
the existing n2 environment; no real calculation is invoked by these tests:

```bash
ssh n2 '
  cd "$XTB_PVO_TEST_CODE"
  test "$(git rev-parse HEAD)" = fa42e48ec8c419c2e44a2b715696ea86821ed3be
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate xtbflow
  export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  export PYTHONPATH=src:vendor/mechai_reusable
  ulimit -f 131072
  timeout 600 python -m pytest -q > "$XTB_NEW_TEST_LOGS/pytest.log" 2>&1
  timeout 600 python -m unittest discover -s tests/reusable -p "test_*.py" > "$XTB_NEW_TEST_LOGS/unittest.log" 2>&1
  python scripts/validate_bootstrap.py > "$XTB_NEW_TEST_LOGS/bootstrap.log"
'
```

This remains a heuristic software fixture, not a trained chemical-event model.
No new training, quantum chemistry, efficacy or independent-confirmation claim.
