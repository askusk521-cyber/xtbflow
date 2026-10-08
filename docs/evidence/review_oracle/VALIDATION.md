# Verification at source 3740e4b

Executed on n2 with `CUDA_VISIBLE_DEVICES=` and one BLAS/OpenMP thread:

- `python scripts/validate_bootstrap.py`: PASS, 426 inventory-covered files. This inventory does not cover every newly added evidence file; independent SHA256SUMS covers this evidence directory.
- `python -m unittest discover -s tests/reusable -p 'test_*.py'`: 56 tests, OK.
- `python -m pytest -q tests --ignore=tests/reusable`: 304 passed, one pre-existing PyTorch scalar-conversion warning.
- `python scripts/review_oracle_benchmark.py analyze --out /home/lhshen/xtbflow-runs/review-align-20261009/oracle-combined` followed by `cmp docs/evidence/review_oracle/results.json .../oracle-combined/results.json`: exit 0, byte-identical reproduction.
- `git diff --name-status 9cc15e0...3740e4b`: only added files, plus bootstrap_inventory.json.
- results.json is approximately 195 KiB; no model weights or large raw data committed.

Remote test log: `/home/lhshen/xtbflow-runs/review-align-20261009/tests-3740e4b.log`.

Remaining evidence caveat: the per-call cost metric times TS/candidate evaluation only, not cached reactant calls. Acquisition process CPU accounting includes all calls. The manifest tracks each acquisition's distinct pushed source commit and config hash; the combined output is reconstructed by concatenating those raw files in the documented acquisition order.
