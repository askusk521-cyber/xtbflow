# xtbflow MVP status

This document describes the bounded software slice delivered in this working
tree.  It fixes one three-atom reactant fixture, one serial candidate path,
one seed, and a five-candidate calculator budget.  The fixture contains no
product, transition-state, reference-energy, or target-selected environment
fields.

Run it from a clean checkout with:

```bash
PYTHONPATH=src python -m xtbflow.mvp run --config configs/mvp_fixture.yaml
```

The run writes `manifest.json`, `candidates.jsonl`, `ledger.json`,
`summary.md`, `summary.png`, and a copy of the frozen input under
`artifacts/mvp_run_<timestamp>/`.  `manifest.json` records the input hash, Git
commit, config hash, seed, candidate order, calculator protocol/build identity,
accepted/rejected/failure counts, failure reason counts, budget totals, and
absolute output paths.

The default `mock` backend implements the existing calculator protocol with a
deterministic harmonic proxy.  Its rows and manifest carry
`backend=mock` and `evidence_level=software_demo`; energy and force values are
proxy metrics only.  `--backend xtb` selects the existing xTB oracle adapter
with explicit GFN2-xTB settings.  An unavailable native module produces
`status=blocked_backend` and `reason_code=BLOCKED_BACKEND` rows; the command
does not switch to mock implicitly.  Both backends use the same
`xtbflow.calculators.CalculatorProtocol/v1` contract and
`energy_forces` operation.

Candidate rows use `schemas/mvp_candidate.schema.json`; the frozen input uses
`schemas/mvp_input.schema.json`.  Every row has `status`, `reason_code`,
`attempt`, `wall_seconds`, `cpu_hours`, and `gpu_hours`, including failures.
The current fixture exercises accepted rows, a force-threshold QC rejection,
and a minimum-distance QC rejection.  Calculator failures are classified as
`BLOCKED_BACKEND`, `CALCULATOR_NOT_CONVERGED`, `CALCULATOR_PROTOCOL`, or
`CALCULATOR_EXECUTION`; budget and input failures have their own explicit
codes.

`replay.deterministic_summary_sha256` covers the fixed seed, input/config
hashes, candidate fingerprints/order, status/reason rows, and counts.  Wall
time, dirty-tree state, and native calculator build identity are intentionally
reported as environment-dependent fields.

Claim levels are explicit: `software_demo` for the runnable workflow,
`proxy_metric` for calculator/QC summary values, and `strict_validation` for
any future chemical interpretation.  This MVP makes no transition-state,
barrier, mechanism, chemical-accuracy, or model-superiority claim.

## Verification recorded for this working tree

All release evidence below was regenerated from exact commit
`08764ae16efd3bc4c9d8b25e33c4626961bdc6ee` in clean worktrees; each evidence
record reports `source_dirty=false`.

- Exact-head mock run: 5 candidates, 3 accepted, and 2 QC rejections.  The
  committed bundle is under `artifacts/mvp_run_20261109T015300/`.
- Exact-head n2 smoke: `node2`, Python 3.10.19, PyTorch 2.10.0+cu128, xTB
  import version 20.2 (distribution 22.1), CPU backend.  The xTB path
  completed with 4 accepted candidates and 1 QC rejection.
- Exact-head MVP plus calculator/ledger regression slice: 24 tests passed on
  n2.  These remain software/proxy evidence, not chemical validation.
- Exact-head evidence and replay records:
  [`mvp_mock_software_exact_08764ae_20261109.json`](docs/evidence/mvp_mock_software_exact_08764ae_20261109.json),
  [`mvp_replay_exact_08764ae_20261109.json`](docs/evidence/mvp_replay_exact_08764ae_20261109.json),
  [`mvp_xtb_smoke_exact_08764ae_20261109.json`](docs/evidence/mvp_xtb_smoke_exact_08764ae_20261109.json).
