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

## Verification recorded for this narrow branch

All release evidence below was regenerated from exact commit
`981e475e3b3384196cc56756fd1831c213f7225e` in clean worktrees; each evidence
record reports `source_dirty=false`.

- Exact-head mock artifact: `artifacts/mvp_run_20261109T020000/`, with 3
  accepted and 2 QC rejections.
- Exact-head n2 xTB smoke: 4 accepted and 1 QC rejection on node2, Python
  3.10.19, xTB distribution 22.1/import version 20.2.
- Exact-head regression slice: 24 tests passed on n2.
- Exact-head evidence:
  [`mvp_mock_software_exact_981e475_20261109.json`](docs/evidence/mvp_mock_software_exact_981e475_20261109.json),
  [`mvp_replay_exact_981e475_20261109.json`](docs/evidence/mvp_replay_exact_981e475_20261109.json),
  [`mvp_xtb_smoke_exact_981e475_20261109.json`](docs/evidence/mvp_xtb_smoke_exact_981e475_20261109.json).
- Bootstrap audit:
  [`mvp_bootstrap_inventory_audit_20261109.json`](docs/evidence/mvp_bootstrap_inventory_audit_20261109.json).

The bootstrap audit reports only two MVP-overlap files (`README.md` and
`src/xtbflow/calculators/__init__.py`); inventory refresh is deferred to a
separate repair commit.
