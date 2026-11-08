# Draft PR: bind xtbflow MVP evidence to exact head

## Summary

This release-hardening change keeps the MVP implementation from commit
`08764ae16efd3bc4c9d8b25e33c4626961bdc6ee` unchanged and replaces its
runtime evidence with a fresh exact-head clean-worktree run.  The committed
artifact and evidence are generated from that commit and report
`source_dirty=false`.

The software scope remains one fixed reactant fixture, one serial candidate
path, shared mock/xTB calculator protocol, QC filtering, budget accounting,
failure taxonomy, replay identity, and machine-readable reports.

## Exact-head evidence

- Mock artifact: `artifacts/mvp_run_20261109T015300/`
- Mock evidence: `docs/evidence/mvp_mock_software_exact_08764ae_20261109.json`
- Replay evidence: `docs/evidence/mvp_replay_exact_08764ae_20261109.json`
- xTB evidence: `docs/evidence/mvp_xtb_smoke_exact_08764ae_20261109.json`
- Bootstrap audit: `docs/evidence/mvp_bootstrap_inventory_audit_20261109.json`

The artifact was copied from the exact-head run into the committed release
directory; only `output_paths` and the human-readable package locator were
relocated to repository-relative paths.  Provenance, calculator identity,
counts, and replay fields were left unchanged.

All exact-head evidence records bind to:

```text
source_commit: 08764ae16efd3bc4c9d8b25e33c4626961bdc6ee
source_dirty: false
config_sha256: 1f16915a72125c572652c3bf4b73ad3eb71d7be1e90d1c5cdc41908c876e0b92
input_hash: 73d70fdc975b9d449666182c98adeac2669656f7ca63ff29f5c816463b7275ce
```

The mock run produced 3 accepted and 2 QC-rejected candidates.  The n2 xTB
smoke produced 4 accepted and 1 QC-rejected candidate.  The replay summary
hash for the mock arm is
`a7a8f6949d6c907f3b5950b35725e614ce2fe1b7adb0db61f85a1c55cc6a4a82`.

## Validation

On the exact branch checkout:

```bash
PYTHONPATH=src python -m xtbflow.mvp run --config configs/mvp_fixture.yaml
```

On n2 (`node2`, Python 3.10.19, PyTorch 2.10.0+cu128, xTB distribution
22.1/import version 20.2):

```bash
PYTHONPATH=src:vendor/mechai_reusable pytest -q \
  tests/test_mvp.py tests/test_calculator_contract.py tests/test_runtime_budget.py
```

Result: `24 passed`; the explicit xTB backend smoke completed with
`status=completed`.  A missing local xTB/NumPy installation remains
fail-closed as `status=blocked_backend`; it never falls back to mock.

## Bootstrap inventory

The clean exact-head audit reports 22 repository-wide changed files.  Only
`README.md` and `src/xtbflow/calculators/__init__.py` overlap this MVP commit;
the remaining 20 files are base/prior drift.  This PR deliberately does not
run `--refresh`, so it does not absorb unrelated inventory changes.  A
separate inventory repair commit is required.

## Claim boundary

This is software/proxy evidence only.  It does not claim a transition state,
barrier, mechanism, chemical accuracy, or model superiority.  `accepted`
means only that the configured geometry/force QC filters passed.

## Follow-up task

The next research task is a one-parent, one-seed, matched-budget closed loop:

```text
reactant snapshot
→ bounded physical feedback
→ geometry/event update
→ independent TS/IRC certification
→ no-feedback matched-budget comparison
```

That task must preserve snapshot visibility, trust radius, collision guard,
event-edit accounting, calculator cost, failure stage, and independent
certification provenance.
