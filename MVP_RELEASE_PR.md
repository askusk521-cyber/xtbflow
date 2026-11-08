# Draft PR: add narrow runnable xtbflow MVP

## Scope

This PR is based directly on the current `main` branch and contains only the
MVP implementation, its fixture/schema/test contract, and exact-head release
evidence. It does not include Reaction-QM history, research dispatch changes,
independent-seed work, training experiments, or inventory refresh.

Implementation commit:

```text
981e475e3b3384196cc56756fd1831c213f7225e
```

## Exact-head evidence

All records below were generated from that implementation commit in clean
worktrees and report `source_dirty=false`:

- `artifacts/mvp_run_20261109T020000/`
- `docs/evidence/mvp_mock_software_exact_981e475_20261109.json`
- `docs/evidence/mvp_replay_exact_981e475_20261109.json`
- `docs/evidence/mvp_xtb_smoke_exact_981e475_20261109.json`
- `docs/evidence/mvp_bootstrap_inventory_audit_20261109.json`

```text
source_commit: 981e475e3b3384196cc56756fd1831c213f7225e
source_dirty: false
config_sha256: 1f16915a72125c572652c3bf4b73ad3eb71d7be1e90d1c5cdc41908c876e0b92
input_hash: 73d70fdc975b9d449666182c98adeac2669656f7ca63ff29f5c816463b7275ce
mock_replay_sha256: a7a8f6949d6c907f3b5950b35725e614ce2fe1b7adb0db61f85a1c55cc6a4a82
```

Mock produced 3 accepted and 2 QC-rejected candidates. The n2 xTB smoke
produced 4 accepted and 1 QC-rejected candidate. The regression slice passed
24 tests.

## Validation commands

```bash
PYTHONPATH=src python -m xtbflow.mvp run --config configs/mvp_fixture.yaml
PYTHONPATH=src:vendor/mechai_reusable pytest -q \
  tests/test_mvp.py tests/test_calculator_contract.py tests/test_runtime_budget.py
```

The xTB path is explicit and fail-closed when unavailable. Mock output is
`software_demo`; calculator/QC values are `proxy_metric` evidence only.

## Bootstrap inventory

The clean exact-head audit found two changed tracked files, both owned by this
MVP (`README.md` and `src/xtbflow/calculators/__init__.py`). No refresh is
included here. Inventory repair is a separate `chore` PR and will not include
runtime or chemistry changes.

## Claim boundary

This PR does not claim transition states, barriers, mechanisms, chemical
accuracy, or model superiority. `accepted` means only that configured geometry
and force QC filters passed.

## Follow-up research task

The next separate task is one-parent/one-seed matched-budget comparison:

```text
reactant snapshot
→ bounded physical feedback
→ geometry/event update through merge gate
→ independent TS/IRC certification
→ no-feedback matched-budget comparison
```

The task must freeze the parent, input, seed, calculator and certification
protocols, and account for trust radius, collision guard, event edits,
guidance cost, unresolved/duplicate cost, failure stage, and certification
provenance.
