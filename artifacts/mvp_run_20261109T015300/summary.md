# xtbflow MVP summary

- status: `completed`
- backend: `mock`
- evidence_level: `software_demo`
- seed: `20261108`
- input_hash: `73d70fdc975b9d449666182c98adeac2669656f7ca63ff29f5c816463b7275ce`
- config_sha256: `1f16915a72125c572652c3bf4b73ad3eb71d7be1e90d1c5cdc41908c876e0b92`
- git_commit: `08764ae16efd3bc4c9d8b25e33c4626961bdc6ee`
- deterministic_summary_sha256: `a7a8f6949d6c907f3b5950b35725e614ce2fe1b7adb0db61f85a1c55cc6a4a82`

## Counts

candidates=5 accepted=3 rejected=2 failure=0

## Candidate outcomes

| index | event | status | reason_code | calculator_calls | force_norm |
| ---: | --- | --- | --- | ---: | ---: |
| 0 | `candidate_000_identity_control` | `accepted` | `` | 1 | 0 |
| 1 | `candidate_001_bend_h1` | `accepted` | `` | 1 | 0.0998274 |
| 2 | `candidate_002_stretch_h1` | `accepted` | `` | 1 | 0.249193 |
| 3 | `candidate_003_large_displacement` | `rejected` | `QC_FORCE_THRESHOLD` | 1 | 0.700682 |
| 4 | `candidate_004_too_close` | `rejected` | `QC_MIN_DISTANCE` | 0 |  |

## Budget

```json
{
  "active_jobs": [],
  "budget": {
    "max_calculator_calls": 5,
    "max_concurrent_jobs": 1,
    "max_cpu_core_hours": 0.1,
    "max_gpu_hours": 0.0,
    "max_retries_per_job": 0,
    "max_storage_gb": null,
    "max_system_evaluations": 5,
    "max_wall_minutes_per_job": 5.0,
    "phase": "mvp"
  },
  "by_calculator": {
    "mvp_mock": 5
  },
  "by_operation": {
    "energy_forces": 5
  },
  "by_status": {
    "failure": 1,
    "success": 4
  },
  "ledger_id": "ledger-8b3313094038128b",
  "phase": "mvp",
  "schema": "xtbflow-run-ledger/v1",
  "totals": {
    "attempts": 5,
    "cache_hits": 0,
    "calculator_calls": 4,
    "cpu_core_hours": 2.2155944444444886e-07,
    "failures": 1,
    "gpu_hours": 0.0,
    "retries": 0,
    "storage_gb": 0.0,
    "successes": 4,
    "system_evaluations": 5
  }
}
```

## Claim boundary

- `software_demo`: the fixed serial workflow, failure handling and replay contract.
- `proxy_metric`: calculator energy/force values and the configured QC acceptance counts.
- `strict_validation`: required before any transition-state, barrier, mechanism or chemical-accuracy claim; this MVP does not make those claims.

Artifacts are packaged under `artifacts/mvp_run_20261109T015300/summary.md`.
