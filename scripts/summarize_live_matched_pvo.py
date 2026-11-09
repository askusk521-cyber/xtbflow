#!/usr/bin/env python3
"""Build a deterministic, paired summary from persisted live PVO rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_median(values: list[float], *, seed: int = 20261110, draws: int = 2000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    medians = [statistics.median(rng.choices(values, k=len(values))) for _ in range(draws)]
    return _quantile(medians, 0.025), _quantile(medians, 0.975)


def _arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [int(row["calculator_calls"]) for row in rows]
    reductions = [float(row["force_rms_reduction"]) for row in rows if row.get("force_rms_reduction") is not None]
    walls = [float(row["wall_seconds"]) for row in rows]
    successes = [row for row in rows if row.get("status") == "success"]
    exact = sum(bool(row.get("event_exact_match")) for row in rows)
    return {
        "rows": len(rows),
        "success_rows": len(successes),
        "failure_rate": 1.0 - (len(successes) / len(rows) if rows else 0.0),
        "calculator_calls": sum(calls),
        "median_force_rms_reduction": statistics.median(reductions) if reductions else None,
        "median_wall_seconds": statistics.median(walls) if walls else None,
        "p95_wall_seconds": _quantile(walls, 0.95),
        "event_exact_matches": exact,
        "event_exact_rate": exact / len(rows) if rows else 0.0,
        "geometry_changed_rate": sum(bool(row.get("geometry_changed")) for row in rows) / len(rows) if rows else 0.0,
        "event_changed_rate": sum(bool(row.get("event_changed")) for row in rows) / len(rows) if rows else 0.0,
        "model_forwards": sum(
            int(
                entry.get("charged_cost", {}).get(
                    "model_forwards",
                    entry.get("measured_cost", {}).get("model_forwards", 0),
                )
            )
            for row in rows
            for entry in row.get("adapter_ledger", [])
        ),
    }


def _paired(rows: list[dict[str, Any]], left_arm: str, right_arm: str, field: str) -> list[tuple[float, float]]:
    indexed = {(row["parent_id"], int(row["seed"]), row["arm"]): row for row in rows}
    keys = sorted({(row["parent_id"], int(row["seed"])) for row in rows})
    pairs = []
    for key in keys:
        left = indexed.get((*key, left_arm))
        right = indexed.get((*key, right_arm))
        if left is not None and right is not None and left.get(field) is not None and right.get(field) is not None:
            pairs.append((float(left[field]), float(right[field])))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--primary-ledger", type=Path, required=True)
    parser.add_argument("--calibration-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = _load(args.protocol)
    primary = _load(args.primary)
    calibration = _load(args.calibration)
    rows = list(primary["rows"])
    calibration_rows = list(calibration["rows"])
    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in protocol["arms"]}
    calibration_by_arm = {arm: [row for row in calibration_rows if row["arm"] == arm] for arm in protocol["arms"]}

    speed_pairs = _paired(rows, "serial_full", "parallel_full", "wall_seconds")
    speed_ratios = [serial / parallel for serial, parallel in speed_pairs if parallel > 0.0]
    physical_pairs = _paired(rows, "physical_only", "none", "force_rms_reduction")
    physical_lift = [physical - baseline for physical, baseline in physical_pairs]
    policy_pairs = _paired(rows, "policy_only", "none", "event_exact_match")
    policy_lift = [float(policy) - float(baseline) for policy, baseline in policy_pairs]
    paired_speed_ci = _bootstrap_median(speed_ratios)
    paired_physical_ci = _bootstrap_median(physical_lift)
    paired_speed_median = statistics.median(speed_ratios) if speed_ratios else 0.0
    paired_physical_median = statistics.median(physical_lift) if physical_lift else 0.0
    policy_lift_mean = statistics.mean(policy_lift) if policy_lift else 0.0

    primary_calls = sum(int(row["calculator_calls"]) for row in rows)
    calibration_calls = sum(int(row["calculator_calls"]) for row in calibration_rows)
    thresholds = protocol["thresholds"]
    summary = {
        "schema": "xtbflow-live-matched-pvo-summary/v4",
        "status": "completed_with_qualification_limit",
        "source_commit": primary["source_commit"],
        "source_primary_sha256": _sha256(args.primary),
        "source_calibration_sha256": _sha256(args.calibration),
        "primary_parent_count": primary["parent_count"],
        "calibration_parent_count": calibration["parent_count"],
        "seed_count": primary["seed_count"],
        "primary_row_count": len(rows),
        "calibration_row_count": len(calibration_rows),
        "arms": list(protocol["arms"]),
        "arms_summary": {arm: _arm_summary(by_arm[arm]) for arm in protocol["arms"]},
        "calibration_summary": {arm: _arm_summary(calibration_by_arm[arm]) for arm in protocol["arms"]},
        "comparisons": {
            "physical_vs_none_paired_median_force_rms_lift": paired_physical_median,
            "physical_vs_none_paired_median_force_rms_lift_bootstrap_95ci": list(paired_physical_ci),
            "chemical_policy_only_vs_none_paired_mean_exact_lift": policy_lift_mean,
            "parallel_vs_serial_paired_median_speedup": paired_speed_median,
            "parallel_vs_serial_paired_speedup_p95_latency_ratio": _quantile(speed_ratios, 0.95),
            "parallel_vs_serial_paired_median_speedup_bootstrap_95ci": list(paired_speed_ci),
            "parallel_vs_serial_ratio_of_medians_reference": (
                statistics.median(by_arm["serial_full"][i]["wall_seconds"] for i in range(len(by_arm["serial_full"])))
                / statistics.median(by_arm["parallel_full"][i]["wall_seconds"] for i in range(len(by_arm["parallel_full"])))
                if by_arm["serial_full"] and by_arm["parallel_full"] else 0.0
            ),
            "threshold_passes": {
                "physical": paired_physical_median >= float(thresholds["physical_median_force_rms_reduction_min"]),
                "chemical": policy_lift_mean >= float(thresholds["chemical_event_exact_match_lift_min"]),
                "parallel": paired_speed_median >= float(thresholds["parallel_wall_speedup_min"]),
            },
        },
        "cost_accounting": {
            "fields": protocol["cost_accounting"]["fields"],
            "primary_calculator_calls": primary_calls,
            "calibration_calculator_calls": calibration_calls,
            "global_calculator_calls": primary_calls + calibration_calls,
            "primary_budget": protocol["budget_policy"]["primary_max_total_calculator_calls"],
            "calibration_budget": protocol["budget_policy"]["calibration_max_total_calculator_calls"],
            "global_budget": protocol["budget_policy"]["global_max_total_calculator_calls"],
            "persistent_primary_ledger_lines": sum(1 for _ in args.primary_ledger.open()),
            "persistent_calibration_ledger_lines": sum(1 for _ in args.calibration_ledger.open()),
            "retry_identity": protocol["retry_policy"]["retry_identity"],
        },
        "ledger_paths": {
            "primary": str(args.primary_ledger),
            "calibration": str(args.calibration_ledger),
            "primary_sha256": _sha256(args.primary_ledger),
            "calibration_sha256": _sha256(args.calibration_ledger),
        },
        "thresholds": thresholds,
        "qualification_status": protocol["calculator_protocol"]["qualification_status"],
        "claim_limits": protocol["claim_limits"] + [
            "The paired performance statistic is a diagnostic wall-time comparison under the frozen CPU protocol; it is not a general performance claim.",
            "The chemical endpoint is an evaluation comparison against a frozen reference event and is not a chemistry-validity claim.",
        ],
    }
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "source_commit": summary["source_commit"], "paired_speedup": paired_speed_median, "physical_lift": paired_physical_median}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
