#!/usr/bin/env python3
"""Validate persisted live PVO artifacts without invoking a calculator.

This validator is intentionally an offline contract check.  It verifies row,
ledger, budget, provenance, and source-tree identities; it does not certify
chemical correctness or calculator accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ARMS = ("none", "physical_only", "policy_only", "serial_full", "parallel_full", "policy_off")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _assert_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        raise ValueError(f"non-finite number at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, f"{path}[{index}]")


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        rows.append(value)
    return rows


def _validate_result(
    result_path: Path,
    ledger_path: Path,
    protocol: dict[str, Any],
    pool: str,
    expected_source: str | None,
) -> dict[str, Any]:
    result = _load(result_path)
    _assert_finite(result)
    rows = result.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{result_path}: rows must be a list")
    parent_ids = protocol[f"{pool}_parent_ids"]
    seeds = protocol["seeds"]
    expected_count = len(parent_ids) * len(seeds) * len(ARMS)
    if len(rows) != expected_count:
        raise ValueError(f"{result_path}: expected {expected_count} rows, got {len(rows)}")
    if not isinstance(result.get("source_commit"), str) or not HEX40.fullmatch(result["source_commit"]):
        raise ValueError(f"{result_path}: source_commit must be a full lowercase SHA-1")
    source_commit = result["source_commit"]
    if expected_source is not None and source_commit != expected_source:
        raise ValueError(f"{result_path}: source_commit does not match requested source")
    try:
        subprocess.run(["git", "cat-file", "-e", f"{source_commit}^{{commit}}"], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "merge-base", "--is-ancestor", source_commit, "HEAD"], check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"{result_path}: source_commit is not a reachable commit") from exc
    provenance = result.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{result_path}: missing provenance")
    recorded_ledger = provenance.get("ledger_path")
    if not isinstance(recorded_ledger, str) or Path(recorded_ledger).is_absolute() or ".." in Path(recorded_ledger).parts:
        raise ValueError(f"{result_path}: ledger_path must be repository-relative")
    if Path(recorded_ledger) != ledger_path:
        raise ValueError(f"{result_path}: ledger_path does not match validator input")
    if result.get("ledger_sha256") != _sha256(ledger_path) or not HEX64.fullmatch(str(result.get("ledger_sha256", ""))):
        raise ValueError(f"{result_path}: ledger SHA-256 mismatch")
    ledger_rows = _read_ledger(ledger_path)
    if len(ledger_rows) != len(rows):
        raise ValueError(f"{result_path}: ledger row count does not match result")
    row_attempts = {row.get("attempt_id") for row in rows}
    ledger_attempts = {row.get("attempt_id") for row in ledger_rows}
    if None in row_attempts or len(row_attempts) != len(rows) or row_attempts != ledger_attempts:
        raise ValueError(f"{result_path}: attempt_id set is not unique or does not match ledger")
    allowed_parents = set(parent_ids)
    allowed_seeds = set(seeds)
    max_row_calls = int(protocol["budget_policy"]["max_calculator_calls_per_row"])
    totals = {"calculator_calls": 0, "rows": len(rows), "success_rows": 0}
    for row in rows:
        if row.get("parent_id") not in allowed_parents or row.get("seed") not in allowed_seeds or row.get("arm") not in ARMS:
            raise ValueError(f"{result_path}: row has an unexpected parent, seed, or arm")
        cost = row.get("cost_accounting")
        if not isinstance(cost, dict) or set(("reserved_cost", "observed_cost", "charged_cost", "overrun")) - set(cost):
            raise ValueError(f"{result_path}: row is missing cost accounting fields")
        calls = int(row.get("calculator_calls", -1))
        if calls < 0 or calls > max_row_calls:
            raise ValueError(f"{result_path}: calculator call cap violated")
        observed = int(cost["observed_cost"]["calculator_calls"])
        charged = int(cost["charged_cost"]["calculator_calls"])
        if observed != calls or charged != calls:
            raise ValueError(f"{result_path}: observed/charged calls do not match row count")
        if int(cost["overrun"]["calculator_calls"]) != max(0, calls - int(cost["reserved_cost"]["calculator_calls"])):
            raise ValueError(f"{result_path}: call overrun is inconsistent")
        totals["calculator_calls"] += calls
        totals["success_rows"] += int(row.get("status") == "success")
        if row["arm"] in {"physical_only", "serial_full", "parallel_full"} and row.get("status") == "success" and not row.get("geometry_changed"):
            raise ValueError(f"{result_path}: physical arm did not change geometry")
        if row["arm"] in {"policy_only", "serial_full", "parallel_full"} and row.get("status") == "success" and not row.get("event_changed"):
            raise ValueError(f"{result_path}: policy arm did not change event")
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--primary-ledger", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--calibration-ledger", type=Path, required=True)
    parser.add_argument("--expected-source-commit", default=None)
    args = parser.parse_args()
    protocol = _load(args.protocol)
    expected_source = args.expected_source_commit
    primary = _validate_result(args.primary, args.primary_ledger, protocol, "primary", expected_source)
    calibration = _validate_result(args.calibration, args.calibration_ledger, protocol, "calibration", expected_source)
    primary_limit = int(protocol["budget_policy"]["primary_max_total_calculator_calls"])
    calibration_limit = int(protocol["budget_policy"]["calibration_max_total_calculator_calls"])
    global_limit = int(protocol["budget_policy"]["global_max_total_calculator_calls"])
    if primary["calculator_calls"] > primary_limit or calibration["calculator_calls"] > calibration_limit:
        raise ValueError("pool calculator-call budget exceeded")
    if primary["calculator_calls"] + calibration["calculator_calls"] > global_limit:
        raise ValueError("global calculator-call budget exceeded")
    summary = {
        "status": "pass",
        "source_commit": _load(args.primary)["source_commit"],
        "primary": primary,
        "calibration": calibration,
        "global_calculator_calls": primary["calculator_calls"] + calibration["calculator_calls"],
        "claim_limits": [
            "offline artifact validation only",
            "no chemistry, physical accuracy, performance, or generalization claim",
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
