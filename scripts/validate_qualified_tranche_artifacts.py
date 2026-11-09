#!/usr/bin/env python3
"""Offline validator for the qualified next-tranche evidence bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import re


HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--benchmark-replicate", action="append", type=Path, default=[])
    parser.add_argument("--benchmark-aggregator", type=Path, default=None)
    args = parser.parse_args()
    status = _load(args.status)
    if status.get("science_ready") is not False:
        raise ValueError("science_ready must remain false")
    qualification = _load(args.qualification)
    baseline = _load(args.baseline)
    split = _load(args.split)
    benchmark = _load(args.benchmark)
    source = str(status.get("source_commit", ""))
    if not HEX40.fullmatch(source):
        raise ValueError("status source_commit must be a full SHA")
    subprocess.run(["git", "cat-file", "-e", f"{source}^{{commit}}"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "merge-base", "--is-ancestor", source, "HEAD"], check=True, stdout=subprocess.DEVNULL)
    if status.get("evidence_generation_commit") != source:
        raise ValueError("status evidence_generation_commit must equal source_commit")
    for name, artifact in (("qualification", qualification), ("baseline", baseline), ("split", split), ("benchmark", benchmark)):
        if artifact.get("source_commit") != source:
            raise ValueError(f"{name} source_commit does not match status source_commit")
    if qualification.get("status") != "qualified_for_tblite_force_response":
        raise ValueError("scoped qualification did not pass")
    if baseline.get("row_count") != 81 or baseline.get("status") != "completed":
        raise ValueError("baseline artifact row/status contract failed")
    if int(baseline.get("calculator_calls", 0)) > int(baseline.get("protocol", {}).get("budget_policy", {}).get("max_total_calculator_calls", 0)):
        raise ValueError("baseline calculator budget exceeded")
    if any(row.get("status") != "success" for row in baseline.get("rows", [])):
        raise ValueError("baseline contains failed rows")
    if split.get("status") != "pass" or split.get("selected_parent_count") != 9 or split.get("selected_family_count") != 9:
        raise ValueError("confirmation split audit failed")
    if benchmark.get("status") != "completed":
        raise ValueError("fixed-hardware benchmark did not complete")
    if benchmark.get("schema") == "xtbflow-fixed-hardware-parallel-benchmark-aggregate/v1":
        if benchmark.get("tasks_per_mode") != [81, 81] or benchmark.get("threshold_pass") is not False:
            raise ValueError("benchmark aggregate task/threshold contract failed")
    else:
        for mode in ("serial", "parallel"):
            if benchmark.get("modes", {}).get(mode, {}).get("summary", {}).get("tasks") != 81:
                raise ValueError(f"benchmark {mode} task count failed")
    expected_hashes = status.get("artifact_sha256", {})
    for key, path in {"gfn2_scoped_requalification": args.qualification, "qualified_baseline_confirmation": args.baseline, "confirmation_split_audit": args.split, "fixed_hardware_parallel_benchmark": args.benchmark}.items():
        if expected_hashes.get(key) != _sha(path):
            raise ValueError(f"status artifact hash mismatch for {key}")
    replicate_hashes = status.get("benchmark_replicates_sha256", {})
    for label, path in zip(("run1", "run2"), args.benchmark_replicate):
        if replicate_hashes.get(label) != _sha(path):
            raise ValueError(f"benchmark replicate hash mismatch for {label}")
    if args.benchmark_aggregator is not None and replicate_hashes.get("aggregator_script") != _sha(args.benchmark_aggregator):
        raise ValueError("benchmark aggregator script hash mismatch")
    summary = {
        "status": "pass",
        "source_commit": source,
        "rows": baseline["row_count"],
        "confirmation_parents": split["selected_parent_count"],
        "confirmation_families": split["selected_family_count"],
        "benchmark_tasks_per_mode": 81,
        "artifact_sha256": {key: _sha(path) for key, path in {
            "status": args.status,
            "qualification": args.qualification,
            "baseline": args.baseline,
            "split": args.split,
            "benchmark": args.benchmark,
        }.items()},
        "claim_limits": [
            "offline software/evidence validation only",
            "no chemistry, physical accuracy, certification, performance, or generalization claim",
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
