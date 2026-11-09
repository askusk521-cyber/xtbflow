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
    args = parser.parse_args()
    status = _load(args.status)
    qualification = _load(args.qualification)
    baseline = _load(args.baseline)
    split = _load(args.split)
    benchmark = _load(args.benchmark)
    source = str(status.get("source_commit", ""))
    if not HEX40.fullmatch(source):
        raise ValueError("status source_commit must be a full SHA")
    subprocess.run(["git", "cat-file", "-e", f"{source}^{{commit}}"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "merge-base", "--is-ancestor", source, "HEAD"], check=True, stdout=subprocess.DEVNULL)
    if qualification.get("status") != "qualified_for_tblite_force_response":
        raise ValueError("scoped qualification did not pass")
    if baseline.get("row_count") != 81 or baseline.get("status") != "completed":
        raise ValueError("baseline artifact row/status contract failed")
    if any(row.get("status") != "success" for row in baseline.get("rows", [])):
        raise ValueError("baseline contains failed rows")
    if split.get("status") != "pass" or split.get("selected_parent_count") != 9 or split.get("selected_family_count") != 9:
        raise ValueError("confirmation split audit failed")
    if benchmark.get("status") != "completed":
        raise ValueError("fixed-hardware benchmark did not complete")
    for mode in ("serial", "parallel"):
        if benchmark.get("modes", {}).get(mode, {}).get("summary", {}).get("tasks") != 81:
            raise ValueError(f"benchmark {mode} task count failed")
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
