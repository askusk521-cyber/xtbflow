#!/usr/bin/env python3
"""Aggregate fixed-hardware benchmark replicates without selecting a favorable run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in args.run]
    sources = {report.get("source_commit") for report in reports}
    if len(sources) != 1:
        raise ValueError("benchmark replicates do not share source commit")
    speedups = [float(report["comparison"]["serial_over_parallel_speedup"]) for report in reports]
    tasks = [report["modes"]["serial"]["summary"]["tasks"] for report in reports]
    if any(value != 81 for value in tasks):
        raise ValueError("every benchmark replicate must contain 81 serial tasks")
    median_speedup = statistics.median(speedups)
    aggregate = {
        "schema": "xtbflow-fixed-hardware-parallel-benchmark-aggregate/v1",
        "status": "completed",
        "source_commit": next(iter(sources)),
        "replicate_paths": [str(path) for path in args.run],
        "replicate_sha256": {str(path): sha(path) for path in args.run},
        "replicate_speedups": speedups,
        "replicate_count": len(reports),
        "tasks_per_mode": tasks,
        "median_serial_over_parallel_speedup": median_speedup,
        "threshold": 1.1,
        "threshold_pass": median_speedup >= 1.1,
        "claim_limits": [
            "This is a fixed-node CPU scheduler diagnostic for the declared workload and replicate set.",
            "No general performance, throughput, chemistry, physical accuracy, certification, or generalization claim.",
        ],
    }
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": aggregate["status"], "replicates": len(speedups), "median_speedup": median_speedup, "threshold_pass": aggregate["threshold_pass"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
