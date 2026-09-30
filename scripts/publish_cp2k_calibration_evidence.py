#!/usr/bin/env python3
"""Publish a path-safe summary of one private CP2K calibration replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xtbflow.validation import (  # noqa: E402
    index_artifacts,
    sanitize_cp2k_calibration_report,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-report", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--publication-source-commit", "--source-commit",
        dest="publication_source_commit",
        help="commit that publishes this evidence; execution commit is read from the private report",
    )
    args = parser.parse_args()
    report_path = args.private_report.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve()
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"invalid private calibration report: {exc}")
    execution_source_commit = payload.get("source_commit")
    if not isinstance(execution_source_commit, str) or not execution_source_commit.strip():
        parser.error("private calibration report must contain the computation source_commit")
    execution_script_sha256 = payload.get("script_sha256")
    if not isinstance(execution_script_sha256, str) or not execution_script_sha256.strip():
        parser.error("private calibration report must contain the execution script_sha256")
    execution_adapter_sha256 = payload.get("cp2k_adapter_sha256") or payload.get(
        "adapter_sha256"
    )
    if not isinstance(execution_adapter_sha256, str) or not execution_adapter_sha256.strip():
        parser.error("private calibration report must contain the execution cp2k_adapter_sha256")
    publication_source_commit = args.publication_source_commit or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    public = sanitize_cp2k_calibration_report(
        payload,
        execution_source_commit=execution_source_commit,
        publication_source_commit=publication_source_commit,
        private_report_sha256=sha256_file(report_path),
        calibration_script_sha256=execution_script_sha256,
        adapter_sha256=execution_adapter_sha256,
        artifacts=index_artifacts(artifact_root),
    )
    encoded = json.dumps(
        public, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": public.get("status"),
                "cases": len(public["cases"]),
                "artifacts": len(public["artifact_index"]),
            },
            sort_keys=True,
        )
    )
    return 0 if public.get("status") == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
