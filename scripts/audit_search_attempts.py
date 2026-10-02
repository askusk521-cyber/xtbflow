#!/usr/bin/env python3
"""Audit append-only search attempts and aggregate recorded calculator cost."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from xtbflow.data.attempts import (
    SEARCH_ATTEMPT_SCHEMA,
    load_attempt_jsonl,
    latest_attempt_versions,
)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_report(
    manifest: Path,
    *,
    repository_base_commit: str,
) -> dict[str, Any]:
    rows = load_attempt_jsonl(manifest)
    latest_rows = latest_attempt_versions(rows)
    status_counts = Counter(row.evidence_status for row in latest_rows)
    call_counts: Counter[str] = Counter()
    for row in latest_rows:
        call_counts.update(row.calculator_calls)
    return {
        "schema_version": "xtbflow-search-attempt-audit/v2",
        "record_schema": SEARCH_ATTEMPT_SCHEMA,
        "manifest": str(manifest),
        "manifest_sha256": hash_file(manifest),
        "repository_base_commit": repository_base_commit,
        "audit_script_sha256": hash_file(Path(__file__)),
        "attempt_contract_sha256": hash_file(
            Path("src/xtbflow/data/attempts.py")
        ),
        "attempt_count": len(latest_rows),
        "version_record_count": len(rows),
        "latest_version_count": len(latest_rows),
        "cost_semantics": "latest_cumulative_snapshot_per_attempt",
        "status_counts": dict(sorted(status_counts.items())),
        "calculator_call_counts": dict(sorted(call_counts.items())),
        "total_calculator_calls": sum(call_counts.values()),
        "total_wall_seconds": sum(row.wall_seconds for row in latest_rows),
        "scientific_claim_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-base-commit", required=True)
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error(f"manifest does not exist: {args.manifest}")
    report = build_report(
        args.manifest,
        repository_base_commit=args.repository_base_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
