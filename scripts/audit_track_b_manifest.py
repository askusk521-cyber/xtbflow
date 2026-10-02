#!/usr/bin/env python3
"""Audit a Track-B JSONL manifest without admitting guessed metadata."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from xtbflow.data.track_b import (
    TRACK_B_SCHEMA,
    TrackBLeakageError,
    audit_track_b_leakage,
    filter_track_b_records,
    load_track_b_jsonl,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_report(
    manifest: Path,
    *,
    strict_family_holdout: bool,
    policy_config: Path,
    repository_base_commit: str,
) -> dict[str, Any]:
    records = load_track_b_jsonl(str(manifest))
    filtered = filter_track_b_records(records)
    reason_counts = Counter(
        reason
        for row in filtered.quarantined
        for reason in row.quarantine_reasons
    )
    leakage_error: str | None = None
    leakage_report: dict[str, Any] = {}
    try:
        leakage_report = audit_track_b_leakage(
            filtered.accepted,
            strict_family_holdout=strict_family_holdout,
        )
    except TrackBLeakageError as exc:
        leakage_error = str(exc)
    admission_counts = Counter(row.admission for row in records)
    report = {
        "schema_version": "xtbflow-track-b-audit/v1",
        "record_schema": TRACK_B_SCHEMA,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "repository_base_commit": repository_base_commit,
        "policy_config": str(policy_config),
        "policy_config_sha256": _sha256(policy_config),
        "audit_script_sha256": _sha256(Path(__file__)),
        "track_b_contract_sha256": _sha256(
            Path("src/xtbflow/data/track_b.py")
        ),
        "input_record_count": len(records),
        "admitted_record_count": len(filtered.accepted),
        "quarantined_record_count": len(filtered.quarantined),
        "input_admission_counts": dict(sorted(admission_counts.items())),
        "quarantine_reason_counts": dict(sorted(reason_counts.items())),
        "strict_family_holdout": strict_family_holdout,
        "leakage_report": leakage_report,
        "leakage_error": leakage_error,
        "scientific_claim_allowed": False,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--policy-config",
        type=Path,
        default=Path("configs/data/track_b_admission_v1.json"),
    )
    parser.add_argument("--repository-base-commit", required=True)
    parser.add_argument(
        "--allow-family-overlap",
        action="store_true",
        help="check parent/input leakage but allow one family in multiple splits",
    )
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error(f"manifest does not exist: {args.manifest}")
    if not args.policy_config.is_file():
        parser.error(f"policy config does not exist: {args.policy_config}")
    report = build_report(
        args.manifest,
        strict_family_holdout=not args.allow_family_overlap,
        policy_config=args.policy_config,
        repository_base_commit=args.repository_base_commit,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 1 if report["leakage_error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
