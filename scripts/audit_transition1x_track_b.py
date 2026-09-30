#!/usr/bin/env python3
"""Audit a pinned Transition1x pickle against the Track-B source gate."""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
from typing import Any

from xtbflow.data.transition1x_audit import (
    Transition1xAuditError,
    audit_transition1x_pickle,
    hash_file,
)


def load_source(config_path: Path, source_id: str) -> dict[str, Any]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    matches = [
        row for row in payload.get("sources", []) if row.get("source_id") == source_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one source config row for {source_id}")
    return dict(matches[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/track_b_source_candidates_v1.json"),
    )
    parser.add_argument("--source-id", default="transition1x_preprocessed")
    parser.add_argument("--repository-base-commit", required=True)
    parser.add_argument("--audit-date", default=date.today().isoformat())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        source = load_source(args.config, args.source_id)
        report = audit_transition1x_pickle(
            args.asset,
            expected_sha256=source["expected_asset_sha256"],
            expected_size_bytes=source.get("expected_asset_size_bytes"),
            source_revision=source["source_revision"],
            source_locator=source["source_locator"],
            license_record=source["license_record"],
            upstream_code_commit=source["upstream_code_commit"],
            expected_record_count=source.get("declared_record_count"),
            expected_md5=source.get("zenodo_md5"),
            license_status=source.get("license_status", "unknown"),
        )
        report["audit_date"] = args.audit_date
        report["repository_base_commit"] = args.repository_base_commit
        report["source_config"] = str(args.config)
        report["source_config_sha256"] = hash_file(args.config)
    except (KeyError, OSError, ValueError, Transition1xAuditError) as exc:
        parser.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
