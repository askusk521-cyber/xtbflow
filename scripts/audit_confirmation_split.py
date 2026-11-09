#!/usr/bin/env python3
"""Audit that the qualified baseline pool is a frozen family-held-out split."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from xtbflow.data.track_b import load_track_b_jsonl


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = load_track_b_jsonl(str(args.manifest))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    selected = [record for record in records if record.parent_reaction_id in set(protocol["parent_ids"])]
    if not selected:
        raise ValueError("confirmation selection is empty")
    if any(record.admission != "development_test" for record in selected):
        raise ValueError("confirmation selection contains non-test records")
    selected_parents = {record.parent_reaction_id for record in selected}
    selected_families = {record.family_id for record in selected}
    other = [record for record in records if record.parent_reaction_id not in selected_parents]
    overlap_families = selected_families & {record.family_id for record in other}
    report: dict[str, Any] = {
        "schema": "xtbflow-confirmation-split-audit/v1",
        "status": "pass" if not overlap_families else "fail",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "protocol": str(args.protocol),
        "protocol_sha256": _sha256(args.protocol),
        "selected_parent_ids": sorted(selected_parents),
        "selected_family_ids": sorted(selected_families),
        "selected_record_count": len(selected),
        "selected_parent_count": len(selected_parents),
        "selected_family_count": len(selected_families),
        "family_overlap_with_non_selected_records": sorted(overlap_families),
        "claim_limits": [
            "This proves only the declared family-held-out split and denominator.",
            "It does not prove chemical independence beyond the manifest audit or any scientific endpoint.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "parents": report["selected_parent_count"], "families": report["selected_family_count"]}, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
