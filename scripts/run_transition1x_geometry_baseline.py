#!/usr/bin/env python3
"""Run the hash-gated endpoint-conditioned Transition1x geometry baseline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess

from xtbflow.data.transition1x_audit import audit_transition1x_pickle, hash_file, load_transition1x_pickle
from xtbflow.evaluation.transition1x_geometry import evaluate_transition1x_guesses


def _source(config_path: Path, source_id: str) -> dict:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    matches = [row for row in payload.get("sources", []) if row.get("source_id") == source_id]
    if len(matches) != 1:
        raise ValueError(f"expected one source config row for {source_id}")
    return dict(matches[0])


def _git_identity() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True).stdout.splitlines()
    return {"commit": commit, "dirty": bool(dirty), "dirty_entries": dirty}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/data/track_b_source_candidates_v1.json"))
    parser.add_argument("--source-id", default="transition1x_preprocessed")
    args = parser.parse_args()
    source = _source(args.config, args.source_id)
    audit = audit_transition1x_pickle(
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
    report = evaluate_transition1x_guesses(
        load_transition1x_pickle(
            args.asset,
            expected_sha256=source["expected_asset_sha256"],
            expected_size_bytes=source.get("expected_asset_size_bytes"),
        )
    )
    report.update(
        {
            "schema": "xtbflow-transition1x-geometry-baseline/v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "execution_host": platform.node(),
            "git": _git_identity(),
            "data_status": "quarantine_endpoint_geometry_diagnostic",
            "claim_limit": "Endpoint-conditioned geometry evidence only; no product-free event, chemical accuracy, or reaction-mechanism claim.",
            "source": {
                "source_id": args.source_id,
                "source_revision": source["source_revision"],
                "asset_path": str(args.asset),
                "asset_sha256": hash_file(args.asset),
                "asset_size_bytes": args.asset.stat().st_size,
                "audit": audit,
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"schema": report["schema"], "data_status": report["data_status"], "n_records": report["n_records"], "complement": report["split"]["complement_diagnostic"]["n"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

