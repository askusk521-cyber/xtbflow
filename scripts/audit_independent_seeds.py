#!/usr/bin/env python3
"""Audit frozen independent-reactant seeds before any channel search."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from xtbflow.data.independent_seeds import (
    INDEPENDENT_SEED_SCHEMA,
    SeedLeakageError,
    audit_seed_leakage,
    load_seed_jsonl,
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
    strict_family_holdout: bool,
    pilot_config: Path,
    repository_base_commit: str,
) -> dict[str, Any]:
    rows = load_seed_jsonl(str(manifest))
    leakage_error: str | None = None
    leakage_report: dict[str, Any] = {}
    try:
        leakage_report = audit_seed_leakage(
            rows,
            strict_family_holdout=strict_family_holdout,
        )
    except SeedLeakageError as exc:
        leakage_error = str(exc)
    return {
        "schema_version": "xtbflow-independent-seed-audit/v1",
        "record_schema": INDEPENDENT_SEED_SCHEMA,
        "manifest": str(manifest),
        "manifest_sha256": hash_file(manifest),
        "pilot_config": str(pilot_config),
        "pilot_config_sha256": hash_file(pilot_config),
        "repository_base_commit": repository_base_commit,
        "audit_script_sha256": hash_file(Path(__file__)),
        "seed_contract_sha256": hash_file(
            Path("src/xtbflow/data/independent_seeds.py")
        ),
        "record_count": len(rows),
        "strict_family_holdout": strict_family_holdout,
        "leakage_report": leakage_report,
        "leakage_error": leakage_error,
        "search_may_start": bool(rows) and leakage_error is None,
        "scientific_claim_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pilot-config",
        type=Path,
        default=Path("configs/seeds/pilot.yaml"),
    )
    parser.add_argument("--repository-base-commit", required=True)
    parser.add_argument("--allow-family-overlap", action="store_true")
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error(f"manifest does not exist: {args.manifest}")
    if not args.pilot_config.is_file():
        parser.error(f"pilot config does not exist: {args.pilot_config}")
    report = build_report(
        args.manifest,
        strict_family_holdout=not args.allow_family_overlap,
        pilot_config=args.pilot_config,
        repository_base_commit=args.repository_base_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 1 if report["leakage_error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
