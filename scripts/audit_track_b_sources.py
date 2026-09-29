#!/usr/bin/env python3
"""Audit Track-B source candidates and any locally supplied source assets."""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


REPORT_SCHEMA = "xtbflow-track-b-source-audit/v1"
CONFIG_SCHEMA = "xtbflow-track-b-source-candidates/v1"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def parse_assets(values: Iterable[str]) -> dict[str, Path]:
    assets: dict[str, Path] = {}
    for value in values:
        source_id, separator, raw_path = value.partition("=")
        if not separator or not source_id or not raw_path:
            raise ValueError(
                "--asset values must use SOURCE_ID=/absolute/or/relative/path"
            )
        if source_id in assets:
            raise ValueError(f"duplicate --asset source ID: {source_id}")
        assets[source_id] = Path(raw_path)
    return assets


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(f"unsupported source config schema: {path}")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("source candidate config must contain sources")
    source_ids = [row.get("source_id") for row in sources]
    if any(not isinstance(value, str) or not value for value in source_ids):
        raise ValueError("every source candidate needs a source_id")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("source candidate IDs must be unique")
    return payload


def audit_source(
    source: dict[str, Any],
    supplied_asset: Path | None,
) -> tuple[dict[str, Any], str | None]:
    result = json.loads(json.dumps(source, sort_keys=True))
    if supplied_asset is None:
        result["local_asset_observed"] = False
        return result, None
    if not supplied_asset.is_file():
        return result, f"asset does not exist: {source['source_id']}"
    observed_size = supplied_asset.stat().st_size
    observed_sha256 = hash_file(supplied_asset)
    expected_size = source.get("expected_asset_size_bytes")
    expected_sha256 = source.get("expected_asset_sha256")
    matches_size = expected_size is None or observed_size == expected_size
    matches_sha256 = expected_sha256 is None or observed_sha256 == expected_sha256
    observed: dict[str, Any] = {
        "asset_name": supplied_asset.name,
        "size_bytes": observed_size,
        "sha256": observed_sha256,
        "matches_expected_size": matches_size,
        "matches_expected_sha256": matches_sha256,
    }
    if supplied_asset.suffix == ".jsonl":
        observed["nonempty_jsonl_rows"] = count_jsonl(supplied_asset)
    result["local_asset_observed"] = True
    result["observed_asset"] = observed
    if not matches_size:
        return result, f"asset size mismatch: {source['source_id']}"
    if not matches_sha256:
        return result, f"asset hash mismatch: {source['source_id']}"
    return result, None


def build_report(
    config_path: Path,
    *,
    supplied_assets: dict[str, Path],
    audit_date: str,
    repository_base_commit: str,
) -> dict[str, Any]:
    config = load_config(config_path)
    configured_ids = {row["source_id"] for row in config["sources"]}
    unknown_assets = set(supplied_assets) - configured_ids
    if unknown_assets:
        raise ValueError(
            f"assets supplied for unknown source IDs: {sorted(unknown_assets)}"
        )
    sources: list[dict[str, Any]] = []
    failures: list[str] = []
    for source in config["sources"]:
        audited, failure = audit_source(
            source,
            supplied_assets.get(source["source_id"]),
        )
        sources.append(audited)
        if failure:
            failures.append(failure)
    admitted = [
        row
        for row in sources
        if str(row.get("decision", "")).startswith("admit_")
    ]
    return {
        "schema_version": REPORT_SCHEMA,
        "audit_date": audit_date,
        "repository_base_commit": repository_base_commit,
        "source_config": str(config_path),
        "source_config_sha256": hash_file(config_path),
        "source_count": len(sources),
        "admitted_source_count": len(admitted),
        "asset_failures": failures,
        "sources": sources,
        "gate_to_open_first_development_corpus": [
            "pin reconstructible reactant-visible graph and geometry bytes",
            "freeze charge, multiplicity, microstate and environment provenance",
            "freeze atom mapping, product graph and event-label representation",
            "pin product-geometry availability, transition geometry units and reference protocol",
            "assign parent, family and split groups before training",
            "pass exact-input, source-record, payload, parent and family leakage audit",
        ],
        "scientific_claim_allowed": False,
        "claim_boundary": (
            "Data-governance evidence only; not model training, TS validation, "
            "or a scientific performance result."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/track_b_source_candidates_v1.json"),
    )
    parser.add_argument("--asset", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-date", default=date.today().isoformat())
    parser.add_argument("--repository-base-commit", required=True)
    args = parser.parse_args()
    try:
        assets = parse_assets(args.asset)
        report = build_report(
            args.config,
            supplied_assets=assets,
            audit_date=args.audit_date,
            repository_base_commit=args.repository_base_commit,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 1 if report["asset_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
