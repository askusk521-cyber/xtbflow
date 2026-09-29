#!/usr/bin/env python3
"""Audit an exact origins candidate registry against the independent-seed gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from xtbflow.data.origin_readiness import (
    OriginReadinessError,
    audit_origin_candidate_registry,
)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--registry-locator", required=True)
    parser.add_argument("--registry-source-commit", required=True)
    parser.add_argument("--repository-base-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        payload = json.loads(args.registry.read_text(encoding="utf-8"))
        report = audit_origin_candidate_registry(
            payload,
            registry_locator=args.registry_locator,
            registry_source_commit=args.registry_source_commit,
            registry_sha256=hash_file(args.registry),
            repository_base_commit=args.repository_base_commit,
        )
        report["audit_script_sha256"] = hash_file(Path(__file__))
        report["readiness_contract_sha256"] = hash_file(
            Path("src/xtbflow/data/origin_readiness.py")
        )
        report["independent_seed_contract_sha256"] = hash_file(
            Path("src/xtbflow/data/independent_seeds.py")
        )
    except (OSError, json.JSONDecodeError, OriginReadinessError) as exc:
        parser.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
