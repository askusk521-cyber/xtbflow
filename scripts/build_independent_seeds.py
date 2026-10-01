#!/usr/bin/env python3
"""Freeze independent reactant seeds without running a calculator."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from xtbflow.data.independent_seeds import build_seed_manifest, write_seed_jsonl, write_seed_manifest  # noqa: E402
from xtbflow.data.records import canonical_hash  # noqa: E402


def _output_dir(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("--output must be an absolute run directory")
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return resolved
    raise ValueError("independent seed output must be outside the repository")


def _load_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config must contain a JSON object")
    return payload


def freeze(config_path: Path, output: Path, *, freeze_id: str | None = None, require_ready: bool = False) -> dict:
    config = _load_config(config_path)
    output.mkdir(parents=True, exist_ok=True)
    config_hash = canonical_hash(config)
    selected_freeze_id = freeze_id or f"freeze-{config_hash[:16]}"
    manifest = build_seed_manifest(config, freeze_id=selected_freeze_id)
    if require_ready and manifest.status != "ready":
        raise RuntimeError(f"seed manifest is blocked: {', '.join(manifest.blockers)}")
    write_seed_manifest(output / "manifest.json", manifest)
    write_seed_jsonl(output / "manifest.jsonl", manifest)
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    freeze_record = {
        "schema": "xtbflow-independent-seed-freeze/v1",
        "freeze_id": manifest.freeze_id,
        "frozen_at": now,
        "config_path": str(config_path.resolve()),
        "config_hash": config_hash,
        "manifest_hash": manifest.manifest_hash,
        "generation_version": manifest.generation_version,
        "qc_calls": 0,
        "status": manifest.status,
        "claim_limits": ["reactant-side input contract only", "no path or chemical validation"],
    }
    (output / "freeze_record.json").write_text(json.dumps(freeze_record, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema": "xtbflow-independent-seed-admission/v1",
        "status": manifest.status,
        "seed_count": len(manifest.seeds),
        "parent_count": manifest.parent_count,
        "max_seeds_per_parent": manifest.max_seeds_per_parent,
        "blockers": list(manifest.blockers),
        "manifest_hash": manifest.manifest_hash,
        "config_hash": config_hash,
        "qc_calls": 0,
        "excluded_records": [],
    }
    (output / "admission_report.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--freeze", action="store_true", help="freeze reactant-side records; never invokes QC")
    parser.add_argument("--output", required=True)
    parser.add_argument("--freeze-id")
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        parser.error("this bounded entry point requires --freeze")
    try:
        output = _output_dir(args.output)
        report = freeze(args.config.resolve(), output, freeze_id=args.freeze_id, require_ready=args.require_ready)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

