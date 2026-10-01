#!/usr/bin/env python3
"""Prepare or execute a bounded independent pilot.

The dry-run path is useful before a driver is qualified.  The execute path is
deliberately fail-closed: it requires a ready frozen manifest, a qualified
driver and an enabled shared budget, and it never invents a calculator when
those records are absent.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from xtbflow.data.independent_seeds import build_seed_manifest, load_seed_manifest  # noqa: E402
from xtbflow.data.records import canonical_hash  # noqa: E402


def _run_dir(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("--output must be an absolute run directory")
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return resolved
    raise ValueError("independent pilot output must be outside the repository")


def _config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config must contain a JSON object")
    return payload


def _manifest(config_path: Path, output: Path):
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        return load_seed_manifest(manifest_path)
    config = _config(config_path)
    return build_seed_manifest(config, freeze_id=f"dry-run-{canonical_hash(config)[:16]}")


def dry_run(config_path: Path, output: Path) -> dict:
    config = _config(config_path)
    manifest = _manifest(config_path, output)
    execution = config.get("execution", {})
    if not isinstance(execution, dict):
        raise ValueError("execution must be an object")
    cap = execution.get("raw_candidate_cap", 64)
    call_cap = execution.get("per_candidate_calculator_call_cap", 64)
    batch_size = execution.get("batch_size", 1024)
    for name, value in (("raw_candidate_cap", cap), ("per_candidate_calculator_call_cap", call_cap), ("batch_size", batch_size)):
        if type(value) is not int or value < 1:
            raise ValueError(f"execution.{name} must be a positive integer")
    systems = [
        {
            "seed_id": seed.seed_id,
            "parent_id": seed.parent_id,
            "raw_candidate_cap": cap,
            "per_candidate_calculator_call_cap": call_cap,
            "status": "not_searched",
            "failure_reason": "dry_run_no_qualified_driver",
        }
        for seed in manifest.seeds
    ]
    plan = {
        "schema": "xtbflow-independent-pilot-plan/v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config_path": str(config_path.resolve()),
        "config_hash": canonical_hash(config),
        "manifest_hash": manifest.manifest_hash,
        "manifest_status": manifest.status,
        "manifest_blockers": list(manifest.blockers),
        "status": "blocked" if manifest.status != "ready" else "ready_for_driver",
        "qc_calls": 0,
        "batch_size": batch_size,
        "systems": systems,
        "claim_limits": ["dry-run only", "no proposals or paths were executed"],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return plan


def execute(config_path: Path, output: Path, *, resume: bool) -> dict:
    config = _config(config_path)
    manifest = _manifest(config_path, output)
    execution = config.get("execution", {})
    if manifest.status != "ready":
        raise RuntimeError(f"execute refused: seed manifest is {manifest.status}: {', '.join(manifest.blockers)}")
    if execution.get("driver_qualified") is not True:
        raise RuntimeError("execute refused: independent search driver is not qualified")
    if execution.get("allow_execute") is not True:
        raise RuntimeError("execute refused: shared budget does not authorize execution")
    raise RuntimeError("execute refused: no qualified calculator adapter is installed in this software-only work package")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.resume and not args.execute:
        parser.error("--resume is valid only with --execute")
    try:
        output = _run_dir(args.output)
        if args.dry_run:
            result = dry_run(args.config.resolve(), output)
        else:
            result = execute(args.config.resolve(), output, resume=args.resume)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

