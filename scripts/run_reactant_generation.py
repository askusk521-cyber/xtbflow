#!/usr/bin/env python3
"""Generate frozen candidates from a target-free reactant manifest.

The command deliberately has no scoring input.  Product/event/TS labels are
joined only by a later report step after this output has been copied or
sealed.  It is bounded by an explicit number of Euler steps and records the
actual decode count rather than claiming that ``candidate_cap`` was filled.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import torch

from xtbflow.evaluation.generation import (
    generate_candidates,
    load_generation_model,
    load_reactant_inputs,
)
from xtbflow.models import CONTROL_MODES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    entries = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(entries), "dirty_entries": entries}


def _safe_locator(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="reactant-only JSONL")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--runtime-config", type=Path, default=Path("configs/models/joint_flow_runtime_v0.1.json")
    )
    parser.add_argument("--mode", choices=CONTROL_MODES, default="joint_bidirectional")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--candidate-cap", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--no-conservation-projection",
        action="store_true",
        help="run the unconstrained diagnostic field; this does not validate chemistry",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.steps < 1 or args.candidate_cap < 1:
        raise SystemExit("steps and candidate-cap must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output = args.output.expanduser()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = output.with_name(f"{output.stem}.run.json")
    if manifest.exists():
        raise SystemExit(f"refusing to overwrite existing run manifest: {manifest}")
    root = Path(__file__).resolve().parents[1]
    prepared = {
        "schema": "xtbflow-reactant-generation-run/v1",
        "status": "prepared",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_host": platform.node(),
        "git": _git_identity(),
        "input_locator": _safe_locator(args.inputs, root),
        "input_sha256": _sha256(args.inputs),
        "checkpoint_locator": _safe_locator(args.checkpoint, root),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "runtime_config_locator": _safe_locator(args.runtime_config, root),
        "runtime_config_sha256": _sha256(args.runtime_config),
        "mode": args.mode,
        "steps": args.steps,
        "candidate_cap": args.candidate_cap,
        "device": str(device),
        "conservation_projection": not args.no_conservation_projection,
        "target_fields_read": False,
    }
    _write_atomic(manifest, prepared)
    stage_start = time.monotonic()
    try:
        reactants = load_reactant_inputs(args.inputs)
        model, max_atoms, restore = load_generation_model(
            args.checkpoint, args.runtime_config, device=device, dtype=torch.float32
        )
        records: list[dict[str, Any]] = []
        for reactant in reactants:
            batch = generate_candidates(
                model,
                reactant,
                max_atoms=max_atoms,
                control_mode=args.mode,
                steps=args.steps,
                candidate_cap=args.candidate_cap,
                conservation_projection=not args.no_conservation_projection,
                device=device,
                dtype=torch.float32,
            )
            records.append(
                {
                    "record_id": reactant.record_id,
                    "parent_reaction_id": reactant.parent_reaction_id,
                    "family_id": reactant.family_id,
                    "input_fingerprint": reactant.input_fingerprint,
                    "attempted": batch.attempted,
                    "accepted": batch.accepted,
                    "rejected": batch.rejected,
                    "rejection_rate": batch.rejection_rate,
                    "rejection_reasons": list(batch.rejection_reasons),
                    "candidates": [candidate.to_mapping() for candidate in batch.candidates],
                }
            )
        report = {
            "schema": "xtbflow-reactant-generation-evidence/v2",
            "status": "completed",
            "run_manifest": manifest.name,
            "execution_host": platform.node(),
            "git": _git_identity(),
            "runtime_config_locator": _safe_locator(args.runtime_config, root),
            "runtime_config_sha256": _sha256(args.runtime_config),
            "checkpoint_locator": _safe_locator(args.checkpoint, root),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "checkpoint_restore": restore,
            "runtime": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            "input_locator": _safe_locator(args.inputs, root),
            "input_sha256": _sha256(args.inputs),
            "input_schema": "xtbflow-reactant-input/v1",
            "target_fields_read": False,
            "target_label_policy": "reactant-only rows reject product/event/TS/reference fields",
            "record_count": len(reactants),
            "checkpoint_atom_count": max_atoms,
            "mode": args.mode,
            "steps": args.steps,
            "dt": 1.0 / args.steps,
            "candidate_cap": args.candidate_cap,
            "conservation_projection": not args.no_conservation_projection,
            "records": records,
            "resource_budget": {
                "max_concurrent_jobs": 1,
                "max_model_trajectories": len(reactants),
                "max_euler_steps": len(reactants) * args.steps,
            },
            "elapsed_seconds": time.monotonic() - stage_start,
            "claim_limit": "generation software output only; no reference scoring or physical validation",
        }
        _write_atomic(output, report)
        prepared["status"] = "completed"
        _write_atomic(manifest, prepared)
    except Exception as exc:
        prepared["status"] = "execution_failed"
        prepared["error"] = f"{type(exc).__name__}: {exc}"
        _write_atomic(manifest, prepared)
        raise
    print(
        json.dumps(
            {
                "status": "completed",
                "records": len(reactants),
                "accepted_decodes": sum(item["accepted"] for item in records),
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
