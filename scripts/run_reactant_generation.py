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
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import torch

from xtbflow.evaluation.generation import (
    generate_candidates,
    generate_repeated_candidates_with_accounting,
    load_generation_model,
    load_reactant_inputs,
)
from xtbflow.evaluation.terminal import GenerationProtocol, INITIAL_STATE_POLICIES
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


def _status_for_failure_stage(stage: str) -> str:
    """Map a runner stage to the durable failure status contract."""

    if stage in {"generation_preflight", "not_run"}:
        return "not_run"
    if stage == "report_write":
        return "report_write_failed"
    if stage == "settlement":
        return "settlement_failed"
    if stage in {"budget", "budget_exhausted"}:
        return "budget_exhausted"
    return "execution_failed"


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
    parser.add_argument(
        "--initial-state-policy",
        choices=INITIAL_STATE_POLICIES,
        default="deterministic_reactant",
        help="stochastic source for repeated attempts; deterministic runs produce one trajectory",
    )
    parser.add_argument(
        "--attempt-seeds",
        default=None,
        help="comma-separated integer seeds; repeated attempts require a stochastic policy",
    )
    parser.add_argument("--noise-scale", type=float, default=0.05)
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=0,
        help="reserved experimental option; nonzero latent inputs are currently unsupported",
    )
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
    if args.noise_scale < 0 or not math.isfinite(args.noise_scale):
        raise SystemExit("noise-scale must be finite and nonnegative")
    if args.latent_dim < 0:
        raise SystemExit("latent-dim must be nonnegative")
    if args.attempt_seeds is None:
        attempt_seeds = (0,)
    else:
        try:
            attempt_seeds = tuple(int(item.strip()) for item in args.attempt_seeds.split(",") if item.strip())
        except ValueError as exc:
            raise SystemExit("attempt-seeds must be comma-separated integers") from exc
        if not attempt_seeds:
            raise SystemExit("attempt-seeds must contain at least one integer")
    try:
        protocol = GenerationProtocol(
            candidate_cap=args.candidate_cap,
            attempt_seeds=attempt_seeds,
            initial_state_policy=args.initial_state_policy,
            latent_dim=args.latent_dim,
        )
        protocol.validate()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
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
        "failure_stage": None,
        "failure_reason": None,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_host": platform.node(),
        "git": None,
        "input_locator": _safe_locator(args.inputs, root),
        "input_sha256": None,
        "checkpoint_locator": _safe_locator(args.checkpoint, root),
        "checkpoint_sha256": None,
        "runtime_config_locator": _safe_locator(args.runtime_config, root),
        "runtime_config_sha256": None,
        "mode": args.mode,
        "steps": args.steps,
        "candidate_cap": args.candidate_cap,
        "generation_protocol": protocol.as_dict(),
        "noise_scale": args.noise_scale,
        "device": str(device),
        "conservation_projection": not args.no_conservation_projection,
        "target_fields_read": False,
    }
    _write_atomic(manifest, prepared)
    stage_start = time.monotonic()
    stage = "initialization"
    try:
        prepared.update(
            {
                "git": _git_identity(),
                "input_sha256": _sha256(args.inputs),
                "checkpoint_sha256": _sha256(args.checkpoint),
                "runtime_config_sha256": _sha256(args.runtime_config),
            }
        )
        _write_atomic(manifest, prepared)
        reactants = load_reactant_inputs(args.inputs)
        model, max_atoms, restore = load_generation_model(
            args.checkpoint, args.runtime_config, device=device, dtype=torch.float32
        )
        records: list[dict[str, Any]] = []
        stage = "integration"
        for reactant in reactants:
            if protocol.stochastic:
                repeated = generate_repeated_candidates_with_accounting(
                    model,
                    reactant,
                    max_atoms=max_atoms,
                    control_mode=args.mode,
                    protocol=protocol,
                    steps=args.steps,
                    conservation_projection=not args.no_conservation_projection,
                    device=device,
                    dtype=torch.float32,
                    noise_scale=args.noise_scale,
                )
                generated = repeated.candidates
                attempted = repeated.attempted
                accepted = repeated.unique_candidates
                decode_accepted = repeated.decode_accepted
                decode_failed = repeated.decode_failed
                invalid = repeated.invalid
                execution_failed = repeated.execution_failed
                duplicate_geometry = repeated.duplicate_geometry
                duplicate_event = repeated.duplicate_event
                event_unique_candidates = repeated.event_unique_candidates
                rejection_reasons = list(repeated.rejection_reasons)
            else:
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
                    initial_state_policy=protocol.initial_state_policy,
                    attempt_seed=protocol.attempt_seeds[0],
                    noise_scale=args.noise_scale,
                )
                generated = batch.candidates
                attempted = batch.attempted
                accepted = batch.accepted
                decode_accepted = batch.accepted
                decode_failed = batch.rejected
                invalid = 0
                execution_failed = 0
                duplicate_geometry = 0
                duplicate_event = 0
                event_unique_candidates = accepted
                rejection_reasons = list(batch.rejection_reasons)
            rejected = decode_failed + invalid + execution_failed
            if execution_failed:
                record_status = "generation_failed"
            elif invalid:
                record_status = "invalid"
            elif decode_failed:
                record_status = "decode_failed"
            elif not generated:
                record_status = "empty"
            else:
                record_status = "completed"
            records.append(
                {
                    "record_id": reactant.record_id,
                    "parent_reaction_id": reactant.parent_reaction_id,
                    "family_id": reactant.family_id,
                    "input_fingerprint": reactant.input_fingerprint,
                    "status": record_status,
                    "attempted": attempted,
                    "accepted": accepted,
                    "decode_accepted": decode_accepted,
                    "decode_failed": decode_failed,
                    "invalid": invalid,
                    "execution_failed": execution_failed,
                    "duplicate_geometry": duplicate_geometry,
                    "duplicate_event": duplicate_event,
                    "unique_candidates": accepted,
                    "event_unique_candidates": event_unique_candidates,
                    "rejected": rejected,
                    "rejection_rate": rejected / attempted if attempted else 0.0,
                    "rejection_reasons": rejection_reasons,
                    "candidates": [candidate.to_mapping() for candidate in generated],
                }
            )
        stage = "report_write"
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
            "generation_protocol": protocol.as_dict(),
            "noise_scale": args.noise_scale,
            "conservation_projection": not args.no_conservation_projection,
            "records": records,
            "resource_budget": {
                "max_concurrent_jobs": 1,
                "max_model_trajectories": len(reactants) * len(protocol.attempt_seeds),
                "max_euler_steps": len(reactants) * len(protocol.attempt_seeds) * args.steps,
            },
            "elapsed_seconds": time.monotonic() - stage_start,
            "claim_limit": "generation software output only; no reference scoring or physical validation",
            "failure_stage": None,
            "failure_reason": None,
        }
        _write_atomic(output, report)
        prepared["status"] = "completed"
        _write_atomic(manifest, prepared)
    except Exception as exc:
        prepared["status"] = _status_for_failure_stage(stage)
        prepared["failure_stage"] = stage
        prepared["failure_reason"] = f"{type(exc).__name__}: {exc}"
        # Keep the original failure visible to the caller even if a broken
        # filesystem prevents the manifest update.  A report-write failure is
        # therefore never relabelled as a generic execution failure.
        try:
            _write_atomic(manifest, prepared)
        except Exception:
            pass
        raise
    print(
        json.dumps(
            {
                "status": "completed",
                "records": len(reactants),
                "accepted_decodes": sum(item["decode_accepted"] for item in records),
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
