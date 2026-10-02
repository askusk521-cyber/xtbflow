#!/usr/bin/env python3
"""Run a bounded reactant-only rollout on the paired pilot.

This command is intentionally inference-only.  A checkpoint is evaluated from
the reactant view at ``t=0``; labels are handed to the benchmark only after all
candidate attempts have been written in memory.  The script does not enlarge
the paired pilot or launch a calculator.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import torch

from xtbflow.data.dft_da import load_dft_da_samples, with_splits
from xtbflow.evaluation import BlindRolloutConfig, benchmark_rollouts, blind_rollout, make_reactant_input
from xtbflow.models import JointFlowRuntimeConfig, total_electron_projector
from xtbflow.training import load_joint_checkpoint


ARM_MODES = {
    "conserved_independent": "both_off",
    "serial_event_to_geometry": "serial_independent",
    "joint_unidirectional": "joint_unidirectional",
    "joint_event_geometry": "joint_bidirectional",
}
FROZEN_FIELDS = (
    "run_id",
    "manifest_sha256",
    "runtime_config_sha256",
    "split",
    "seed",
    "rollout",
    "geometry_threshold",
    "bootstrap_resamples",
    "checkpoints",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_identity(root: Path) -> dict[str, Any]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "dirty_entries": status}


def _parse_checkpoints(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"checkpoint must be ARM=PATH: {value}")
        arm, path = value.split("=", 1)
        if arm not in ARM_MODES:
            raise SystemExit(f"unknown arm {arm!r}; choose one of {sorted(ARM_MODES)}")
        if arm in result:
            raise SystemExit(f"duplicate checkpoint arm: {arm}")
        result[arm] = Path(path).expanduser()
    if not result:
        raise SystemExit("at least one ARM=PATH checkpoint is required")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/paired_pilot_v1/manifest.jsonl"))
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/models/joint_flow_runtime_v0.1.json"))
    parser.add_argument("--checkpoint", action="append", required=True, help="ARM=CHECKPOINT; may be repeated")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--nfe", type=int, default=4)
    parser.add_argument("--candidate-cap", type=int, default=64)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--geometry-threshold", type=float, default=0.50)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true", help="reuse an existing run directory only when its frozen config matches")
    return parser


def main() -> int:
    args = _parser().parse_args()
    checkpoints = _parse_checkpoints(args.checkpoint)
    if args.nfe < 1 or args.candidate_cap < 1 or args.seed < 0 or args.geometry_threshold <= 0 or args.bootstrap_resamples < 0:
        raise SystemExit("nfe, candidate-cap, seed, geometry-threshold and bootstrap-resamples are invalid")
    output = args.output.expanduser()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise SystemExit(f"refusing to overwrite non-empty run directory; choose a unique output or pass --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    run_config = {
        "schema": "xtbflow-blind-rollout-run/v1",
        "run_id": args.run_id,
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "runtime_config": str(args.runtime_config),
        "runtime_config_sha256": _sha256(args.runtime_config),
        "split": args.split,
        "seed": args.seed,
        "rollout": asdict(BlindRolloutConfig(nfe=args.nfe, candidate_cap=args.candidate_cap)),
        "geometry_threshold": args.geometry_threshold,
        "bootstrap_resamples": args.bootstrap_resamples,
        "checkpoints": {arm: {"path": str(path), "sha256": _sha256(path), "mode": ARM_MODES[arm]} for arm, path in checkpoints.items()},
        "git": _git_identity(root),
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__},
        "claim_limit": "development pilot blind rollout; no confirmatory chemistry claim",
    }
    config_path = output / "run_config.json"
    resumed_from: str | None = None
    if config_path.exists():
        if not args.resume:
            raise SystemExit(f"refusing to overwrite existing run; pass --resume explicitly: {output}")
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if any(old.get(field) != run_config.get(field) for field in FROZEN_FIELDS):
            raise SystemExit("--resume requires the same frozen run configuration")
        resumed_from = _sha256(config_path)
    else:
        config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    samples, source_audit = load_dft_da_samples(args.cache_root)
    samples = [sample for sample in with_splits(samples) if sample.record.admission.removeprefix("development_") == args.split]
    if not samples:
        raise SystemExit(f"split is empty: {args.split}")
    max_atoms = max(len(sample.atomic_numbers) for sample in samples)
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    runtime_config = JointFlowRuntimeConfig.load(args.runtime_config)
    rollout_config = BlindRolloutConfig(nfe=args.nfe, candidate_cap=args.candidate_cap)
    arm_reports: dict[str, Any] = {}
    run_start = time.monotonic()
    for arm, checkpoint in checkpoints.items():
        projector = total_electron_projector(("C",) * max_atoms, device=device)
        model = runtime_config.build(projector).to(device=device, dtype=torch.float32)
        load_joint_checkpoint(checkpoint, model, restore_rng=False)
        model.eval()
        rollouts = []
        for sample in samples:
            reactant = make_reactant_input(sample, max_atoms, device=device)
            rollouts.append(blind_rollout(model, reactant, arm_id=ARM_MODES[arm], seed=args.seed, config=rollout_config))
        by_record = {sample.record.record_id: sample for sample in samples}
        report = benchmark_rollouts(
            rollouts,
            by_record,
            geometry_threshold=args.geometry_threshold,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.seed,
        )
        report["rollouts"] = [result.to_dict() for result in rollouts]
        arm_reports[arm] = report

    evidence = {
        "schema": "xtbflow-blind-rollout-evidence/v1",
        "status": "completed",
        "run_id": args.run_id,
        "resumed_from_config_sha256": resumed_from,
        "run_config_sha256": _sha256(config_path),
        "source_audit": source_audit,
        "split": args.split,
        "record_count": len(samples),
        "parent_count": len({sample.record.parent_reaction_id for sample in samples}),
        "elapsed_seconds": time.monotonic() - run_start,
        "arms": arm_reports,
        "claim_limit": "blind reactant-only development benchmark; labels used only after generation for scoring",
    }
    evidence_path = output / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": evidence["status"], "run_id": args.run_id, "records": len(samples), "parents": evidence["parent_count"], "evidence": str(evidence_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
