#!/usr/bin/env python3
"""Run a bounded reactant-only rollout on the paired pilot.

This command is intentionally inference-only.  A checkpoint is evaluated from
the reactant view at ``t=0``; labels are handed to the benchmark only after all
candidate attempts have been written in memory.  The script does not enlarge
the paired pilot or launch a calculator.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import torch
import numpy as np

from xtbflow.data.dft_da import load_dft_da_samples
from xtbflow.data.paired_loader import load_paired_pilot
from xtbflow.evaluation import BlindRolloutConfig, CandidateAttempt, BlindRolloutResult, benchmark_rollouts, blind_rollout, make_reactant_input
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
    "split_index_sha256",
    "runtime_config_sha256",
    "split",
    "seed",
    "rollout",
    "geometry_threshold",
    "bootstrap_resamples",
    "checkpoints",
    "code_commit",
    "source_archive_sha256",
    "git_dirty",
    "device",
    "skip_label_invariance",
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


def _result_from_dict(payload: dict[str, Any]) -> BlindRolloutResult:
    return BlindRolloutResult(
        record_id=payload["record_id"],
        parent_reaction_id=payload["parent_reaction_id"],
        arm_id=payload["arm_id"],
        seed=int(payload["seed"]),
        nfe=int(payload["nfe"]),
        candidate_cap=int(payload["candidate_cap"]),
        actual_candidate_attempts=int(payload["actual_candidate_attempts"]),
        actual_flow_evaluations=int(payload["actual_flow_evaluations"]),
        attempts=tuple(CandidateAttempt(**row) for row in payload["attempts"]),
        accepted=tuple(CandidateAttempt(**row) for row in payload["accepted"]),
        physical_validation_calls=int(payload.get("physical_validation_calls", 0)),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/paired_pilot_v1/manifest.jsonl"))
    parser.add_argument("--split-index", type=Path, default=Path("data/manifests/paired_pilot_v1/split_index.json"))
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
    parser.add_argument("--initial-event-noise-scale", type=float, default=0.0)
    parser.add_argument("--initial-geometry-noise-scale", type=float, default=0.0)
    parser.add_argument("--skip-label-invariance", action="store_true")
    parser.add_argument("--resume", action="store_true", help="reuse an existing run directory only when its frozen config matches")
    return parser


def main() -> int:
    args = _parser().parse_args()
    checkpoints = _parse_checkpoints(args.checkpoint)
    if args.nfe < 1 or args.candidate_cap < 1 or args.seed < 0 or args.geometry_threshold <= 0 or args.bootstrap_resamples < 0 or args.initial_event_noise_scale < 0 or args.initial_geometry_noise_scale < 0:
        raise SystemExit("nfe, candidate-cap, seed, geometry-threshold and bootstrap-resamples are invalid")
    output = args.output.expanduser()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise SystemExit(f"refusing to overwrite non-empty run directory; choose a unique output or pass --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    source_archive = args.cache_root / "DATASET_DA_F.tar.gz"
    if not args.manifest.is_file() or not args.split_index.is_file() or not args.runtime_config.is_file():
        raise SystemExit("manifest, split-index and runtime-config must exist before freezing the run")
    if not all(path.is_file() for path in checkpoints.values()):
        raise SystemExit("every checkpoint must exist before freezing the run")
    if not source_archive.is_file():
        raise SystemExit(f"source archive is required for a frozen run: {source_archive}")
    source_archive_sha256 = _sha256(source_archive)
    checkpoint_hashes = {arm: _sha256(path) for arm, path in checkpoints.items()}
    if len(set(checkpoint_hashes.values())) != len(checkpoint_hashes):
        raise SystemExit("each requested arm must use a distinct checkpoint hash")
    git = _git_identity(root)
    if git["dirty"]:
        raise SystemExit("blind rollout requires a clean git worktree; commit code and inventory before running")
    resolved_device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    run_config = {
        "schema": "xtbflow-blind-rollout-run/v1",
        "run_id": args.run_id,
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "split_index": str(args.split_index),
        "split_index_sha256": _sha256(args.split_index),
        "runtime_config": str(args.runtime_config),
        "runtime_config_sha256": _sha256(args.runtime_config),
        "split": args.split,
        "seed": args.seed,
        "rollout": asdict(BlindRolloutConfig(nfe=args.nfe, candidate_cap=args.candidate_cap, initial_event_noise_scale=args.initial_event_noise_scale, initial_geometry_noise_scale=args.initial_geometry_noise_scale)),
        "geometry_threshold": args.geometry_threshold,
        "bootstrap_resamples": args.bootstrap_resamples,
        "checkpoints": {arm: {"path": str(path), "sha256": checkpoint_hashes[arm], "mode": ARM_MODES[arm]} for arm, path in checkpoints.items()},
        "git": git,
        "code_commit": git["commit"],
        "git_dirty": git["dirty"],
        "device": resolved_device,
        "skip_label_invariance": args.skip_label_invariance,
        "source_archive": str(source_archive),
        "source_archive_sha256": source_archive_sha256,
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
        if args.resume:
            raise SystemExit("--resume requires an existing run_config.json")
        config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    samples, source_audit = load_dft_da_samples(args.cache_root)
    # The frozen manifest is the authoritative split/identity view.  The
    # parser audit above only reconstructs source arrays and validates the
    # archive; it must not silently choose a different split.
    loader = load_paired_pilot(args.manifest, args.cache_root, batch_size=8)
    all_samples = list(loader.samples)
    samples = [sample for sample in all_samples if sample.record.admission.removeprefix("development_") == args.split]
    if not samples:
        raise SystemExit(f"split is empty: {args.split}")
    split_payload = json.loads(args.split_index.read_text(encoding="utf-8"))
    split_entry = split_payload.get(args.split)
    if not isinstance(split_entry, dict):
        raise SystemExit(f"split index has no entry for {args.split}")
    observed_ids = sorted(sample.record.record_id for sample in samples)
    observed_parents = sorted({sample.record.parent_reaction_id for sample in samples})
    observed_families = sorted({sample.record.family_id for sample in samples})
    if (
        split_entry.get("count") != len(observed_ids)
        or sorted(split_entry.get("record_ids", [])) != observed_ids
        or sorted(split_entry.get("parent_groups", [])) != observed_parents
        or sorted(split_entry.get("family_groups", [])) != observed_families
    ):
        raise SystemExit(f"split index does not match manifest/cache identities for {args.split}")
    max_atoms = max(len(sample.atomic_numbers) for sample in all_samples)
    device = torch.device(resolved_device)
    runtime_config = JointFlowRuntimeConfig.load(args.runtime_config)
    rollout_config = BlindRolloutConfig(nfe=args.nfe, candidate_cap=args.candidate_cap, initial_event_noise_scale=args.initial_event_noise_scale, initial_geometry_noise_scale=args.initial_geometry_noise_scale)
    arm_reports: dict[str, Any] = {}
    run_start = time.monotonic()
    for arm, checkpoint in checkpoints.items():
        projector = total_electron_projector(("C",) * max_atoms, device=device)
        model = runtime_config.build(projector).to(device=device, dtype=torch.float32)
        load_joint_checkpoint(checkpoint, model, restore_rng=False)
        model.eval()
        rollouts = []
        failures = []
        invariance_flow_evaluations = 0
        ledger_path = output / f"{arm}.jsonl"
        arm_start = time.monotonic()
        prior: dict[str, BlindRolloutResult] = {}
        if args.resume and ledger_path.is_file():
            for line in ledger_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") == "completed" and isinstance(row.get("result"), dict):
                    prior[row["record_id"]] = _result_from_dict(row["result"])
            rollouts.extend(prior.values())
        ledger = ledger_path.open("a", encoding="utf-8")
        for sample in samples:
            if sample.record.record_id in prior:
                continue
            row_start = time.monotonic()
            try:
                reactant = make_reactant_input(sample, max_atoms, device=device)
                result = blind_rollout(model, reactant, arm_id=ARM_MODES[arm], seed=args.seed, config=rollout_config)
                if not args.skip_label_invariance:
                    altered = replace(sample, product_bonds=np.zeros_like(sample.product_bonds), ts_coordinates=np.zeros_like(sample.ts_coordinates))
                    altered_result = blind_rollout(model, make_reactant_input(altered, max_atoms, device=device), arm_id=ARM_MODES[arm], seed=args.seed, config=rollout_config)
                    invariance_flow_evaluations += altered_result.actual_flow_evaluations
                    if [item.candidate_hash for item in altered_result.attempts] != [item.candidate_hash for item in result.attempts]:
                        raise RuntimeError("label_replacement_invariance_failed")
                rollouts.append(result)
                ledger.write(json.dumps({"status": "completed", "record_id": result.record_id, "wall_time_seconds": time.monotonic() - row_start, "result": result.to_dict()}, ensure_ascii=False, sort_keys=True) + "\n")
            except Exception as exc:
                failure = {"status": "failure", "record_id": sample.record.record_id, "error_type": type(exc).__name__, "error": str(exc), "wall_time_seconds": time.monotonic() - row_start}
                failures.append(failure)
                ledger.write(json.dumps(failure, ensure_ascii=False, sort_keys=True) + "\n")
        ledger.close()
        by_record = {sample.record.record_id: sample for sample in samples}
        report = benchmark_rollouts(
            rollouts,
            by_record,
            geometry_threshold=args.geometry_threshold,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.seed,
        ) if rollouts else {
            "schema": "xtbflow-blind-rollout-benchmark/v1",
            "record_count": 0,
            "parent_count": 0,
            "record_weighted": {},
            "parent_macro": {},
            "parent_metrics": [],
            "record_metrics": [],
            "parent_bootstrap": {},
        }
        report["rollouts"] = [result.to_dict() for result in rollouts]
        report["failures"] = failures
        report["status"] = "completed" if not failures else ("partial" if rollouts else "failed")
        report["actual_candidate_attempts"] = sum(result.actual_candidate_attempts for result in rollouts)
        report["actual_flow_evaluations"] = sum(result.actual_flow_evaluations for result in rollouts)
        report["physical_validation_calls"] = sum(result.physical_validation_calls for result in rollouts)
        report["label_invariance_flow_evaluations"] = invariance_flow_evaluations
        report["label_invariance_candidate_attempts"] = invariance_flow_evaluations // args.nfe if args.nfe else 0
        report["wall_time_seconds"] = time.monotonic() - arm_start
        arm_reports[arm] = report

    evidence = {
        "schema": "xtbflow-blind-rollout-evidence/v1",
        "status": "completed" if all(report.get("status") == "completed" for report in arm_reports.values()) else "partial",
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
        "physical_validation_calls": 0,
        "control_matrix": {"conserved_independent": "implemented", "complete_serial": "implemented", "same_integrator_one_way": "implemented", "joint_bidirectional": "implemented", "matched_unconstrained": "pending separate unconstrained checkpoint/control"},
        "scientific_comparison_status": "incomplete_pending_matched_unconstrained",
    }
    evidence_path = output / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": evidence["status"], "run_id": args.run_id, "records": len(samples), "parents": evidence["parent_count"], "evidence": str(evidence_path)}, sort_keys=True))
    return 0 if evidence["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
