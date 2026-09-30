#!/usr/bin/env python3
"""Run the first bounded real-data C/D/E joint-flow training matrix.

The runner consumes only the reactant view during optimization.  Product/event
and TS coordinates are loaded separately as labels to construct a flow-matching
target.  Every C, D and E arm receives the same parent/family split, examples,
optimizer schedule, and per-seed initial state; only the registered control
mode changes.  The output is development evidence for issue 58, not a
confirmatory chemistry benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import socket
import subprocess
import sys
import time
from typing import Any, Iterable
from uuid import uuid4

import numpy as np
import torch

from xtbflow.data.dft_da import DftDaSample, load_dft_da_samples, with_splits, write_manifest
from xtbflow.data.records import canonical_hash
from xtbflow.data.track_b import audit_track_b_leakage, write_track_b_jsonl
from xtbflow.evaluation.grouped import paired_group_comparison, summarize_by_group
from xtbflow.models import JointFlowRuntimeConfig, pack_be, total_electron_projector, upper_triangle_indices
from xtbflow.proposals import enumerate_core_events
from xtbflow.proposals.roles import infer_roles
from xtbflow.training import joint_flow_loss, model_state_identity, save_joint_checkpoint
from mechai.data.events import LewisState


ARM_SPECS = (
    ("conserved_independent", "both_off", True),
    ("serial_event_to_geometry", "serial_independent", True),
    ("current_event_to_geometry", "joint_unidirectional", True),
    ("joint_event_geometry", "joint_bidirectional", True),
    ("unconstrained_event", "joint_bidirectional", False),
)
ELEMENT_INDEX = {1: 0, 6: 1, 7: 2, 8: 3}
VALENCE = {1: 1.0, 6: 4.0, 7: 5.0, 8: 6.0}
FORBIDDEN_INPUT_KEYS = {
    "product", "mapped_product", "product_coordinates", "ts_geometry", "reference_ts",
    "reference_mode", "active_water_from_ts", "target_derived_solvent", "sealed_test_label",
}


@dataclass(frozen=True)
class TensorSample:
    sample: DftDaSample
    split: str
    event_state: torch.Tensor
    target_event: torch.Tensor
    coordinates: torch.Tensor
    target_geometry: torch.Tensor
    node_features: torch.Tensor
    atom_mask: torch.Tensor
    event_mask: torch.Tensor
    condition_features: torch.Tensor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_text(path: Path, text: str) -> None:
    """Write a complete file in the run directory before replacing its name."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:12]}"


def _start_run(output_root: Path, requested_run_id: str | None = None) -> tuple[str, Path, dict[str, Any]]:
    """Create a fresh output directory and durable stage state.

    A training output is an evidence artifact.  Reusing a directory would
    silently mix checkpoints and reports from different source/configuration
    identities, so an existing path is rejected even when it is non-empty.
    """

    run_id = requested_run_id.strip() if isinstance(requested_run_id, str) and requested_run_id.strip() else _new_run_id()
    if any(character in run_id for character in "/\\"):
        raise ValueError("run-id must not contain a path separator")
    output_root = output_root.expanduser()
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse training output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    state = {
        "schema": "xtbflow-real-joint-training-run/v1",
        "run_id": run_id,
        "status": "prepared",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "stages": {
            "input_audit": {"status": "not_started"},
            "training": {"status": "not_started"},
            "report": {"status": "not_started"},
        },
    }
    state_path = output_root / "run_state.json"
    _atomic_write_json(state_path, state)
    return run_id, state_path, state


def _update_stage(state_path: Path, state: dict[str, Any], stage: str, status: str, **details: Any) -> None:
    if stage not in state["stages"]:
        raise ValueError(f"unknown run stage: {stage}")
    state["stages"][stage] = {"status": status, **details}
    if status == "running":
        state["status"] = "running"
    state["updated_at"] = _utc_now()
    _atomic_write_json(state_path, state)


def _fail_run(state_path: Path, state: dict[str, Any], stage: str, exc: BaseException) -> None:
    message = str(exc)
    lower = message.lower()
    if stage == "training" and ("budget" in lower or "resource limit" in lower):
        stage_status = "budget_exhausted"
    elif stage == "report":
        stage_status = "report_write_failed"
    else:
        stage_status = f"{stage}_failed"
    _update_stage(
        state_path,
        state,
        stage,
        stage_status,
        failed_at=_utc_now(),
        error_type=type(exc).__name__,
        error=message,
    )
    state["status"] = "failed"
    state["failure"] = {
        "stage": stage,
        "status": stage_status,
        "error_type": type(exc).__name__,
        "error": message,
    }
    state["updated_at"] = _utc_now()
    _atomic_write_json(state_path, state)


def _git_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "dirty_entries": status}


def _runtime_identity() -> dict[str, Any]:
    result: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
    }
    if torch.cuda.is_available():
        result["cuda_device"] = torch.cuda.get_device_name(0)
        result["cuda_capability"] = list(torch.cuda.get_device_capability(0))
    try:
        result["rdkit"] = __import__("rdkit").__version__
    except Exception:
        result["rdkit"] = "unavailable"
    return result


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _center(coordinates: np.ndarray) -> np.ndarray:
    return coordinates - coordinates.mean(axis=0, keepdims=True)


def _kabsch_align(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rigidly align target onto reference without changing atom identity."""

    ref = _center(reference)
    moving = _center(target)
    covariance = moving.T @ ref
    left, _, right_transpose = np.linalg.svd(covariance)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1.0
        rotation = left @ right_transpose
    return moving @ rotation


def _matrix_state(bonds: np.ndarray, atomic_numbers: tuple[int, ...], max_atoms: int) -> np.ndarray:
    matrix = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    n_atoms = len(atomic_numbers)
    matrix[:n_atoms, :n_atoms] = bonds.astype(np.float32)
    degree = bonds.sum(axis=1)
    for index, number in enumerate(atomic_numbers):
        matrix[index, index] = VALENCE[number] - float(degree[index])
    return matrix


def _node_features(sample: DftDaSample, max_atoms: int) -> np.ndarray:
    features = np.zeros((max_atoms, 8), dtype=np.float32)
    degree = sample.reactant_bonds.sum(axis=1)
    for index, number in enumerate(sample.atomic_numbers):
        features[index, ELEMENT_INDEX[number]] = 1.0
        features[index, 4] = VALENCE[number] / 6.0
        features[index, 5] = number / 8.0
        features[index, 6] = 1.0
        features[index, 7] = float(degree[index]) / 4.0
    return features


def _tensor_sample(sample: DftDaSample, max_atoms: int, *, dtype: torch.dtype) -> TensorSample:
    n_atoms = len(sample.atomic_numbers)
    reactant = _center(sample.reactant_coordinates).astype(np.float32)
    transition = _kabsch_align(sample.reactant_coordinates, sample.ts_coordinates).astype(np.float32)
    reactant_state = _matrix_state(sample.reactant_bonds, sample.atomic_numbers, max_atoms)
    product_state = _matrix_state(sample.product_bonds, sample.atomic_numbers, max_atoms)
    target_event_matrix = product_state - reactant_state
    event_state = pack_be(torch.from_numpy(reactant_state).unsqueeze(0)).to(dtype=dtype)
    target_event = pack_be(torch.from_numpy(target_event_matrix).unsqueeze(0)).to(dtype=dtype)
    coordinates = torch.zeros((1, max_atoms, 3), dtype=dtype)
    target_geometry = torch.zeros_like(coordinates)
    coordinates[0, :n_atoms] = torch.from_numpy(reactant).to(dtype=dtype)
    target_geometry[0, :n_atoms] = torch.from_numpy(transition - reactant).to(dtype=dtype)
    atom_mask = torch.zeros((1, max_atoms), dtype=torch.bool)
    atom_mask[0, :n_atoms] = True
    rows, cols = upper_triangle_indices(max_atoms)
    event_mask = ((rows < n_atoms) & (cols < n_atoms)).unsqueeze(0)
    condition = torch.tensor(
        [[0.0, 0.0, n_atoms / max_atoms, sum(number != 1 for number in sample.atomic_numbers) / max_atoms]],
        dtype=dtype,
    )
    return TensorSample(
        sample=sample,
        split=sample.record.admission.removeprefix("development_"),
        event_state=event_state,
        target_event=target_event,
        coordinates=coordinates,
        target_geometry=target_geometry,
        node_features=torch.from_numpy(_node_features(sample, max_atoms)).unsqueeze(0).to(dtype=dtype),
        atom_mask=atom_mask,
        event_mask=event_mask,
        condition_features=condition,
    )


def _batch(samples: list[TensorSample], device: torch.device) -> dict[str, torch.Tensor]:
    keys = ("event_state", "target_event", "coordinates", "target_geometry", "node_features", "atom_mask", "event_mask", "condition_features")
    result: dict[str, torch.Tensor] = {}
    for key in keys:
        result[key] = torch.cat([getattr(sample, key) for sample in samples], dim=0).to(device=device)
    return result


def _loss(model: torch.nn.Module, mode: str, batch: dict[str, torch.Tensor], *, conservation_projection: bool) -> dict[str, torch.Tensor]:
    batch_size = batch["event_state"].shape[0]
    # Fixed, reproducible interpolation times still expose multiple flow times
    # to every arm and avoid using any held-out label to choose a schedule.
    tau = torch.linspace(0.15, 0.85, batch_size, dtype=batch["event_state"].dtype, device=batch["event_state"].device)[:, None]
    state = batch["event_state"] + tau * batch["target_event"]
    coordinates = batch["coordinates"] + tau[:, None, :] * batch["target_geometry"]
    output = model.forward_control(
        mode,
        state,
        coordinates,
        batch["node_features"],
        batch["atom_mask"],
        tau=tau,
        dt=0.25,
        condition_features=batch["condition_features"],
        conservation_projection=conservation_projection,
    )
    return joint_flow_loss(
        output,
        batch["target_event"],
        batch["target_geometry"],
        event_mask=batch["event_mask"],
        atom_mask=batch["atom_mask"],
        require_nonzero_target=True,
    )


def _evaluate(model: torch.nn.Module, mode: str, samples: list[TensorSample], *, device: torch.device, conservation_projection: bool, projector: Any) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    event_values: list[float] = []
    geometry_values: list[float] = []
    endpoint_values: list[float] = []
    residual_values: list[float] = []
    per_parent: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample in samples:
            batch = _batch([sample], device)
            tau = torch.full((1, 1), 0.5, dtype=batch["event_state"].dtype, device=device)
            state = batch["event_state"] + tau * batch["target_event"]
            coordinates = batch["coordinates"] + tau[:, None, :] * batch["target_geometry"]
            output = model.forward_control(
                mode,
                state,
                coordinates,
                batch["node_features"],
                batch["atom_mask"],
                tau=tau,
                dt=0.25,
                condition_features=batch["condition_features"],
                conservation_projection=conservation_projection,
            )
            losses = joint_flow_loss(
                output,
                batch["target_event"],
                batch["target_geometry"],
                event_mask=batch["event_mask"],
                atom_mask=batch["atom_mask"],
            )
            endpoint = coordinates + output.geometry_velocity
            observed = batch["atom_mask"][..., None].expand_as(endpoint)
            endpoint_error = (endpoint[observed] - (coordinates + batch["target_geometry"])[observed]).square().mean()
            residual = projector.residual(output.event_velocity).abs().mean()
            event_value = float(losses["event"].cpu())
            geometry_value = float(losses["geometry"].cpu())
            endpoint_value = float(endpoint_error.cpu())
            residual_value = float(residual.cpu())
            event_values.append(event_value)
            geometry_values.append(geometry_value)
            endpoint_values.append(endpoint_value)
            residual_values.append(residual_value)
            per_parent.append(
                {
                    "record_id": sample.sample.record.record_id,
                    "parent_reaction_id": sample.sample.record.parent_reaction_id,
                    "family_id": sample.sample.record.family_id,
                    "split": sample.split,
                    "event_velocity_mse": event_value,
                    "geometry_velocity_mse": geometry_value,
                    "geometry_endpoint_mse": endpoint_value,
                    "event_conservation_residual_abs": residual_value,
                }
            )
    return (
        {
            "event_velocity_mse": float(np.mean(event_values)),
            "geometry_velocity_mse": float(np.mean(geometry_values)),
            "geometry_endpoint_mse": float(np.mean(endpoint_values)),
            "event_conservation_residual_abs": float(np.mean(residual_values)),
        },
        per_parent,
    )


def _strong_rule_baseline(samples: list[DftDaSample]) -> dict[str, Any]:
    """Evaluate the registered reactant-only rule against labels as diagnostics."""

    attempted = 0
    applicable = 0
    covered = 0
    candidate_counts: list[int] = []
    failures: Counter[str] = Counter()
    per_parent: list[dict[str, Any]] = []
    for sample in samples:
        attempted += 1
        symbols = tuple({1: "H", 6: "C", 7: "N", 8: "O"}[number] for number in sample.atomic_numbers)
        be = np.rint(sample.reactant_bonds).astype(np.int64)
        degree = be.sum(axis=1).astype(np.int64)
        matrix = be.copy()
        for index, number in enumerate(sample.atomic_numbers):
            matrix[index, index] = int(VALENCE[number] - degree[index])
        try:
            state = LewisState(symbols=symbols, be=tuple(tuple(int(value) for value in row) for row in matrix), charge=0, multiplicity=1)
            proposals = enumerate_core_events(state, infer_roles(state))
        except (ValueError, KeyError) as exc:
            failures[type(exc).__name__ + ":" + str(exc)] += 1
            continue
        applicable += 1
        target_edits = {
            (min(left, right), max(left, right), int(round(delta)))
            for left, right, delta in _event_edit_indices(sample)
        }
        predicted = {
            (min(edit.i, edit.j), max(edit.i, edit.j), int(edit.delta))
            for proposal in proposals.candidates
            for edit in proposal.event.bond_edits
        }
        candidate_counts.append(len(proposals.candidates))
        hit = bool(target_edits) and any(
            {(min(edit.i, edit.j), max(edit.i, edit.j), int(edit.delta)) for edit in proposal.event.bond_edits} == target_edits
            for proposal in proposals.candidates
        )
        covered += int(hit)
        per_parent.append(
            {
                "record_id": sample.record.record_id,
                "parent_reaction_id": sample.record.parent_reaction_id,
                "split_group": sample.record.split_group,
                "candidate_count": len(proposals.candidates),
                "target_event_in_candidates": int(hit),
                "candidate_event_signature_count": len(predicted),
            }
        )
    by_parent = (
        summarize_by_group(
            per_parent,
            group_key="parent_reaction_id",
            metric_keys=("candidate_count", "target_event_in_candidates", "candidate_event_signature_count"),
        )
        if per_parent
        else {"group_key": "parent_reaction_id", "record_count": 0, "group_count": 0, "groups": []}
    )
    return {
        "arm_id": "strong_rule",
        "status": "completed",
        "implementation": "xtbflow.proposals.enumerate_core_events",
        "input_origin": "reactant_only",
        "calculator_calls": 0,
        "records_attempted": attempted,
        "records_applicable": applicable,
        "records_with_target_event_in_candidates": covered,
        "target_event_coverage_diagnostic": (covered / applicable) if applicable else None,
        "mean_candidate_count": float(np.mean(candidate_counts)) if candidate_counts else None,
        "failure_counts": dict(failures),
        "per_record": per_parent,
        "by_parent": by_parent,
        "claim_limit": "reactant-only rule diagnostic; label-assisted coverage is not a discovery claim",
    }


def _event_edit_indices(sample: DftDaSample) -> list[tuple[int, int, int]]:
    indices: list[tuple[int, int, int]] = []
    for left in range(len(sample.atomic_numbers)):
        for right in range(left + 1, len(sample.atomic_numbers)):
            delta = int(round(sample.product_bonds[left, right] - sample.reactant_bonds[left, right]))
            if delta:
                indices.append((left, right, delta))
    return indices


def _train_arm(
    *,
    seed: int,
    arm_id: str,
    mode: str,
    conservation_projection: bool,
    base_state: dict[str, torch.Tensor],
    config: JointFlowRuntimeConfig,
    projector: Any,
    train_samples: list[TensorSample],
    validation_samples: list[TensorSample],
    test_samples: list[TensorSample],
    output_root: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_wall_seconds: float,
) -> dict[str, Any]:
    model = config.build(projector).to(device=device, dtype=torch.float32)
    model.load_state_dict(base_state, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    start = time.monotonic()
    losses_history: list[dict[str, float]] = []
    global_step = 0
    stop_reason = "completed"
    for epoch in range(epochs):
        model.train()
        order = list(range(len(train_samples)))
        random.Random(seed * 1009 + epoch).shuffle(order)
        epoch_values: list[float] = []
        for offset in range(0, len(order), batch_size):
            if time.monotonic() - start >= max_wall_seconds:
                stop_reason = "budget_exhausted"
                break
            chosen = [train_samples[index] for index in order[offset : offset + batch_size]]
            batch = _batch(chosen, device)
            optimizer.zero_grad(set_to_none=True)
            losses = _loss(model, mode, batch, conservation_projection=conservation_projection)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            value = float(losses["total"].detach().cpu())
            epoch_values.append(value)
            global_step += 1
        losses_history.append({"epoch": epoch + 1, "train_total_mse": float(np.mean(epoch_values)) if epoch_values else float("nan")})
        if stop_reason != "completed":
            break
    checkpoint_path = output_root / "checkpoints" / f"seed_{seed}" / f"{arm_id}.pt"
    metadata = {
        "issue": 58,
        "arm_id": arm_id,
        "control_mode": mode,
        "conservation_projection": conservation_projection,
        "seed": seed,
        "claim_limit": "development evidence only",
    }
    save_joint_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        loop={"global_step": global_step, "data_order": list(range(len(train_samples)))},
        sampler_state={"seed": seed, "epochs_completed": len(losses_history)},
        metadata=metadata,
    )
    validation_metrics, validation_rows = _evaluate(model, mode, validation_samples, device=device, conservation_projection=conservation_projection, projector=projector)
    test_metrics, test_rows = _evaluate(model, mode, test_samples, device=device, conservation_projection=conservation_projection, projector=projector)
    metric_keys = tuple(validation_metrics)
    validation_by_parent = summarize_by_group(
        validation_rows,
        group_key="parent_reaction_id",
        metric_keys=metric_keys,
    )
    test_by_parent = summarize_by_group(
        test_rows,
        group_key="parent_reaction_id",
        metric_keys=metric_keys,
    )
    return {
        **metadata,
        "status": stop_reason,
        "elapsed_seconds": time.monotonic() - start,
        "global_step": global_step,
        "loss_history": losses_history,
        "checkpoint_locator": f"checkpoints/seed_{seed}/{arm_id}.pt",
        "checkpoint_sha256": _sha256(checkpoint_path),
        "model_identity": model_state_identity(model),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        # Keep record rows for audit, but label them honestly.  Parent-level
        # rows below are means over repeated records from that parent.
        "validation_per_record": validation_rows,
        "test_per_record": test_rows,
        "validation_by_parent": validation_by_parent,
        "test_by_parent": test_by_parent,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", type=str, default=None, help="optional stable label; output paths are always single-use")
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/models/joint_flow_runtime_v0.1.json"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-wall-minutes", type=float, default=110.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 17, 23])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def _paired_test_comparisons(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare every learned arm with the serial arm within each seed."""

    serial_by_seed = {
        int(result["seed"]): result
        for result in results
        if result.get("arm_id") == "serial_event_to_geometry"
    }
    metric_keys = (
        "event_velocity_mse",
        "geometry_velocity_mse",
        "geometry_endpoint_mse",
        "event_conservation_residual_abs",
    )
    comparisons: list[dict[str, Any]] = []
    for result in results:
        arm_id = result.get("arm_id")
        if arm_id in {"strong_rule", "serial_event_to_geometry"}:
            continue
        reference = serial_by_seed.get(int(result["seed"]))
        if reference is None:
            continue
        comparison = paired_group_comparison(
            result["test_per_record"],
            reference["test_per_record"],
            metric_keys=metric_keys,
            left_label=f"{arm_id}/seed_{result['seed']}",
            right_label=f"serial_event_to_geometry/seed_{result['seed']}",
            group_key="parent_reaction_id",
            bootstrap_samples=2000,
            seed=1000 + int(result["seed"]),
        )
        comparisons.append(comparison)
    return comparisons


def _run(args: argparse.Namespace, *, run_id: str, output_root: Path, state_path: Path, state: dict[str, Any]) -> int:
    stage = "input_audit"
    _update_stage(state_path, state, stage, "running", started_at=_utc_now())
    git_identity = _git_identity()
    state["git"] = git_identity
    state["updated_at"] = _utc_now()
    _atomic_write_json(state_path, state)
    config = JointFlowRuntimeConfig.load(args.runtime_config)
    samples, source_audit = load_dft_da_samples(args.cache_root)
    samples = with_splits(samples)
    records = [sample.record for sample in samples]
    leakage = audit_track_b_leakage(records, strict_family_holdout=True)
    manifest_path = output_root / "track_b_manifest.jsonl"
    manifest_temporary = output_root / ".track_b_manifest.jsonl.tmp"
    write_manifest(manifest_temporary, samples)
    os.replace(manifest_temporary, manifest_path)
    # The input firewall is checked before tensor construction and is reported
    # in evidence so a later consumer can distinguish labels from model inputs.
    for record in records:
        if FORBIDDEN_INPUT_KEYS.intersection(record.reactant_view()):
            raise ValueError(f"forbidden target key in reactant view: {record.record_id}")
    split_counts = Counter(sample.record.admission for sample in samples)
    max_atoms = max(len(sample.atomic_numbers) for sample in samples)
    dtype = torch.float32
    tensor_samples = [_tensor_sample(sample, max_atoms, dtype=dtype) for sample in samples]
    train_samples = [sample for sample in tensor_samples if sample.split == "train"]
    validation_samples = [sample for sample in tensor_samples if sample.split == "validation"]
    test_samples = [sample for sample in tensor_samples if sample.split == "test"]
    if not train_samples or not validation_samples or not test_samples:
        raise ValueError(f"group split is empty: {split_counts}")
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    torch.set_float32_matmul_precision("high")
    projector = total_electron_projector(["C"] * max_atoms, dtype=dtype, device=device)
    split_fingerprint = canonical_hash({row.record.record_id: row.record.admission for row in samples})
    input_fingerprint = canonical_hash({row.record.record_id: row.record.input_fingerprint() for row in samples})
    run_config = {
        "schema": "xtbflow-real-joint-training/v2",
        "run_id": run_id,
        "git": git_identity,
        "issue": 58,
        "runtime_config": config.constructor_record(),
        "arms": [{"arm_id": "strong_rule", "kind": "reactant_only_rule"}] + [
            {"arm_id": arm, "control_mode": mode, "conservation_projection": projection}
            for arm, mode, projection in ARM_SPECS
        ],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seeds": list(args.seeds),
        "candidate_cap": 64,
        "max_wall_minutes": args.max_wall_minutes,
        "device": str(device),
        "max_atoms": max_atoms,
        "input_firewall": sorted(FORBIDDEN_INPUT_KEYS),
        "source_archive_sha256": source_audit.get("source_archive_sha256"),
        "split_fingerprint": split_fingerprint,
        "reactant_input_view_fingerprint": input_fingerprint,
    }
    config_path = output_root / "run_config.json"
    _atomic_write_json(config_path, run_config)
    _update_stage(
        state_path,
        state,
        stage,
        "completed",
        completed_at=_utc_now(),
        record_count=len(samples),
        parent_count=len({sample.record.parent_reaction_id for sample in samples}),
        config_sha256=_sha256(config_path),
        manifest_sha256=_sha256(manifest_path),
    )

    stage = "training"
    _update_stage(state_path, state, stage, "running", started_at=_utc_now())
    state["training_progress"] = {"completed_arms": [], "updates_observed": 0}
    state["updated_at"] = _utc_now()
    _atomic_write_json(state_path, state)
    results: list[dict[str, Any]] = [_strong_rule_baseline([sample.sample for sample in tensor_samples])]
    run_start = time.monotonic()
    for seed in args.seeds:
        _seed_everything(seed)
        base_model = config.build(projector).to(device=device, dtype=dtype)
        base_state = {name: value.detach().cpu().clone() for name, value in base_model.state_dict().items()}
        for arm_id, mode, conservation_projection in ARM_SPECS:
            remaining = args.max_wall_minutes * 60.0 - (time.monotonic() - run_start)
            if remaining <= 0:
                raise RuntimeError("run-level wall budget exhausted before all arms")
            result = _train_arm(
                seed=seed,
                arm_id=arm_id,
                mode=mode,
                conservation_projection=conservation_projection,
                base_state=base_state,
                config=config,
                projector=projector,
                train_samples=train_samples,
                validation_samples=validation_samples,
                test_samples=test_samples,
                output_root=output_root,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                max_wall_seconds=remaining,
            )
            results.append(result)
            state["training_progress"]["completed_arms"].append(
                {"arm_id": result["arm_id"], "seed": seed, "status": result["status"]}
            )
            state["training_progress"]["updates_observed"] += int(result.get("global_step", 0))
            state["updated_at"] = _utc_now()
            _atomic_write_json(state_path, state)
            if result["status"] != "completed":
                raise RuntimeError(f"arm stopped at resource limit: {result['arm_id']} seed={seed}")
    _update_stage(
        state_path,
        state,
        stage,
        "completed",
        completed_at=_utc_now(),
        arm_count=len(results),
        training_updates=sum(int(result.get("global_step", 0)) for result in results if result.get("arm_id") != "strong_rule"),
    )

    stage = "report"
    _update_stage(state_path, state, stage, "running", started_at=_utc_now())
    paired_comparisons = _paired_test_comparisons(results)
    evidence = {
        "schema": "xtbflow-real-joint-training-evidence/v2",
        "status": "completed",
        "run_id": run_id,
        "run_state": "run_state.json",
        "issue": 58,
        "claim_limit": "development evidence only; no physical refinement or confirmatory claim",
        "git": git_identity,
        "runtime": _runtime_identity(),
        "source_audit": source_audit,
        "manifest_sha256": _sha256(manifest_path),
        "config_sha256": _sha256(config_path),
        "split_fingerprint": split_fingerprint,
        "reactant_input_view_fingerprint": input_fingerprint,
        "split_counts": dict(sorted(split_counts.items())),
        "parent_counts_by_split": {
            split: len({sample.sample.record.parent_reaction_id for sample in tensor_samples if sample.split == split})
            for split in ("train", "validation", "test")
        },
        "leakage_audit": leakage,
        "record_count": len(samples),
        "parent_count": len({sample.record.parent_reaction_id for sample in samples}),
        "max_atoms": max_atoms,
        "resource_budget": {
            "max_concurrent_jobs": 1,
            "max_wall_minutes": args.max_wall_minutes,
            "max_retries": 0,
            "training_calls": len(args.seeds) * len(ARM_SPECS) * args.epochs * ((len(train_samples) + args.batch_size - 1) // args.batch_size),
            "training_updates_observed": sum(int(result.get("global_step", 0)) for result in results if result.get("arm_id") != "strong_rule"),
        },
        "elapsed_seconds": time.monotonic() - run_start,
        "evaluation": {
            "group_key": "parent_reaction_id",
            "weighting": "record_weighted_and_parent_equal",
            "paired_test_comparisons": paired_comparisons,
        },
        "arms": results,
    }
    evidence_path = output_root / "evidence.json"
    _atomic_write_json(evidence_path, evidence)
    _update_stage(state_path, state, stage, "completed", completed_at=_utc_now(), evidence_sha256=_sha256(evidence_path))
    state["status"] = "completed"
    state["evidence"] = "evidence.json"
    state["updated_at"] = _utc_now()
    _atomic_write_json(state_path, state)
    print(json.dumps({"status": evidence["status"], "records": len(samples), "parents": evidence["parent_count"], "split_counts": evidence["split_counts"], "evidence": str(evidence_path)}, sort_keys=True))
    return 0


def main() -> int:
    args = _parser().parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.max_wall_minutes <= 0:
        raise SystemExit("epochs, batch-size, learning-rate and max-wall-minutes must be positive")
    output_root = args.output.expanduser()
    run_id, state_path, state = _start_run(output_root, args.run_id)
    stage = "input_audit"
    try:
        return _run(args, run_id=run_id, output_root=output_root, state_path=state_path, state=state)
    except BaseException as exc:
        # Preserve the consumed-cost boundary and the exact stage where work
        # stopped.  The original exception is re-raised for scheduler logs.
        _fail_run(state_path, state, stage if state["stages"]["training"]["status"] == "not_started" else ("training" if state["stages"]["report"]["status"] == "not_started" else "report"), exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
