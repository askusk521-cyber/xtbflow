#!/usr/bin/env python3
"""Run a bounded real-data joint-flow development comparison.

This entry point deliberately consumes the audited Transition1x endpoint
asset in quarantine.  It derives a binary event diagnostic from the reactant
and product coordinates, trains the current joint model on reactant-to-TS
displacements, and evaluates the same parameter state through the registered
``both_off``, ``serial_independent``, and ``joint_bidirectional`` controls.
The output is development evidence only: it does not open the Track-B gate or
support a product-free discovery claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
from typing import Any

import numpy as np
import torch

from xtbflow.data.transition1x_audit import (
    audit_transition1x_pickle,
    hash_file,
    load_transition1x_pickle,
)
from xtbflow.data.transition1x_development import development_sample
from xtbflow.models import JointFlowRuntimeConfig, pack_be, total_electron_projector, upper_triangle_indices
from xtbflow.training import coupled_control_manifest, joint_flow_loss, model_state_identity


CONTROLS = ("both_off", "serial_independent", "joint_bidirectional")
_ELEMENT_INDEX = {1: 0, 6: 1, 7: 2, 8: 3}
_VALENCE = {1: 1.0, 6: 4.0, 7: 5.0, 8: 6.0}


def _git_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(dirty), "dirty_entries": dirty}


def _source(config_path: Path, source_id: str) -> dict[str, Any]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    matches = [row for row in payload.get("sources", []) if row.get("source_id") == source_id]
    if len(matches) != 1:
        raise ValueError(f"expected one source config row for {source_id}")
    return dict(matches[0])


def _center(coordinates: np.ndarray) -> np.ndarray:
    return coordinates - coordinates.mean(axis=0, keepdims=True)


def _features(atomic_numbers: np.ndarray, bonds: np.ndarray, max_atoms: int) -> np.ndarray:
    result = np.zeros((max_atoms, 8), dtype=np.float64)
    for index, number in enumerate(atomic_numbers.tolist()):
        number = int(number)
        result[index, _ELEMENT_INDEX[number]] = 1.0
        result[index, 4] = _VALENCE[number] / 6.0
        result[index, 5] = number / 8.0
        result[index, 6] = 1.0
        result[index, 7] = float(bonds[index].sum()) / 4.0
    return result


def _matrix_state(sample: dict[str, Any], max_atoms: int) -> np.ndarray:
    numbers = sample["atomic_numbers"]
    bonds = sample["event"]["reactant_bond_matrix"]
    matrix = np.zeros((max_atoms, max_atoms), dtype=np.float64)
    degree = bonds.sum(axis=1)
    for index, number in enumerate(numbers.tolist()):
        matrix[index, index] = _VALENCE[int(number)] - float(degree[index])
    matrix[: len(numbers), : len(numbers)] += bonds
    return matrix


def _tensor_sample(sample: dict[str, Any], max_atoms: int, projector) -> dict[str, torch.Tensor]:
    numbers = sample["atomic_numbers"]
    n_atoms = len(numbers)
    reactant = _center(np.asarray(sample["reactant_coordinates"], dtype=np.float64))
    transition = _center(np.asarray(sample["transition_state_coordinates"], dtype=np.float64))
    matrix = _matrix_state(sample, max_atoms)
    delta = np.zeros_like(matrix)
    delta[:n_atoms, :n_atoms] = np.asarray(sample["event_delta"], dtype=np.float64)
    event_state = pack_be(torch.from_numpy(matrix).unsqueeze(0))
    target_event = pack_be(torch.from_numpy(delta).unsqueeze(0))
    coordinates = torch.zeros((1, max_atoms, 3), dtype=torch.float64)
    target_geometry = torch.zeros_like(coordinates)
    coordinates[0, :n_atoms] = torch.from_numpy(reactant)
    target_geometry[0, :n_atoms] = torch.from_numpy(transition - reactant)
    atom_mask = torch.zeros((1, max_atoms), dtype=torch.bool)
    atom_mask[0, :n_atoms] = True
    rows, cols = upper_triangle_indices(max_atoms)
    event_mask = ((rows < n_atoms) & (cols < n_atoms)).unsqueeze(0)
    node_features = torch.from_numpy(_features(numbers, sample["event"]["reactant_bond_matrix"], max_atoms)).unsqueeze(0)
    conditions = torch.zeros((1, 4), dtype=torch.float64)
    return {
        "event_state": event_state,
        "target_event": target_event,
        "coordinates": coordinates,
        "target_geometry": target_geometry,
        "node_features": node_features,
        "atom_mask": atom_mask,
        "event_mask": event_mask,
        "condition_features": conditions,
        "projector": projector,
    }


def _loss_for_mode(model, mode: str, sample: dict[str, torch.Tensor], tau: float) -> dict[str, torch.Tensor]:
    event_state = sample["event_state"] + tau * sample["target_event"]
    coordinates = sample["coordinates"] + tau * sample["target_geometry"]
    output = model.forward_control(
        mode,
        event_state,
        coordinates,
        sample["node_features"],
        sample["atom_mask"],
        tau=tau,
        dt=1.0,
        condition_features=sample["condition_features"],
    )
    return joint_flow_loss(
        output,
        sample["target_event"],
        sample["target_geometry"],
        event_mask=sample["event_mask"],
        atom_mask=sample["atom_mask"],
    )


def _evaluate(model, samples: list[dict[str, torch.Tensor]]) -> dict[str, dict[str, float]]:
    """Evaluate label-assisted midpoint diagnostics.

    The state passed to this function contains the target event and target
    geometry at ``tau=0.5``. These metrics diagnose the learned vector field
    under teacher forcing; they are not endpoint generation from reactants.
    """
    result: dict[str, dict[str, float]] = {}
    model.eval()
    with torch.no_grad():
        for mode in CONTROLS:
            event_losses: list[float] = []
            geometry_losses: list[float] = []
            endpoint_losses: list[float] = []
            for sample in samples:
                losses = _loss_for_mode(model, mode, sample, 0.5)
                event_losses.append(float(losses["event"]))
                geometry_losses.append(float(losses["geometry"]))
                output = model.forward_control(
                    mode,
                    sample["event_state"] + 0.5 * sample["target_event"],
                    sample["coordinates"] + 0.5 * sample["target_geometry"],
                    sample["node_features"],
                    sample["atom_mask"],
                    tau=0.5,
                    dt=1.0,
                    condition_features=sample["condition_features"],
                )
                endpoint = sample["coordinates"] + output.geometry_velocity
                observed = sample["atom_mask"][..., None].expand_as(endpoint)
                endpoint_losses.append(float((endpoint[observed] - (sample["coordinates"] + sample["target_geometry"])[observed]).square().mean()))
            result[mode] = {
                "event_velocity_mse": float(np.mean(event_losses)),
                "geometry_velocity_mse": float(np.mean(geometry_losses)),
                "label_assisted_geometry_endpoint_mse": float(np.mean(endpoint_losses)),
            }
    return result


def collect_development_samples(
    payload: dict[str, Any], source_count: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert all source rows while retaining every rejected row."""

    candidates: list[dict[str, Any]] = []
    filter_failures: list[dict[str, Any]] = []
    for index in range(source_count):
        try:
            candidates.append(development_sample(payload, index))
        except (KeyError, TypeError, ValueError) as exc:
            filter_failures.append(
                {
                    "index": index,
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                }
            )
    return candidates, filter_failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/data/track_b_source_candidates_v1.json"))
    parser.add_argument("--source-id", default="transition1x_preprocessed")
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/models/joint_flow_runtime_v0.1.json"))
    parser.add_argument("--max-records", type=int, default=32)
    parser.add_argument(
        "--epochs", "--steps", dest="epochs", type=int, default=8,
        help="training epochs; --steps is retained as a compatibility alias",
    )
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    if args.max_records < 8 or args.epochs < 1:
        raise SystemExit("max-records must be at least 8 and epochs must be positive")
    started = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    source = _source(args.config, args.source_id)
    audit = audit_transition1x_pickle(
        args.asset,
        expected_sha256=source["expected_asset_sha256"],
        expected_size_bytes=source.get("expected_asset_size_bytes"),
        source_revision=source["source_revision"],
        source_locator=source["source_locator"],
        license_record=source["license_record"],
        upstream_code_commit=source["upstream_code_commit"],
        expected_record_count=source.get("declared_record_count"),
        expected_md5=source.get("zenodo_md5"),
        license_status=source.get("license_status", "unknown"),
    )
    payload = load_transition1x_pickle(
        args.asset,
        expected_sha256=source["expected_asset_sha256"],
        expected_size_bytes=source.get("expected_asset_size_bytes"),
    )
    source_count = int(audit["structure"]["record_count"])
    candidates, filter_failures = collect_development_samples(payload, source_count)
    by_split = {
        split: sorted(
            (row for row in candidates if row["split"] == split),
            key=lambda row: (row["formula"], row["reaction_id"]),
        )
        for split in ("train", "validation", "test")
    }
    if any(not by_split[split] for split in by_split):
        raise RuntimeError("deterministic development split did not populate all roles")
    train_target = max(1, int(round(args.max_records * 0.75)))
    validation_target = max(1, int(round(args.max_records * 0.125)))
    test_target = max(1, args.max_records - train_target - validation_target)
    selected = (
        by_split["train"][:train_target]
        + by_split["validation"][:validation_target]
        + by_split["test"][:test_target]
    )
    if len(selected) < 4:
        raise RuntimeError("fewer than four valid development samples were found")
    max_atoms = max(len(row["atomic_numbers"]) for row in selected)
    split_counts = Counter(row["split"] for row in selected)
    # Keep at least one held-out row while preserving deterministic ordering.
    train_rows = [row for row in selected if row["split"] == "train"]
    heldout_rows = [row for row in selected if row["split"] in {"validation", "test"}]
    if not train_rows or not heldout_rows:
        raise RuntimeError("deterministic development split did not contain train and held-out rows")
    projector = total_electron_projector(["H"] * max_atoms, dtype=torch.float64)
    train_samples = [_tensor_sample(row, max_atoms, projector) for row in train_rows]
    heldout_samples_by_split = {
        split: [
            _tensor_sample(row, max_atoms, projector)
            for row in selected
            if row["split"] == split
        ]
        for split in ("validation", "test")
    }
    runtime = JointFlowRuntimeConfig.load(args.runtime_config)
    model = runtime.build(projector).to(dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses: list[float] = []
    model.train()
    for step in range(args.epochs):
        order = list(range(len(train_samples)))
        random.Random(args.seed + step).shuffle(order)
        for offset in order:
            sample = train_samples[offset]
            tau = 0.2 + 0.6 * ((step + offset) % 5) / 4.0
            optimizer.zero_grad(set_to_none=True)
            loss = _loss_for_mode(model, "joint_bidirectional", sample, tau)
            loss["total"].backward()
            optimizer.step()
            losses.append(float(loss["total"].detach()))
    metrics = {
        split: _evaluate(model, samples)
        for split, samples in heldout_samples_by_split.items()
    }
    control_manifest = coupled_control_manifest(
        modes=CONTROLS,
        models={mode: model for mode in CONTROLS},
        physical_call_budgets={mode: 0 for mode in CONTROLS},
        coupling_strength=1.0,
        guidance_strength=0.0,
        dataset_fingerprint=hash_file(args.asset),
        split_fingerprint=hashlib.sha256(
            json.dumps([(row["reaction_id"], row["split"]) for row in selected], sort_keys=True).encode()
        ).hexdigest(),
        input_view_fingerprint="transition1x-reactant-ts-development-v1",
    )
    report = {
        "schema": "xtbflow-transition1x-joint-development/v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_host": __import__("platform").node(),
        "git": _git_identity(),
        "asset": {
            "path": str(args.asset),
            "sha256": hash_file(args.asset),
            "size_bytes": args.asset.stat().st_size,
            "source_id": args.source_id,
            "source_revision": source["source_revision"],
            "source_audit": audit,
        },
        "data_status": "quarantine_development_only",
        "claim_limit": "Endpoint-derived event labels and source metadata do not qualify product-free Track-B training or scientific benchmark claims.",
        "selection": {
            "source_record_count": source_count,
            "candidate_count": len(candidates),
            "filter_failure_count": len(filter_failures),
            "filter_failures": filter_failures,
            "selected_count": len(selected),
            "max_atoms": max_atoms,
            "split_counts": dict(sorted(split_counts.items())),
            "train_count": len(train_samples),
            "validation_count": len(heldout_samples_by_split["validation"]),
            "test_count": len(heldout_samples_by_split["test"]),
            "element_scope": "H/C/N/O",
            "event_label": "covalent-radius-binary-endpoint-edit-v1",
            "electronic_state": "unresolved; condition features are zero-filled and excluded from claims",
        },
        "runtime_config": runtime.constructor_record(),
        "runtime_config_sha256": hash_file(args.runtime_config),
        "seed": args.seed,
        "epochs": args.epochs,
        "optimizer_updates": len(losses),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "model_identity": model_state_identity(model),
        "controls": metrics,
        "evaluation_protocol": {
            "kind": "label_assisted_teacher_forced_midpoint",
            "uses_target_event": True,
            "uses_target_geometry": True,
            "supports_reactant_only_generation_claim": False,
        },
        "control_manifest": control_manifest,
        "conservation_residual_max_abs": max(float(abs(row["event"]["conservation_residual"])) for row in selected),
        "elapsed_seconds": time.monotonic() - started,
        "limits": [
            "The source remains outside the repository and is hash-gated before deserialization.",
            "The product endpoint is used to derive the diagnostic event label.",
            "Geometry endpoint metrics are label-assisted teacher-forced diagnostics, not reactant-only generation.",
            "The reported optimizer update count is epochs multiplied by the selected training rows.",
            "Every rejected source row is retained in selection.filter_failures.",
            "No calculator calls or independent reactant seed searches were made.",
            "Results are software/data-pipeline development evidence only.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"schema": report["schema"], "data_status": report["data_status"], "selected_count": len(selected), "controls": metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
