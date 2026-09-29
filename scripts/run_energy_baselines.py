#!/usr/bin/env python3
"""Train matched direct and Delta conservative E/F baselines on a frozen pair cache."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xtbflow.models import DeltaEnergyModel, DirectEnergyModel  # noqa: E402

HARTREE_TO_KCAL_MOL = 627.5094740631
ELEMENTS = (1, 6, 7, 8, 16)
DEFAULT_SEED = 20260929
TRAIN_PARENT_RANK_SEED = "spice2-openff-learning-curve-v1"
FORCE_DIRECTION_MIN_REFERENCE_NORM = 1.0e-6
EPOCHS = 80
BATCH_SIZE = 4
LR = 1.0e-3
HIDDEN_DIM = 32
RADIAL_FEATURES = 8
DISTANCE_SCALE = 8.0
FORCE_WEIGHT = 10.0
GRADIENT_CLIP = 10.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(pair_path: Path, manifest_path: Path) -> list[dict]:
    manifest_by_id = {
        row["source_record_id"]: row
        for row in map(json.loads, manifest_path.read_text(encoding="utf-8").splitlines())
        if row
    }
    rows = []
    for row in map(json.loads, pair_path.read_text(encoding="utf-8").splitlines()):
        if row["status"] != "success":
            continue
        source_id = row["source_record_id"]
        if source_id not in manifest_by_id:
            raise SystemExit(f"pair source_record_id is absent from manifest: {source_id}")
        manifest_row = manifest_by_id[source_id]
        row["split"] = manifest_row["split"]
        row["parent_record_id"] = manifest_row["parent_record_id"]
        row["config_index"] = int(manifest_row.get("config_index", 0))
        rows.append(row)
    return rows


def _element_counts(row: dict) -> np.ndarray:
    symbols = row["symbols"]
    z_by_symbol = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16}
    numbers = [z_by_symbol[symbol] for symbol in symbols]
    return np.asarray([numbers.count(z) for z in ELEMENTS], dtype=np.float64)


def _fit_offsets(rows: list[dict], target: str) -> np.ndarray:
    x = np.stack([_element_counts(row) for row in rows])
    if target == "direct":
        y = np.asarray([row["reference_energy"] for row in rows], dtype=np.float64)
    else:
        y = np.asarray([row["reference_energy"] - row["semi_empirical_energy"] for row in rows], dtype=np.float64)
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _offset(row: dict, coefficients: np.ndarray) -> float:
    return float(_element_counts(row) @ coefficients)


def _rank_training_parents(parent_ids: list[str], *, seed: str = TRAIN_PARENT_RANK_SEED) -> list[str]:
    """Return a deterministic label-blind parent ordering for nested train subsets."""

    unique = sorted(set(parent_ids))
    return sorted(unique, key=lambda parent: hashlib.sha256(f"{seed}:{parent}".encode()).digest())


def _select_training_parent_ids(parent_ids: list[str], count: int) -> list[str]:
    ranked = _rank_training_parents(parent_ids)
    if count < 1 or count > len(ranked):
        raise ValueError(f"training parent count must be between 1 and {len(ranked)}")
    return ranked[:count]


def _force_direction_angles(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1, 3)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
    reference_norm = np.linalg.norm(reference, axis=1)
    keep = reference_norm > FORCE_DIRECTION_MIN_REFERENCE_NORM
    if not np.any(keep):
        return np.asarray([], dtype=np.float64)
    predicted = predicted[keep]
    reference = reference[keep]
    reference_norm = reference_norm[keep]
    predicted_norm = np.linalg.norm(predicted, axis=1)
    cosine = np.empty(len(predicted), dtype=np.float64)
    nonzero = predicted_norm > 1.0e-12
    cosine[~nonzero] = -1.0
    cosine[nonzero] = np.sum(predicted[nonzero] * reference[nonzero], axis=1) / (
        predicted_norm[nonzero] * reference_norm[nonzero]
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _prediction_summary(predictions: list[dict], *, evaluation_wall_seconds: float | None = None) -> dict:
    if not predictions:
        return {"systems": 0, "independent_parent_count": 0}
    energy_error = np.asarray([
        item["predicted_energy"] - item["reference_energy"] for item in predictions
    ], dtype=np.float64)
    force_error = np.concatenate([
        (item["predicted_forces"] - item["reference_forces"]).reshape(-1)
        for item in predictions
    ])
    force_angle_values = [
        _force_direction_angles(item["predicted_forces"], item["reference_forces"])
        for item in predictions
    ]
    force_angles = np.concatenate([value for value in force_angle_values if value.size]) if any(
        value.size for value in force_angle_values
    ) else np.asarray([], dtype=np.float64)

    grouped: dict[str, list[dict]] = {}
    for item in predictions:
        grouped.setdefault(item["parent_record_id"], []).append(item)

    parent_metrics = []
    relative_errors = []
    for parent_id in sorted(grouped):
        group = sorted(grouped[parent_id], key=lambda item: item["config_index"])
        group_energy_error = np.asarray([
            item["predicted_energy"] - item["reference_energy"] for item in group
        ], dtype=np.float64)
        group_force_error = np.concatenate([
            (item["predicted_forces"] - item["reference_forces"]).reshape(-1)
            for item in group
        ])
        group_angles_list = [
            _force_direction_angles(item["predicted_forces"], item["reference_forces"])
            for item in group
        ]
        group_angles = np.concatenate([value for value in group_angles_list if value.size]) if any(
            value.size for value in group_angles_list
        ) else np.asarray([], dtype=np.float64)
        if len(group) > 1:
            anchor = group[0]
            group_relative_errors = np.asarray([
                (
                    (item["predicted_energy"] - anchor["predicted_energy"])
                    - (item["reference_energy"] - anchor["reference_energy"])
                )
                for item in group[1:]
            ], dtype=np.float64)
            relative_errors.extend(group_relative_errors.tolist())
            relative_mae = float(np.mean(np.abs(group_relative_errors)) * HARTREE_TO_KCAL_MOL)
        else:
            relative_mae = None
        parent_metrics.append({
            "parent_record_id": parent_id,
            "configs": len(group),
            "charge": int(group[0]["charge"]),
            "contains_sulfur": bool(group[0]["contains_sulfur"]),
            "energy_mae_kcal_per_mol": float(np.mean(np.abs(group_energy_error)) * HARTREE_TO_KCAL_MOL),
            "relative_energy_mae_kcal_per_mol": relative_mae,
            "force_component_mae_hartree_per_angstrom": float(np.mean(np.abs(group_force_error))),
            "force_vector_mean_angle_degrees": float(np.mean(group_angles)) if group_angles.size else None,
        })

    result = {
        "systems": len(predictions),
        "independent_parent_count": len(grouped),
        "energy_mae_kcal_per_mol": float(np.mean(np.abs(energy_error)) * HARTREE_TO_KCAL_MOL),
        "energy_rmse_kcal_per_mol": float(np.sqrt(np.mean(energy_error ** 2)) * HARTREE_TO_KCAL_MOL),
        "energy_max_abs_kcal_per_mol": float(np.max(np.abs(energy_error)) * HARTREE_TO_KCAL_MOL),
        "relative_energy_mae_kcal_per_mol": (
            float(np.mean(np.abs(np.asarray(relative_errors))) * HARTREE_TO_KCAL_MOL)
            if relative_errors else None
        ),
        "relative_energy_comparisons": len(relative_errors),
        "force_component_mae_hartree_per_angstrom": float(np.mean(np.abs(force_error))),
        "force_component_rmse_hartree_per_angstrom": float(np.sqrt(np.mean(force_error ** 2))),
        "force_component_p95_abs_hartree_per_angstrom": float(np.quantile(np.abs(force_error), 0.95)),
        "force_vector_mean_angle_degrees": float(np.mean(force_angles)) if force_angles.size else None,
        "force_vector_p95_angle_degrees": float(np.quantile(force_angles, 0.95)) if force_angles.size else None,
        "force_direction_reference_norm_min_hartree_per_angstrom": FORCE_DIRECTION_MIN_REFERENCE_NORM,
        "force_direction_vectors": int(force_angles.size),
        "parent_metrics": parent_metrics,
    }
    if evaluation_wall_seconds is not None:
        result["evaluation_wall_seconds"] = evaluation_wall_seconds
    return result


def _bare_predictions(rows: list[dict], coefficients: np.ndarray) -> list[dict]:
    output = []
    for row in rows:
        output.append({
            "parent_record_id": row["parent_record_id"],
            "config_index": row["config_index"],
            "charge": row["charge"],
            "contains_sulfur": "S" in row["symbols"],
            "predicted_energy": float(row["semi_empirical_energy"] + _offset(row, coefficients)),
            "reference_energy": float(row["reference_energy"]),
            "predicted_forces": np.asarray(row["semi_empirical_forces"], dtype=np.float64),
            "reference_forces": np.asarray(row["reference_forces"], dtype=np.float64),
        })
    return output


def _batch(rows: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    max_atoms = max(len(row["symbols"]) for row in rows)
    batch = len(rows)
    coordinates = torch.zeros(batch, max_atoms, 3, dtype=torch.float32, device=device)
    species = torch.zeros(batch, max_atoms, dtype=torch.long, device=device)
    mask = torch.zeros(batch, max_atoms, dtype=torch.bool, device=device)
    reference_energy = torch.zeros(batch, dtype=torch.float32, device=device)
    reference_forces = torch.zeros(batch, max_atoms, 3, dtype=torch.float32, device=device)
    baseline_energy = torch.zeros(batch, dtype=torch.float32, device=device)
    baseline_forces = torch.zeros(batch, max_atoms, 3, dtype=torch.float32, device=device)
    charge = torch.zeros(batch, dtype=torch.float32, device=device)
    multiplicity = torch.ones(batch, dtype=torch.float32, device=device)
    atom_count = torch.zeros(batch, dtype=torch.float32, device=device)
    z_by_symbol = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16}
    for i, row in enumerate(rows):
        n = len(row["symbols"])
        coordinates[i, :n] = torch.tensor(row["coordinates"], dtype=torch.float32, device=device)
        species[i, :n] = torch.tensor([z_by_symbol[s] for s in row["symbols"]], dtype=torch.long, device=device)
        mask[i, :n] = True
        reference_energy[i] = float(row["reference_energy"])
        reference_forces[i, :n] = torch.tensor(row["reference_forces"], dtype=torch.float32, device=device)
        baseline_energy[i] = float(row["semi_empirical_energy"])
        baseline_forces[i, :n] = torch.tensor(row["semi_empirical_forces"], dtype=torch.float32, device=device)
        charge[i] = float(row["charge"])
        multiplicity[i] = float(row["multiplicity"])
        atom_count[i] = float(n)
    return {
        "coordinates": coordinates,
        "species": species,
        "mask": mask,
        "reference_energy": reference_energy,
        "reference_forces": reference_forces,
        "baseline_energy": baseline_energy,
        "baseline_forces": baseline_forces,
        "charge": charge,
        "multiplicity": multiplicity,
        "atom_count": atom_count,
    }


def _predict(model, model_kind: str, batch: dict[str, torch.Tensor], offsets: torch.Tensor, *, create_graph: bool):
    coordinates = batch["coordinates"]
    if model_kind == "direct":
        prediction = model.predict(
            coordinates, batch["species"], batch["mask"],
            charge=batch["charge"], multiplicity=batch["multiplicity"],
            create_graph=create_graph,
        )
        energy = prediction.energy + offsets
        forces = prediction.forces
    else:
        prediction = model.predict(
            coordinates, batch["species"], batch["mask"],
            charge=batch["charge"], multiplicity=batch["multiplicity"],
            baseline_energy=batch["baseline_energy"] + offsets,
            baseline_forces=batch["baseline_forces"],
            create_graph=create_graph,
        )
        energy = prediction.energy
        forces = prediction.forces
    return energy, forces


def _loss(model, model_kind: str, batch_rows: list[dict], coefficients: np.ndarray, device: torch.device):
    batch = _batch(batch_rows, device)
    offsets = torch.tensor([_offset(row, coefficients) for row in batch_rows], dtype=torch.float32, device=device)
    energy, forces = _predict(model, model_kind, batch, offsets, create_graph=True)
    energy_error_per_atom = (energy - batch["reference_energy"]) / batch["atom_count"]
    force_mask = batch["mask"][..., None].expand_as(forces)
    force_error = torch.where(force_mask, forces - batch["reference_forces"], torch.zeros_like(forces))
    force_mse = force_error.square().sum() / force_mask.sum()
    energy_mse = energy_error_per_atom.square().mean()
    return energy_mse + FORCE_WEIGHT * force_mse, energy_mse.detach(), force_mse.detach()


def _bare_metrics(rows: list[dict], coefficients: np.ndarray) -> dict:
    return _prediction_summary(_bare_predictions(rows, coefficients))


def _model_predictions(model, model_kind: str, rows: list[dict], coefficients: np.ndarray, device: torch.device) -> tuple[list[dict], float]:
    output = []
    start = time.perf_counter()
    for begin in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[begin:begin + BATCH_SIZE]
        batch = _batch(batch_rows, device)
        offsets = torch.tensor([_offset(row, coefficients) for row in batch_rows], dtype=torch.float32, device=device)
        energy, forces = _predict(model, model_kind, batch, offsets, create_graph=False)
        energy_values = energy.detach().cpu().double().numpy()
        force_values = forces.detach().cpu().double().numpy()
        reference_energy = batch["reference_energy"].detach().cpu().double().numpy()
        reference_forces = batch["reference_forces"].detach().cpu().double().numpy()
        for index, row in enumerate(batch_rows):
            atom_count = len(row["symbols"])
            output.append({
                "parent_record_id": row["parent_record_id"],
                "config_index": row["config_index"],
                "charge": row["charge"],
                "contains_sulfur": "S" in row["symbols"],
                "predicted_energy": float(energy_values[index]),
                "reference_energy": float(reference_energy[index]),
                "predicted_forces": np.asarray(force_values[index, :atom_count], dtype=np.float64),
                "reference_forces": np.asarray(reference_forces[index, :atom_count], dtype=np.float64),
            })
    return output, time.perf_counter() - start


def _metrics(model, model_kind: str, rows: list[dict], coefficients: np.ndarray, device: torch.device) -> dict:
    predictions, elapsed = _model_predictions(model, model_kind, rows, coefficients, device)
    return _prediction_summary(predictions, evaluation_wall_seconds=elapsed)


def _train(model_kind: str, split: dict[str, list[dict]], device: torch.device, seed: int) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model_cls = DirectEnergyModel if model_kind == "direct" else DeltaEnergyModel
    model = model_cls(
        num_species=17,
        hidden_dim=HIDDEN_DIM,
        radial_features=RADIAL_FEATURES,
        distance_scale=DISTANCE_SCALE,
    ).to(device)
    coefficients = _fit_offsets(split["train"], model_kind)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model.eval()
    initial_val = _metrics(model, model_kind, split["validation"], coefficients, device)
    initial_score = initial_val["energy_mae_kcal_per_mol"] / 50.0 + initial_val["force_component_mae_hartree_per_angstrom"] / 0.01
    best = initial_score
    best_state = copy.deepcopy(model.state_dict())
    history = [{
        "epoch": 0,
        "train_total_loss": None,
        "train_energy_mse_per_atom": None,
        "train_force_component_mse": None,
        "validation_score": initial_score,
        "validation_energy_mae_kcal_per_mol": initial_val["energy_mae_kcal_per_mol"],
        "validation_force_component_mae_hartree_per_angstrom": initial_val["force_component_mae_hartree_per_angstrom"],
    }]
    train_started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = list(split["train"])
        random.Random(seed + epoch).shuffle(order)
        total = energy_sum = force_sum = 0.0
        batches = 0
        for begin in range(0, len(order), BATCH_SIZE):
            rows = order[begin:begin + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss, energy_mse, force_mse = _loss(model, model_kind, rows, coefficients, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            total += float(loss.detach())
            energy_sum += float(energy_mse)
            force_sum += float(force_mse)
            batches += 1
        model.eval()
        val = _metrics(model, model_kind, split["validation"], coefficients, device)
        score = val["energy_mae_kcal_per_mol"] / 50.0 + val["force_component_mae_hartree_per_angstrom"] / 0.01
        history.append({
            "epoch": epoch,
            "train_total_loss": total / batches,
            "train_energy_mse_per_atom": energy_sum / batches,
            "train_force_component_mse": force_sum / batches,
            "validation_score": score,
            "validation_energy_mae_kcal_per_mol": val["energy_mae_kcal_per_mol"],
            "validation_force_component_mae_hartree_per_angstrom": val["force_component_mae_hartree_per_angstrom"],
        })
        if best is None or score < best:
            best = score
            best_state = copy.deepcopy(model.state_dict())
    train_seconds = time.perf_counter() - train_started
    assert best_state is not None
    model.load_state_dict(best_state)
    best_epoch = min(history, key=lambda row: row["validation_score"])["epoch"]
    test_slices = {}
    for name, predicate in {
        "charged": lambda row: row["charge"] != 0,
        "neutral": lambda row: row["charge"] == 0,
        "sulfur": lambda row: "S" in row["symbols"],
        "no_sulfur": lambda row: "S" not in row["symbols"],
    }.items():
        selected = [row for row in split["test"] if predicate(row)]
        test_slices[name] = _metrics(model, model_kind, selected, coefficients, device) if selected else {"systems": 0}
    return {
        "model_kind": model_kind,
        "seed": seed,
        "model_config": {
            "hidden_dim": HIDDEN_DIM,
            "radial_features": RADIAL_FEATURES,
            "distance_scale": DISTANCE_SCALE,
            "num_species": 17,
        },
        "training": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "force_weight": FORCE_WEIGHT,
            "gradient_clip": GRADIENT_CLIP,
            "best_epoch": best_epoch,
            "wall_seconds": train_seconds,
            "device": str(device),
        },
        "element_offsets_hartree": {str(z): float(value) for z, value in zip(ELEMENTS, coefficients)},
        "metrics": {
            name: _metrics(model, model_kind, split[name], coefficients, device)
            for name in ("train", "validation", "test")
        },
        "test_slices": test_slices,
        "history": history,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-parent-count", type=int, choices=(12, 24, 48), default=48)
    args = parser.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    rows = _load(args.pairs, args.manifest)
    split = {name: [row for row in rows if row["split"] == name] for name in ("train", "validation", "test")}
    parent_to_split: dict[str, str] = {}
    parent_counts: dict[str, int] = {}
    for row in rows:
        parent = row["parent_record_id"]
        observed = parent_to_split.setdefault(parent, row["split"])
        if observed != row["split"]:
            raise SystemExit(f"parent group leaks across splits: {parent}")
        parent_counts[parent] = parent_counts.get(parent, 0) + 1
    parent_split_counts = {
        name: sum(split_name == name for split_name in parent_to_split.values())
        for name in ("train", "validation", "test")
    }
    if parent_split_counts != {"train": 48, "validation": 8, "test": 8}:
        raise SystemExit(f"unexpected parent split sizes: {parent_split_counts}")
    configs_per_parent_values = sorted(set(parent_counts.values()))
    if len(configs_per_parent_values) != 1 or configs_per_parent_values[0] not in {1, 4}:
        raise SystemExit(f"unexpected configurations per parent: {configs_per_parent_values}")
    configs_per_parent = configs_per_parent_values[0]
    expected_split_sizes = {name: count * configs_per_parent for name, count in parent_split_counts.items()}
    if {name: len(value) for name, value in split.items()} != expected_split_sizes:
        raise SystemExit(f"unexpected row split sizes: { {name: len(value) for name, value in split.items()} }")
    ranked_training_parents = _rank_training_parents([
        parent for parent, split_name in parent_to_split.items() if split_name == "train"
    ])
    selected_training_parents = _select_training_parent_ids(ranked_training_parents, args.train_parent_count)
    selected_training_parent_set = set(selected_training_parents)
    split["train"] = [
        row for row in split["train"] if row["parent_record_id"] in selected_training_parent_set
    ]
    expected_train_rows = args.train_parent_count * configs_per_parent
    if len(split["train"]) != expected_train_rows:
        raise SystemExit(f"selected train rows {len(split['train'])}; expected {expected_train_rows}")
    baseline_coefficients = _fit_offsets(split["train"], "delta")
    baseline = {
        "element_offsets_hartree": {str(z): float(value) for z, value in zip(ELEMENTS, baseline_coefficients)},
        "metrics": {name: _bare_metrics(split[name], baseline_coefficients) for name in ("train", "validation", "test")},
        "test_slices": {},
    }
    for name, predicate in {
        "charged": lambda row: row["charge"] != 0,
        "neutral": lambda row: row["charge"] == 0,
        "sulfur": lambda row: "S" in row["symbols"],
        "no_sulfur": lambda row: "S" not in row["symbols"],
    }.items():
        selected = [row for row in split["test"] if predicate(row)]
        baseline["test_slices"][name] = _bare_metrics(selected, baseline_coefficients) if selected else {"systems": 0}
    results = [_train(kind, split, device, args.seed) for kind in ("direct", "delta")]
    report = {
        "schema": "xtbflow-energy-baseline-run/v1",
        "data_identity": {
            "pairs_sha256": _sha256(args.pairs),
            "manifest_sha256": _sha256(args.manifest),
            "pair_count": len(rows),
            "split_counts": {name: len(value) for name, value in split.items()},
            "available_parent_split_counts": parent_split_counts,
            "selected_train_parent_count": args.train_parent_count,
            "selected_training_parents": selected_training_parents,
            "training_parent_rank_seed": TRAIN_PARENT_RANK_SEED,
            "configs_per_parent": configs_per_parent,
        },
        "protocol": {
            "seed": args.seed,
            "split_seed": "spice2-openff-pilot-v1-7",
            "split_ratios": {"train": 0.75, "validation": 0.125, "test": 0.125},
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "force_weight": FORCE_WEIGHT,
            "gradient_clip": GRADIENT_CLIP,
            "validation_score_formula": "energy_mae_kcal_per_mol/50 + force_component_mae_hartree_per_angstrom/0.01",
            "model_capacity_matched": True,
            "validation_checkpoint_selection": True,
            "test_used_during_selection": False,
            "train_parent_count": args.train_parent_count,
            "training_parent_rank_seed": TRAIN_PARENT_RANK_SEED,
            "force_direction_min_reference_norm_hartree_per_angstrom": FORCE_DIRECTION_MIN_REFERENCE_NORM,
        },
        "bare_gfn2": baseline,
        "results": results,
        "limits": [
            "One specified training seed per run; aggregate across the frozen multi-seed matrix before drawing stability conclusions.",
            "Element offsets are fit on train only and are constant with respect to coordinates, so they contribute no force.",
            "Direct and Delta use identical invariant network capacity and optimization budget.",
            "The test split is evaluated only after validation selects the checkpoint.",
            "Parent identities and split assignments are frozen; additional configurations never cross parent-group split boundaries.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        result["model_kind"]: {
            "best_epoch": result["training"]["best_epoch"],
            "test": result["metrics"]["test"],
            "wall_seconds": result["training"]["wall_seconds"],
        }
        for result in results
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
