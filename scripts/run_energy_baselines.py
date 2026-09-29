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
SEED = 20260929
EPOCHS = 80
BATCH_SIZE = 4
LR = 1.0e-3
HIDDEN_DIM = 32
RADIAL_FEATURES = 8
DISTANCE_SCALE = 8.0
FORCE_WEIGHT = 10.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(pair_path: Path, manifest_path: Path) -> list[dict]:
    split_by_id = {
        row["source_record_id"]: row["split"]
        for row in map(json.loads, manifest_path.read_text(encoding="utf-8").splitlines())
        if row
    }
    rows = []
    for row in map(json.loads, pair_path.read_text(encoding="utf-8").splitlines()):
        if row["status"] != "success":
            continue
        row["split"] = split_by_id[row["source_record_id"]]
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
    energy_error = np.asarray([
        row["semi_empirical_energy"] + _offset(row, coefficients) - row["reference_energy"]
        for row in rows
    ], dtype=np.float64)
    force_error = np.concatenate([
        (np.asarray(row["semi_empirical_forces"], dtype=np.float64) - np.asarray(row["reference_forces"], dtype=np.float64)).reshape(-1)
        for row in rows
    ])
    return {
        "systems": len(rows),
        "energy_mae_kcal_per_mol": float(np.mean(np.abs(energy_error)) * HARTREE_TO_KCAL_MOL),
        "energy_rmse_kcal_per_mol": float(np.sqrt(np.mean(energy_error ** 2)) * HARTREE_TO_KCAL_MOL),
        "energy_max_abs_kcal_per_mol": float(np.max(np.abs(energy_error)) * HARTREE_TO_KCAL_MOL),
        "force_component_mae_hartree_per_angstrom": float(np.mean(np.abs(force_error))),
        "force_component_rmse_hartree_per_angstrom": float(np.sqrt(np.mean(force_error ** 2))),
        "force_component_p95_abs_hartree_per_angstrom": float(np.quantile(np.abs(force_error), 0.95)),
    }


def _metrics(model, model_kind: str, rows: list[dict], coefficients: np.ndarray, device: torch.device) -> dict:
    energies = []
    targets = []
    force_errors = []
    start = time.perf_counter()
    for begin in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[begin:begin + BATCH_SIZE]
        batch = _batch(batch_rows, device)
        offsets = torch.tensor([_offset(row, coefficients) for row in batch_rows], dtype=torch.float32, device=device)
        energy, forces = _predict(model, model_kind, batch, offsets, create_graph=False)
        energies.extend(energy.detach().cpu().double().tolist())
        targets.extend(batch["reference_energy"].detach().cpu().double().tolist())
        for i, row in enumerate(batch_rows):
            n = len(row["symbols"])
            diff = (forces[i, :n] - batch["reference_forces"][i, :n]).detach().cpu().double().numpy()
            force_errors.append(diff.reshape(-1))
    elapsed = time.perf_counter() - start
    energy_error = np.asarray(energies) - np.asarray(targets)
    force_error = np.concatenate(force_errors)
    return {
        "systems": len(rows),
        "energy_mae_kcal_per_mol": float(np.mean(np.abs(energy_error)) * HARTREE_TO_KCAL_MOL),
        "energy_rmse_kcal_per_mol": float(np.sqrt(np.mean(energy_error ** 2)) * HARTREE_TO_KCAL_MOL),
        "energy_max_abs_kcal_per_mol": float(np.max(np.abs(energy_error)) * HARTREE_TO_KCAL_MOL),
        "force_component_mae_hartree_per_angstrom": float(np.mean(np.abs(force_error))),
        "force_component_rmse_hartree_per_angstrom": float(np.sqrt(np.mean(force_error ** 2))),
        "force_component_p95_abs_hartree_per_angstrom": float(np.quantile(np.abs(force_error), 0.95)),
        "evaluation_wall_seconds": elapsed,
    }


def _train(model_kind: str, split: dict[str, list[dict]], device: torch.device) -> dict:
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
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
        random.Random(SEED + epoch).shuffle(order)
        total = energy_sum = force_sum = 0.0
        batches = 0
        for begin in range(0, len(order), BATCH_SIZE):
            rows = order[begin:begin + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss, energy_mse, force_mse = _loss(model, model_kind, rows, coefficients, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
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
    return {
        "model_kind": model_kind,
        "seed": SEED,
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
            "best_epoch": best_epoch,
            "wall_seconds": train_seconds,
            "device": str(device),
        },
        "element_offsets_hartree": {str(z): float(value) for z, value in zip(ELEMENTS, coefficients)},
        "metrics": {
            name: _metrics(model, model_kind, split[name], coefficients, device)
            for name in ("train", "validation", "test")
        },
        "history": history,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    rows = _load(args.pairs, args.manifest)
    split = {name: [row for row in rows if row["split"] == name] for name in ("train", "validation", "test")}
    if {name: len(value) for name, value in split.items()} != {"train": 48, "validation": 8, "test": 8}:
        raise SystemExit(f"unexpected split sizes: { {name: len(value) for name, value in split.items()} }")
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
    results = [_train(kind, split, device) for kind in ("direct", "delta")]
    report = {
        "schema": "xtbflow-energy-baseline-run/v1",
        "data_identity": {
            "pairs_sha256": _sha256(args.pairs),
            "manifest_sha256": _sha256(args.manifest),
            "pair_count": len(rows),
            "split_counts": {name: len(value) for name, value in split.items()},
        },
        "protocol": {
            "seed": SEED,
            "split_seed": "spice2-openff-pilot-v1-7",
            "split_ratios": {"train": 0.75, "validation": 0.125, "test": 0.125},
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "force_weight": FORCE_WEIGHT,
            "model_capacity_matched": True,
            "validation_checkpoint_selection": True,
            "test_used_during_selection": False,
        },
        "bare_gfn2": baseline,
        "results": results,
        "limits": [
            "One development seed only; this is not a final learning curve.",
            "Element offsets are fit on train only and are constant with respect to coordinates, so they contribute no force.",
            "Direct and Delta use identical invariant network capacity and optimization budget.",
            "The test split is evaluated only after validation selects the checkpoint.",
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
