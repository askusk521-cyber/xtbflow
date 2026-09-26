"""Masked energy/force objectives and reproducible training manifests."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from xtbflow.models.delta_energy import DeltaEnergyModel, EnergyForcePrediction


def _masked_mean(value: Tensor, mask: Tensor, *, name: str) -> Tensor:
    if mask.dtype is not torch.bool or mask.shape != value.shape:
        raise ValueError(f"{name} mask must be boolean with shape {tuple(value.shape)}")
    count = mask.sum()
    if int(count) == 0:
        return value.sum() * 0.0
    return torch.where(mask, value, torch.zeros_like(value)).sum() / count


def energy_force_loss(
    predicted_energy: Tensor,
    target_energy: Tensor,
    predicted_forces: Tensor | None = None,
    target_forces: Tensor | None = None,
    *,
    energy_mask: Tensor | None = None,
    force_mask: Tensor | None = None,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
) -> dict[str, Tensor]:
    """Return separate energy, force, and weighted losses.

    Missing labels are represented by masks and contribute exactly zero; no
    placeholder target is silently treated as a physical value.
    """

    if predicted_energy.shape != target_energy.shape or predicted_energy.ndim != 1 or not predicted_energy.is_floating_point() or not target_energy.is_floating_point():
        raise ValueError("energy tensors must be floating [B] with matching shapes")
    if energy_mask is None:
        energy_mask = torch.ones_like(target_energy, dtype=torch.bool)
    if energy_mask.shape != target_energy.shape or energy_mask.dtype is not torch.bool:
        raise ValueError("energy_mask must be boolean [B]")
    if not bool(torch.isfinite(predicted_energy[energy_mask]).all()) or not bool(torch.isfinite(target_energy[energy_mask]).all()):
        raise ValueError("observed energy labels must be finite")
    energy = _masked_mean((predicted_energy - target_energy).square(), energy_mask, name="energy")
    force = predicted_energy.sum() * 0.0
    if predicted_forces is not None or target_forces is not None:
        if predicted_forces is None or target_forces is None or predicted_forces.shape != target_forces.shape or predicted_forces.ndim != 3 or predicted_forces.shape[-1] != 3:
            raise ValueError("predicted_forces and target_forces must both have shape [B,N,3]")
        if predicted_forces.shape[0] != predicted_energy.shape[0]:
            raise ValueError("force batch dimension must match energy")
        if force_mask is None:
            force_mask = torch.ones(predicted_forces.shape[:2], dtype=torch.bool, device=predicted_forces.device)
        if force_mask.shape != predicted_forces.shape[:2] or force_mask.dtype is not torch.bool:
            raise ValueError("force_mask must be boolean [B,N]")
        component_mask = force_mask[..., None].expand_as(predicted_forces)
        if not bool(torch.isfinite(predicted_forces[component_mask]).all()) or not bool(torch.isfinite(target_forces[component_mask]).all()):
            raise ValueError("observed force labels must be finite")
        force = _masked_mean((predicted_forces - target_forces).square(), component_mask, name="force")
    if energy_weight < 0 or force_weight < 0:
        raise ValueError("loss weights must be nonnegative")
    return {"energy": energy, "force": force, "total": energy_weight * energy + force_weight * force}


@dataclass(frozen=True)
class EnergyBatch:
    """Explicit inputs and labels for one Delta/direct training batch."""

    coordinates: Tensor
    species: Tensor
    atom_mask: Tensor
    charge: Tensor
    multiplicity: Tensor
    reference_energy: Tensor
    reference_forces: Tensor | None = None
    baseline_energy: Tensor | None = None
    baseline_forces: Tensor | None = None
    energy_mask: Tensor | None = None
    force_mask: Tensor | None = None


def predict_delta_batch(model: DeltaEnergyModel, batch: EnergyBatch, *, create_graph: bool = True) -> EnergyForcePrediction:
    """Run a Delta model against cached semi-empirical labels only."""

    if batch.baseline_energy is None or batch.baseline_forces is None:
        raise ValueError("Delta training requires cached baseline energy and forces")
    return model.predict(batch.coordinates, batch.species, batch.atom_mask, charge=batch.charge, multiplicity=batch.multiplicity, baseline_energy=batch.baseline_energy, baseline_forces=batch.baseline_forces, create_graph=create_graph)


def delta_batch_loss(model: DeltaEnergyModel, batch: EnergyBatch, *, energy_weight: float = 1.0, force_weight: float = 1.0) -> dict[str, Tensor]:
    prediction = predict_delta_batch(model, batch)
    return energy_force_loss(prediction.energy, batch.reference_energy, prediction.forces, batch.reference_forces, energy_mask=batch.energy_mask, force_mask=batch.force_mask if batch.force_mask is not None else batch.atom_mask, energy_weight=energy_weight, force_weight=force_weight)


def make_run_manifest(*, model: str, model_config: Mapping[str, Any], data_identity: str, calculator_protocol: str, seed: int, train_ids: list[str], validation_ids: list[str]) -> dict[str, Any]:
    """Create a JSON-ready manifest without inspecting validation labels."""

    if model not in {"delta", "direct"} or not data_identity.strip() or not calculator_protocol.strip():
        raise ValueError("model, data_identity, and calculator_protocol are required")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if set(train_ids) & set(validation_ids):
        raise ValueError("train and validation IDs must be disjoint")
    payload = {"schema": "xtbflow-energy-run/v1", "model": model, "model_config": dict(model_config), "data_identity": data_identity, "calculator_protocol": calculator_protocol, "seed": seed, "train_ids": list(train_ids), "validation_ids": list(validation_ids)}
    payload["manifest_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    return payload


def write_run_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    """Persist a JSON-ready manifest for later run reconstruction."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(manifest), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

