"""Invariant scalar Delta-energy potentials with conservative forces.

The model consumes atom species, coordinates, charge, multiplicity, and a
padding mask. Its scalar output is invariant to translations, proper
rotations, and simultaneous atom permutations. Forces come from the scalar
correction, so a separate force head cannot drift away from the energy.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


def _check_inputs(coordinates: Tensor, species: Tensor, atom_mask: Tensor, charge: Tensor, multiplicity: Tensor) -> tuple[int, int]:
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3 or not coordinates.is_floating_point():
        raise ValueError("coordinates must be floating [B,N,3]")
    batch, atoms, _ = coordinates.shape
    if species.shape != (batch, atoms) or species.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("species must be integer [B,N]")
    if atom_mask.shape != (batch, atoms) or atom_mask.dtype is not torch.bool or atom_mask.device != coordinates.device:
        raise ValueError("atom_mask must be boolean [B,N] on the coordinate device")
    for name, value in (("charge", charge), ("multiplicity", multiplicity)):
        numeric = value.is_floating_point() or value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64)
        if value.shape != (batch,) or not numeric or value.device != coordinates.device:
            raise ValueError(f"{name} must be a numeric [B] tensor on the coordinate device")
    if not bool(torch.isfinite(coordinates).all()) or bool((species < 0).any()):
        raise ValueError("coordinates must be finite and species indices nonnegative")
    if not bool(torch.isfinite(charge.to(dtype=coordinates.dtype)).all()) or not bool(torch.isfinite(multiplicity.to(dtype=coordinates.dtype)).all()):
        raise ValueError("charge and multiplicity must be finite")
    if bool((multiplicity < 1).any()) or not bool(atom_mask.any(dim=1).all()):
        raise ValueError("multiplicity must be positive and each item needs one active atom")
    return batch, atoms


@dataclass(frozen=True)
class EnergyForcePrediction:
    """Energy and conservative force tensors returned by a potential."""

    energy: Tensor
    forces: Tensor
    delta_energy: Tensor | None = None


class _InvariantEnergyNetwork(nn.Module):
    """Small pairwise invariant network shared by Delta and direct baselines."""

    def __init__(self, *, num_species: int, hidden_dim: int, radial_features: int, distance_scale: float):
        super().__init__()
        if type(num_species) is not int or num_species < 1:
            raise ValueError("num_species must be a positive integer")
        if any(type(value) is not int or value < 1 for value in (hidden_dim, radial_features)):
            raise ValueError("hidden_dim and radial_features must be positive integers")
        if not isinstance(distance_scale, (int, float)) or not math.isfinite(float(distance_scale)) or distance_scale <= 0:
            raise ValueError("distance_scale must be finite and positive")
        self.num_species = num_species
        self.embedding = nn.Embedding(num_species, hidden_dim)
        self.global_features = nn.Sequential(nn.Linear(2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.atom_readout = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.pair_readout = nn.Sequential(nn.Linear(2 * hidden_dim + radial_features, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.register_buffer("radial_centers", torch.linspace(0.0, float(distance_scale), radial_features))
        spacing = float(distance_scale) / max(radial_features - 1, 1)
        self.register_buffer("radial_width", torch.tensor(max(spacing, 1e-6)))

    def forward(self, coordinates: Tensor, species: Tensor, atom_mask: Tensor, charge: Tensor, multiplicity: Tensor) -> Tensor:
        _, atoms = _check_inputs(coordinates, species, atom_mask, charge, multiplicity)
        if bool((species >= self.num_species).any()):
            raise ValueError(f"species index must be smaller than num_species={self.num_species}")
        node = self.embedding(species.to(dtype=torch.long))
        global_input = torch.stack((charge.to(dtype=coordinates.dtype), multiplicity.to(dtype=coordinates.dtype)), dim=-1)
        node = node + self.global_features(global_input)[:, None, :]
        active = atom_mask.to(dtype=coordinates.dtype)
        atom_energy = self.atom_readout(node).squeeze(-1) * active

        displacement = coordinates[:, :, None, :] - coordinates[:, None, :, :]
        distance = torch.sqrt(displacement.square().sum(dim=-1) + 1e-12)
        centers = self.radial_centers.to(device=coordinates.device, dtype=coordinates.dtype)
        width = self.radial_width.to(device=coordinates.device, dtype=coordinates.dtype)
        radial = torch.exp(-((distance[..., None] - centers) / width).square())
        left = node[:, :, None, :].expand(-1, -1, atoms, -1)
        right = node[:, None, :, :].expand(-1, atoms, -1, -1)
        pair_input = torch.cat((left + right, left * right, radial), dim=-1)
        pair_energy = self.pair_readout(pair_input).squeeze(-1)
        pair_mask = atom_mask[:, :, None] & atom_mask[:, None, :]
        pair_mask = pair_mask & ~torch.eye(atoms, dtype=torch.bool, device=coordinates.device)[None]
        pair_energy = torch.where(pair_mask, pair_energy, torch.zeros_like(pair_energy))
        return atom_energy.sum(dim=1) + 0.5 * pair_energy.sum(dim=(1, 2))


class DeltaEnergyModel(nn.Module):
    """Learn ``DeltaE`` while keeping semi-empirical values external."""

    def __init__(self, *, num_species: int = 128, hidden_dim: int = 64, radial_features: int = 16, distance_scale: float = 8.0):
        super().__init__()
        self.network = _InvariantEnergyNetwork(num_species=num_species, hidden_dim=hidden_dim, radial_features=radial_features, distance_scale=distance_scale)

    def forward(self, coordinates: Tensor, species: Tensor, atom_mask: Tensor, *, charge: Tensor, multiplicity: Tensor) -> Tensor:
        return self.network(coordinates, species, atom_mask, charge, multiplicity)

    def correction_forces(self, coordinates: Tensor, species: Tensor, atom_mask: Tensor, *, charge: Tensor, multiplicity: Tensor, create_graph: bool = False) -> tuple[Tensor, Tensor]:
        coordinates_for_grad = coordinates if coordinates.requires_grad else coordinates.detach().requires_grad_(True)
        delta = self.forward(coordinates_for_grad, species, atom_mask, charge=charge, multiplicity=multiplicity)
        gradient = torch.autograd.grad(delta.sum(), coordinates_for_grad, create_graph=create_graph, retain_graph=create_graph)[0]
        return delta, -gradient

    def predict(self, coordinates: Tensor, species: Tensor, atom_mask: Tensor, *, charge: Tensor, multiplicity: Tensor, baseline_energy: Tensor, baseline_forces: Tensor, create_graph: bool = False) -> EnergyForcePrediction:
        batch, _ = _check_inputs(coordinates, species, atom_mask, charge, multiplicity)
        if baseline_energy.shape != (batch,) or baseline_energy.device != coordinates.device or not baseline_energy.is_floating_point():
            raise ValueError("baseline_energy must be floating [B] on the coordinate device")
        if baseline_forces.shape != coordinates.shape or baseline_forces.device != coordinates.device or baseline_forces.dtype != coordinates.dtype:
            raise ValueError("baseline_forces must be floating [B,N,3] matching coordinates")
        if not bool(torch.isfinite(baseline_energy).all()) or not bool(torch.isfinite(baseline_forces).all()):
            raise ValueError("baseline energy and forces must be finite")
        delta, correction_forces = self.correction_forces(coordinates, species, atom_mask, charge=charge, multiplicity=multiplicity, create_graph=create_graph)
        forces = torch.where(atom_mask[..., None], baseline_forces + correction_forces, torch.zeros_like(baseline_forces))
        return EnergyForcePrediction(energy=baseline_energy + delta, forces=forces, delta_energy=delta)

