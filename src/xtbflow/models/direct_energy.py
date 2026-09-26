"""Direct reference-energy baseline with the same invariant capacity as DeltaE."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .delta_energy import EnergyForcePrediction, _InvariantEnergyNetwork, _check_inputs


class DirectEnergyModel(nn.Module):
    """Fit reference energy directly using the Delta model's architecture."""

    def __init__(self, *, num_species: int = 128, hidden_dim: int = 64, radial_features: int = 16, distance_scale: float = 8.0):
        super().__init__()
        self.network = _InvariantEnergyNetwork(num_species=num_species, hidden_dim=hidden_dim, radial_features=radial_features, distance_scale=distance_scale)

    def forward(self, coordinates: Tensor, species: Tensor, atom_mask: Tensor, *, charge: Tensor, multiplicity: Tensor) -> Tensor:
        return self.network(coordinates, species, atom_mask, charge, multiplicity)

    def predict(self, coordinates: Tensor, species: Tensor, atom_mask: Tensor, *, charge: Tensor, multiplicity: Tensor, create_graph: bool = False) -> EnergyForcePrediction:
        _check_inputs(coordinates, species, atom_mask, charge, multiplicity)
        coordinates_for_grad = coordinates if coordinates.requires_grad else coordinates.detach().requires_grad_(True)
        energy = self.forward(coordinates_for_grad, species, atom_mask, charge=charge, multiplicity=multiplicity)
        gradient = torch.autograd.grad(energy.sum(), coordinates_for_grad, create_graph=create_graph, retain_graph=create_graph)[0]
        forces = torch.where(atom_mask[..., None], -gradient, torch.zeros_like(gradient))
        return EnergyForcePrediction(energy=energy, forces=forces)

