from __future__ import annotations

import math

import pytest
import torch

from xtbflow.models import DeltaEnergyModel, DirectEnergyModel
from xtbflow.training import energy_force_loss, make_run_manifest


def batch():
    coordinates = torch.tensor([[[0.0, 0.0, 0.0], [0.95, 0.1, 0.0], [-0.2, 0.9, 0.2], [0.0, 0.0, 0.0]]], dtype=torch.float64)
    species = torch.tensor([[8, 1, 1, 0]], dtype=torch.long)
    atom_mask = torch.tensor([[True, True, True, False]])
    charge = torch.tensor([0.0], dtype=torch.float64)
    multiplicity = torch.tensor([1.0], dtype=torch.float64)
    return coordinates, species, atom_mask, charge, multiplicity


def test_delta_energy_force_is_conservative_and_invariant():
    torch.manual_seed(4)
    coordinates, species, atom_mask, charge, multiplicity = batch()
    model = DeltaEnergyModel(num_species=16, hidden_dim=12, radial_features=6, distance_scale=4.0).double()
    baseline_energy = torch.tensor([[-1.0]], dtype=torch.float64).reshape(1)
    baseline_forces = torch.zeros_like(coordinates)
    result = model.predict(coordinates, species, atom_mask, charge=charge, multiplicity=multiplicity, baseline_energy=baseline_energy, baseline_forces=baseline_forces, create_graph=True)
    assert result.energy.shape == (1,)
    assert result.forces.shape == coordinates.shape
    assert torch.isfinite(result.forces).all()
    assert torch.allclose(result.forces[0, 3], torch.zeros(3, dtype=torch.float64))

    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    moved = coordinates @ rotation.T + torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64)
    moved_result = model.predict(moved, species, atom_mask, charge=charge, multiplicity=multiplicity, baseline_energy=baseline_energy, baseline_forces=baseline_forces, create_graph=False)
    assert torch.allclose(result.energy, moved_result.energy, atol=1e-8, rtol=1e-8)
    assert torch.allclose(result.forces @ rotation.T, moved_result.forces, atol=1e-7, rtol=1e-7)

    permutation = torch.tensor([1, 0, 2, 3])
    permuted = model.predict(coordinates[:, permutation], species[:, permutation], atom_mask[:, permutation], charge=charge, multiplicity=multiplicity, baseline_energy=baseline_energy, baseline_forces=baseline_forces[:, permutation], create_graph=False)
    assert torch.allclose(result.energy, permuted.energy, atol=1e-8, rtol=1e-8)
    assert torch.allclose(result.forces[:, permutation], permuted.forces, atol=1e-7, rtol=1e-7)


def test_direct_model_force_matches_finite_difference():
    torch.manual_seed(9)
    coordinates, species, atom_mask, charge, multiplicity = batch()
    model = DirectEnergyModel(num_species=16, hidden_dim=8, radial_features=5, distance_scale=4.0).double()
    result = model.predict(coordinates, species, atom_mask, charge=charge, multiplicity=multiplicity)
    index = (0, 1, 0)
    step = 1e-5
    plus = coordinates.detach().clone()
    minus = coordinates.detach().clone()
    plus[index] += step
    minus[index] -= step
    e_plus = model(plus, species, atom_mask, charge=charge, multiplicity=multiplicity)
    e_minus = model(minus, species, atom_mask, charge=charge, multiplicity=multiplicity)
    finite_difference_force = -float((e_plus - e_minus) / (2 * step))
    assert result.forces[index].item() == pytest.approx(finite_difference_force, rel=2e-4, abs=2e-5)


def test_energy_force_loss_masks_missing_labels_and_manifest_splits():
    predicted_energy = torch.tensor([1.0, 2.0])
    target_energy = torch.tensor([0.0, float("nan")])
    predicted_forces = torch.zeros(2, 2, 3)
    target_forces = torch.zeros_like(predicted_forces)
    target_forces[0, 0, 0] = 2.0
    losses = energy_force_loss(predicted_energy, target_energy, predicted_forces, target_forces, energy_mask=torch.tensor([True, False]), force_mask=torch.tensor([[True, False], [False, False]]))
    assert losses["energy"].item() == pytest.approx(1.0)
    assert losses["force"].item() == pytest.approx(4.0 / 3.0)
    manifest = make_run_manifest(model="delta", model_config={"hidden_dim": 8}, data_identity="data-v1", calculator_protocol="gfn2-v1", seed=3, train_ids=["a"], validation_ids=["b"])
    assert manifest["manifest_hash"]
    with pytest.raises(ValueError, match="disjoint"):
        make_run_manifest(model="delta", model_config={}, data_identity="d", calculator_protocol="p", seed=1, train_ids=["same"], validation_ids=["same"])

