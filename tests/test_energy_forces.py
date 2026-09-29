from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from xtbflow.models import DeltaEnergyModel, DirectEnergyModel
from xtbflow.training import energy_force_loss, make_run_manifest

_SPEC = importlib.util.spec_from_file_location(
    "run_energy_baselines_for_test",
    Path(__file__).resolve().parents[1] / "scripts" / "run_energy_baselines.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)
_prediction_summary = _RUNNER._prediction_summary
_select_training_parent_ids = _RUNNER._select_training_parent_ids


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



def test_energy_models_start_from_zero_residual_and_delta_baseline():
    coordinates, species, atom_mask, charge, multiplicity = batch()
    direct = DirectEnergyModel(num_species=16, hidden_dim=8, radial_features=5, distance_scale=4.0).double()
    delta = DeltaEnergyModel(num_species=16, hidden_dim=8, radial_features=5, distance_scale=4.0).double()
    direct_result = direct.predict(coordinates, species, atom_mask, charge=charge, multiplicity=multiplicity)
    baseline_energy = torch.tensor([-1.25], dtype=torch.float64)
    baseline_forces = torch.tensor([[[0.1, -0.2, 0.0], [0.0, 0.3, -0.1], [0.2, 0.0, -0.2], [0.0, 0.0, 0.0]]], dtype=torch.float64)
    delta_result = delta.predict(
        coordinates,
        species,
        atom_mask,
        charge=charge,
        multiplicity=multiplicity,
        baseline_energy=baseline_energy,
        baseline_forces=baseline_forces,
    )
    assert torch.equal(direct_result.energy, torch.zeros_like(direct_result.energy))
    assert torch.equal(direct_result.forces, torch.zeros_like(direct_result.forces))
    assert torch.equal(delta_result.energy, baseline_energy)
    assert torch.equal(delta_result.forces, baseline_forces)


def test_pair_interactions_have_smooth_finite_cutoff():
    model = DirectEnergyModel(num_species=16, hidden_dim=8, radial_features=5, distance_scale=2.0).double()
    with torch.no_grad():
        model.network.pair_readout[-1].bias.fill_(1.0)
    species = torch.tensor([[1, 1]], dtype=torch.long)
    mask = torch.tensor([[True, True]])
    charge = torch.tensor([0.0], dtype=torch.float64)
    multiplicity = torch.tensor([1.0], dtype=torch.float64)
    far = torch.tensor([[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]], dtype=torch.float64)
    just_inside = torch.tensor([[[0.0, 0.0, 0.0], [1.999, 0.0, 0.0]]], dtype=torch.float64)
    far_energy = model(far, species, mask, charge=charge, multiplicity=multiplicity)
    inside_energy = model(just_inside, species, mask, charge=charge, multiplicity=multiplicity)
    assert far_energy.item() == pytest.approx(0.0, abs=1e-12)
    assert 0.0 < inside_energy.item() < 1e-5

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



def test_learning_curve_parent_subsets_are_deterministic_and_nested():
    parents = [f"parent-{index:02d}" for index in range(48)]
    selected_12 = _select_training_parent_ids(parents, 12)
    selected_24 = _select_training_parent_ids(list(reversed(parents)), 24)
    selected_48 = _select_training_parent_ids(parents, 48)
    assert selected_12 == selected_24[:12]
    assert selected_24 == selected_48[:24]
    assert set(selected_48) == set(parents)


def test_prediction_summary_reports_parent_relative_energy_and_force_direction():
    predictions = [
        {
            "parent_record_id": "parent-a",
            "config_index": 0,
            "charge": 0,
            "contains_sulfur": False,
            "predicted_energy": 0.0,
            "reference_energy": 0.0,
            "predicted_forces": np.asarray([[1.0, 0.0, 0.0]]),
            "reference_forces": np.asarray([[1.0, 0.0, 0.0]]),
        },
        {
            "parent_record_id": "parent-a",
            "config_index": 3,
            "charge": 0,
            "contains_sulfur": False,
            "predicted_energy": 1.2,
            "reference_energy": 1.0,
            "predicted_forces": np.asarray([[0.0, 1.0, 0.0]]),
            "reference_forces": np.asarray([[1.0, 0.0, 0.0]]),
        },
    ]
    summary = _prediction_summary(predictions)
    assert summary["independent_parent_count"] == 1
    assert summary["relative_energy_comparisons"] == 1
    assert summary["relative_energy_mae_kcal_per_mol"] == pytest.approx(0.2 * 627.5094740631)
    assert summary["force_component_mae_hartree_per_angstrom"] == pytest.approx(1.0 / 3.0)
    assert summary["force_vector_mean_angle_degrees"] == pytest.approx(45.0)
    parent = summary["parent_metrics"][0]
    assert parent["configs"] == 2
    assert parent["relative_energy_mae_kcal_per_mol"] == pytest.approx(0.2 * 627.5094740631)


def test_learning_curve_frozen_config_matches_runner_protocol():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "spice2_energy_learning_curve_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["train_parent_rank_seed"] == _RUNNER.TRAIN_PARENT_RANK_SEED
    assert config["train_parent_counts"] == [12, 24, 48]
    assert config["training_seeds"] == [20260929, 20260930, 20261001]
    assert config["training"]["epochs"] == _RUNNER.EPOCHS
    assert config["training"]["batch_size"] == _RUNNER.BATCH_SIZE
    assert config["training"]["learning_rate"] == _RUNNER.LR
    assert config["training"]["force_weight"] == _RUNNER.FORCE_WEIGHT
    assert config["training"]["gradient_clip"] == _RUNNER.GRADIENT_CLIP
    assert config["training"]["hidden_dim"] == _RUNNER.HIDDEN_DIM
    assert config["training"]["radial_features"] == _RUNNER.RADIAL_FEATURES
    assert config["training"]["distance_scale_angstrom"] == _RUNNER.DISTANCE_SCALE
