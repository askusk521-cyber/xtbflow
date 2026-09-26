from __future__ import annotations

import math

import pytest
import torch

from xtbflow.data import EndpointGeometryInput, GeometryTarget, ReactantEventGeometryInput
from xtbflow.models import EquivariantGeometryFlow, ReactionDirectionHead, sign_invariant_direction_loss
from xtbflow.training import endpoint_geometry_loss, reactant_direction_loss, result_metadata


def fixture():
    torch.manual_seed(4)
    coords = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.1, 1.1, -0.2], [0.0, 0.0, 0.0]]])
    features = torch.randn(1, 4, 3)
    mask = torch.tensor([[True, True, True, False]])
    return coords, features, mask


def test_equivariant_flow_respects_translation_rotation_and_padding():
    coords, features, mask = fixture()
    model = EquivariantGeometryFlow(3, hidden_dim=12, radial_features=5).double()
    coords, features = coords.double(), features.double()
    output = model(coords, features, mask)
    theta = torch.tensor(math.pi / 3, dtype=torch.float64)
    rotation = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0.0], [torch.sin(theta), torch.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    shifted = coords + torch.tensor([5.0, -3.0, 2.0])
    rotated = torch.matmul(coords, rotation.T)
    assert torch.allclose(model(shifted, features, mask), output, atol=1e-8, rtol=1e-7)
    assert torch.allclose(model(rotated, features, mask), torch.matmul(output, rotation.T), atol=1e-8, rtol=1e-7)
    assert torch.equal(output[:, 3], torch.zeros(1, 3, dtype=output.dtype))


def test_flow_is_permutation_equivariant_and_pair_condition_is_used():
    coords, features, mask = fixture()
    model = EquivariantGeometryFlow(3, hidden_dim=12, radial_features=5)
    pair = torch.zeros(1, 4, 4)
    pair[:, 0, 1] = pair[:, 1, 0] = 1.0
    order = torch.tensor([2, 0, 1, 3])
    output = model(coords, features, mask, pair_features=pair)
    permuted = model(coords[:, order], features[:, order], mask[:, order], pair_features=pair[:, order][:, :, order])
    restored = torch.empty_like(permuted)
    restored[:, order] = permuted
    assert torch.allclose(restored, output, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(output, model(coords, features, mask), atol=1e-7, rtol=1e-7)


def test_direction_head_and_loss_are_proper_rotation_and_sign_invariant():
    coords, features, mask = fixture()
    head = ReactionDirectionHead(3, hidden_dim=10).double()
    coords, features = coords.double(), features.double()
    direction = head(features, coords, mask)
    theta = torch.tensor(0.41, dtype=torch.float64)
    rotation = torch.tensor([[torch.cos(theta), 0.0, torch.sin(theta)], [0.0, 1.0, 0.0], [-torch.sin(theta), 0.0, torch.cos(theta)]])
    rotated = head(features, torch.matmul(coords, rotation.T), mask)
    assert torch.allclose(rotated, torch.matmul(direction, rotation.T), atol=1e-8, rtol=1e-7)
    assert sign_invariant_direction_loss(direction, direction, mask) == pytest.approx(0.0, abs=1e-7)
    assert sign_invariant_direction_loss(direction, -direction, mask) == pytest.approx(0.0, abs=1e-7)


def test_task_views_keep_endpoint_and_reactant_modes_separate():
    coords, features, mask = fixture()
    endpoint = EndpointGeometryInput(coords, mask, coords + 0.1)
    endpoint.validate()
    target = GeometryTarget(coords + 0.1, mask.clone())
    assert endpoint_geometry_loss(coords + 0.1, endpoint, target) == pytest.approx(0.0, abs=1e-8)
    assert result_metadata(endpoint)[0] == "task_mode:endpoint_geometry"
    pair = torch.zeros(1, 1, 4, 4)
    reactant = ReactantEventGeometryInput(coords, mask, features[:, :1], pair)
    reactant.validate()
    assert result_metadata(reactant)[1] == "information:reactant_only"
    assert reactant_direction_loss(coords, -coords, reactant) == pytest.approx(0.0, abs=1e-7)
    masked_target = GeometryTarget(torch.full_like(coords, float("nan")), torch.zeros_like(mask))
    masked_target.validate(batch=1, atoms=4, device=coords.device, dtype=coords.dtype)


def test_input_firewall_rejects_bad_modes_and_invalid_pair_edits():
    coords, features, mask = fixture()
    with pytest.raises(ValueError, match="endpoint_geometry"):
        EndpointGeometryInput(coords, mask, coords, mode="reactant_event_geometry").validate()
    bad_pair = torch.zeros(1, 1, 4, 4)
    bad_pair[:, :, 0, 1] = 1.0
    reactant = ReactantEventGeometryInput(coords, mask, features[:, :1], bad_pair)
    with pytest.raises(ValueError, match="symmetric"):
        reactant.validate()
    with pytest.raises(ValueError, match="direction tensors"):
        reactant_direction_loss(torch.zeros(1, 3, 3), coords, ReactantEventGeometryInput(coords, mask, features[:, :1], torch.zeros(1, 1, 4, 4)))
