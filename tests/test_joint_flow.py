from __future__ import annotations

import pytest
import torch

from xtbflow.models import ConservationProjector, JointEventGeometryFlow, SerialEventGeometryFlow
from xtbflow.training import coupled_control_manifest, joint_flow_loss
from xtbflow.evaluation import TerminalCandidate, deduplicate_terminal_candidates


def inputs(dtype=torch.float64):
    event = torch.tensor([[0.5, -0.2, 0.1]], dtype=dtype)
    coordinates = torch.tensor([[[0.0, 0.0, 0.0], [0.8, 0.1, 0.0], [-0.2, 0.7, 0.1]]], dtype=dtype)
    node = torch.randn(1, 3, 4, dtype=dtype)
    mask = torch.tensor([[True, True, True]])
    return event, coordinates, node, mask


def test_joint_flow_exchanges_states_and_zero_coupling_is_exact_control():
    torch.manual_seed(2)
    event, coordinates, node, mask = inputs()
    projector = ConservationProjector(torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64))
    model = JointEventGeometryFlow(projector, 4, hidden_dim=8, radial_features=4).double()
    coupled = model(event, coordinates, node, mask, coupling_strength=1.0)
    zero = model(event, coordinates, node, mask, coupling_strength=0.0)
    assert coupled.event_velocity.shape == event.shape
    assert coupled.geometry_velocity.shape == coordinates.shape
    assert not torch.allclose(coupled.event_velocity, zero.event_velocity)
    assert not torch.allclose(coupled.geometry_velocity, zero.geometry_velocity)
    assert torch.allclose(zero.event_velocity @ projector.constraint_matrix.T, torch.zeros(1, 1, dtype=torch.float64), atol=1e-7)
    serial = SerialEventGeometryFlow(projector, 4, hidden_dim=8, radial_features=4).double()
    serial_out = serial(event, coordinates, node, mask)
    assert serial_out.geometry_velocity.shape == coordinates.shape


def test_joint_loss_masks_missing_labels_and_terminal_slots_are_shared():
    torch.manual_seed(3)
    event, coordinates, node, mask = inputs()
    projector = ConservationProjector(torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64))
    model = JointEventGeometryFlow(projector, 4, hidden_dim=8, radial_features=4).double()
    output = model(event, coordinates, node, mask)
    losses = joint_flow_loss(output, torch.zeros_like(output.event_velocity), torch.zeros_like(output.geometry_velocity), event_mask=torch.tensor([[True, False, True]]), atom_mask=mask)
    assert losses["total"].item() >= 0
    candidates = [
        TerminalCandidate(((0.0, 0.0, 0.0),), "event-a", score=0.5),
        TerminalCandidate(((0.00001, 0.0, 0.0),), "event-b", score=0.2),
        TerminalCandidate(((2.0, 0.0, 0.0),), "event-c", score=0.3),
    ]
    selected = deduplicate_terminal_candidates(candidates, max_slots=2, tolerance=1e-3)
    assert [item.event_key for item in selected] == ["event-b", "event-c"]


def test_control_manifest_keeps_equal_budget_statement():
    manifest = coupled_control_manifest()
    assert manifest["same_generation_weights_for_controls"] is True
    assert manifest["same_physical_budget_for_controls"] is True
    with pytest.raises(ValueError):
        coupled_control_manifest(modes=("unknown",))

