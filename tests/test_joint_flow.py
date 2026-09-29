from __future__ import annotations

import pytest
import torch

from xtbflow.models import ConservationProjector, JointEventGeometryFlow, SerialEventGeometryFlow
from xtbflow.sampling import euler_step, serial_euler_step
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


def test_joint_coordinate_and_parameter_gradients_stay_finite_at_self_and_collision_pairs():
    torch.manual_seed(9)
    event, coordinates, node, mask = inputs()
    coordinates = coordinates.detach().requires_grad_(True)
    coordinates.data[:, 1] = coordinates.data[:, 0]
    projector = ConservationProjector(torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64))
    model = JointEventGeometryFlow(projector, 4, hidden_dim=8, radial_features=4).double()
    output = model(event, coordinates, node, mask)
    loss = output.geometry_velocity.square().sum() + output.event_velocity.square().sum()
    loss.backward()
    assert torch.isfinite(coordinates.grad).all()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_time_condition_is_explicit_across_multiple_tau_values_and_preserves_event_constraint():
    torch.manual_seed(12)
    event, coordinates, node, mask = inputs()
    projector = ConservationProjector(torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64))
    model = JointEventGeometryFlow(projector, 4, hidden_dim=8, radial_features=4).double()
    outputs = [model(event, coordinates, node, mask, tau=tau) for tau in (0.1, 0.5, 0.9)]
    assert all(torch.isfinite(output.event_velocity).all() and torch.isfinite(output.geometry_velocity).all() for output in outputs)
    assert max((outputs[0].event_velocity - output.event_velocity).abs().max().item() for output in outputs[1:]) > 0
    for output in outputs:
        residual = output.event_velocity @ projector.constraint_matrix.T
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-7)


def test_joint_and_serial_integrators_use_their_declared_call_contracts():
    torch.manual_seed(13)
    event, coordinates, node, mask = inputs()
    projector = ConservationProjector(torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64))
    joint = JointEventGeometryFlow(projector, 4, hidden_dim=8, radial_features=4).double()
    serial = SerialEventGeometryFlow(projector, 4, hidden_dim=8, radial_features=4).double()
    _, _, joint_output = euler_step(joint, event, coordinates, node, mask, tau=0.0, dt=0.25, coupling_strength=0.0)
    _, _, serial_output = serial_euler_step(serial, event, coordinates, node, mask, tau=0.0, dt=0.25)
    assert joint_output.event_velocity.shape == serial_output.event_velocity.shape
    assert joint_output.geometry_velocity.shape == serial_output.geometry_velocity.shape
    with pytest.raises(ValueError):
        serial_euler_step(serial, event, coordinates, node, mask, tau=0.9, dt=0.2)


def test_joint_loss_masks_nan_unobserved_labels_before_arithmetic():
    torch.manual_seed(10)
    event, coordinates, node, mask = inputs()
    projector = ConservationProjector(torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64))
    model = JointEventGeometryFlow(projector, 4, hidden_dim=8, radial_features=4).double()
    output = model(event, coordinates, node, mask)
    target_event = torch.zeros_like(output.event_velocity)
    target_event[:, 1] = float("nan")
    losses = joint_flow_loss(output, target_event, torch.zeros_like(output.geometry_velocity), event_mask=torch.tensor([[True, False, True]]), atom_mask=mask)
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
