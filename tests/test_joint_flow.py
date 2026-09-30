from __future__ import annotations

import pytest
import torch

from xtbflow.evaluation import TerminalCandidate, deduplicate_terminal_candidates
from xtbflow.models import (
    JOINT_UNIDIRECTIONAL_MODE,
    JointEventGeometryFlow,
    SerialEventGeometryFlow,
    pack_be,
    total_electron_projector,
    unpack_be,
    upper_triangle_indices,
)
from xtbflow.sampling import control_euler_step, euler_step, serial_euler_step
from xtbflow.training import coupled_control_manifest, joint_flow_loss


SYMBOLS = ("C", "O", "H")


def inputs(dtype: torch.dtype = torch.float64):
    event_matrix = torch.tensor(
        [[[4.0, 0.2, 0.1], [0.2, 6.0, 0.3], [0.1, 0.3, 1.0]]],
        dtype=dtype,
    )
    event = pack_be(event_matrix)
    coordinates = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.8, 0.1, 0.0], [-0.2, 0.7, 0.1]]],
        dtype=dtype,
    )
    node = torch.randn(1, 3, 4, dtype=dtype)
    mask = torch.tensor([[True, True, True]])
    condition = torch.tensor([[0.3, -0.4]], dtype=dtype)
    return event, coordinates, node, mask, condition


def model(dtype: torch.dtype = torch.float64) -> tuple[JointEventGeometryFlow, object]:
    projector = total_electron_projector(SYMBOLS, dtype=dtype)
    flow = JointEventGeometryFlow(
        projector,
        4,
        condition_feature_dim=2,
        hidden_dim=12,
        radial_features=5,
    ).to(dtype=dtype)
    return flow, projector


def test_joint_flow_exchanges_states_and_registered_controls_share_parameters():
    torch.manual_seed(2)
    event, coordinates, node, mask, condition = inputs()
    flow, projector = model()
    joint = flow.forward_control(
        "joint_bidirectional",
        event,
        coordinates,
        node,
        mask,
        tau=0.25,
        dt=0.2,
        condition_features=condition,
    )
    off = flow.forward_control(
        "both_off",
        event,
        coordinates,
        node,
        mask,
        tau=0.25,
        dt=0.2,
        condition_features=condition,
    )
    serial = flow.forward_control(
        "serial_independent",
        event,
        coordinates,
        node,
        mask,
        tau=0.25,
        dt=0.2,
        condition_features=condition,
    )
    assert joint.event_velocity.shape == event.shape
    assert joint.geometry_velocity.shape == coordinates.shape
    assert not torch.allclose(joint.event_velocity, off.event_velocity)
    assert not torch.allclose(joint.geometry_velocity, off.geometry_velocity)
    assert torch.count_nonzero(off.event_message) == 0
    assert torch.count_nonzero(off.geometry_message) == 0
    assert torch.count_nonzero(serial.event_message) == 0
    assert torch.allclose(serial.event_velocity, off.event_velocity)
    assert not torch.allclose(serial.geometry_velocity, off.geometry_velocity)
    assert torch.allclose(
        projector.residual(joint.event_velocity),
        torch.zeros(1, 1, dtype=torch.float64),
        atol=1e-7,
    )

    serial_wrapper = SerialEventGeometryFlow(joint_model=flow)
    assert serial_wrapper.joint is flow
    assert list(serial_wrapper.parameters()) == list(flow.parameters())
    wrapped = serial_wrapper(
        event,
        coordinates,
        node,
        mask,
        tau=0.25,
        dt=0.2,
        condition_features=condition,
    )
    torch.testing.assert_close(wrapped.event_velocity, serial.event_velocity)
    torch.testing.assert_close(wrapped.geometry_velocity, serial.geometry_velocity)
    with pytest.raises(ValueError, match="explicit positive dt"):
        flow.forward_control(
            "serial_independent",
            event,
            coordinates,
            node,
            mask,
            tau=0.25,
            condition_features=condition,
        )


def test_unidirectional_control_uses_current_state_with_the_same_euler_integrator():
    torch.manual_seed(24)
    event, coordinates, node, mask, condition = inputs()
    flow, _ = model()
    bidirectional = flow.forward_control(
        "joint_bidirectional", event, coordinates, node, mask,
        tau=0.2, dt=0.25, condition_features=condition,
    )
    unidirectional = flow.forward_control(
        JOINT_UNIDIRECTIONAL_MODE, event, coordinates, node, mask,
        tau=0.2, dt=0.25, condition_features=condition,
    )
    # The event-to-geometry channel is unchanged and both controls evaluate
    # the same current state; only the geometry-to-event message is removed.
    torch.testing.assert_close(unidirectional.geometry_velocity, bidirectional.geometry_velocity)
    assert not torch.allclose(unidirectional.event_velocity, bidirectional.event_velocity)
    _, _, stepped = control_euler_step(
        flow, event, coordinates, node, mask,
        mode=JOINT_UNIDIRECTIONAL_MODE, tau=0.2, dt=0.25,
        condition_features=condition,
    )
    torch.testing.assert_close(stepped.event_velocity, unidirectional.event_velocity)
    torch.testing.assert_close(stepped.geometry_velocity, unidirectional.geometry_velocity)
    with pytest.raises(ValueError, match="joint_unidirectional requires"):
        flow.forward_control(
            JOINT_UNIDIRECTIONAL_MODE,
            event,
            coordinates,
            node,
            mask,
            tau=0.2,
            condition_features=condition,
        )


def test_joint_loss_masks_missing_labels_and_terminal_slots_are_shared():
    torch.manual_seed(3)
    event, coordinates, node, mask, condition = inputs()
    flow, _ = model()
    output = flow(event, coordinates, node, mask, tau=0.5, condition_features=condition)
    event_mask = torch.ones_like(output.event_velocity, dtype=torch.bool)
    event_mask[:, 2] = False
    losses = joint_flow_loss(
        output,
        torch.zeros_like(output.event_velocity),
        torch.zeros_like(output.geometry_velocity),
        event_mask=event_mask,
        atom_mask=mask,
    )
    assert losses["total"].item() >= 0
    candidates = [
        TerminalCandidate(((0.0, 0.0, 0.0),), "event-a", score=0.5),
        TerminalCandidate(((0.00001, 0.0, 0.0),), "event-b", score=0.2),
        TerminalCandidate(((2.0, 0.0, 0.0),), "event-c", score=0.3),
    ]
    selected = deduplicate_terminal_candidates(candidates, max_slots=2, tolerance=1e-3)
    assert [item.event_key for item in selected] == ["event-b", "event-c"]


def test_control_manifest_distinguishes_declarations_from_measured_identity():
    torch.manual_seed(5)
    flow, _ = model()
    modes = ("both_off", "serial_independent", "joint_bidirectional")
    declaration = coupled_control_manifest(modes=modes)
    assert declaration["same_generation_weights_for_controls"] is None
    assert declaration["same_physical_budget_for_controls"] is None
    measured = coupled_control_manifest(
        modes=modes,
        models={mode: flow for mode in modes},
        physical_call_budgets={mode: 0 for mode in modes},
    )
    assert measured["same_generation_weights_for_controls"] is True
    assert measured["same_physical_budget_for_controls"] is True
    assert {identity["trainable_parameters"] for identity in measured["model_identities"].values()} == {
        sum(parameter.numel() for parameter in flow.parameters() if parameter.requires_grad)
    }

    changed, _ = model()
    with torch.no_grad():
        next(changed.parameters()).add_(1.0)
    unequal = coupled_control_manifest(
        modes=modes,
        models={"both_off": flow, "serial_independent": flow, "joint_bidirectional": changed},
        physical_call_budgets={"both_off": 0, "serial_independent": 1, "joint_bidirectional": 0},
    )
    assert unequal["same_generation_weights_for_controls"] is False
    assert unequal["same_physical_budget_for_controls"] is False


def test_joint_coordinate_and_parameter_gradients_stay_finite_at_collision_pairs():
    torch.manual_seed(9)
    event, coordinates, node, mask, condition = inputs()
    coordinates = coordinates.detach().requires_grad_(True)
    coordinates.data[:, 1] = coordinates.data[:, 0]
    flow, _ = model()
    output = flow(event, coordinates, node, mask, tau=0.4, condition_features=condition)
    loss = output.geometry_velocity.square().sum() + output.event_velocity.square().sum()
    loss.backward()
    assert torch.isfinite(coordinates.grad).all()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in flow.parameters())


def test_time_and_reactant_condition_are_explicit_and_event_velocity_is_conserved():
    torch.manual_seed(12)
    event, coordinates, node, mask, condition = inputs()
    flow, projector = model()
    outputs = [
        flow(event, coordinates, node, mask, tau=tau, condition_features=condition)
        for tau in (0.1, 0.5, 0.9)
    ]
    assert all(
        torch.isfinite(output.event_velocity).all()
        and torch.isfinite(output.geometry_velocity).all()
        for output in outputs
    )
    assert max(
        (outputs[0].event_velocity - output.event_velocity).abs().max().item()
        for output in outputs[1:]
    ) > 0
    changed_condition = flow(
        event,
        coordinates,
        node,
        mask,
        tau=0.5,
        condition_features=condition + 0.7,
    )
    assert not torch.allclose(changed_condition.event_velocity, outputs[1].event_velocity)
    for output in outputs:
        torch.testing.assert_close(
            projector.residual(output.event_velocity),
            torch.zeros(1, 1, dtype=torch.float64),
            atol=1e-7,
            rtol=0,
        )


def test_joint_and_serial_integrators_use_declared_control_paths():
    torch.manual_seed(13)
    event, coordinates, node, mask, condition = inputs()
    flow, _ = model()
    serial = SerialEventGeometryFlow(joint_model=flow)
    _, _, joint_output = euler_step(
        flow,
        event,
        coordinates,
        node,
        mask,
        tau=0.0,
        dt=0.25,
        coupling_strength=1.0,
        condition_features=condition,
    )
    _, _, off_output = control_euler_step(
        flow,
        event,
        coordinates,
        node,
        mask,
        mode="both_off",
        tau=0.0,
        dt=0.25,
        condition_features=condition,
    )
    _, _, serial_output = serial_euler_step(
        serial,
        event,
        coordinates,
        node,
        mask,
        tau=0.0,
        dt=0.25,
        condition_features=condition,
    )
    assert joint_output.event_velocity.shape == serial_output.event_velocity.shape
    assert joint_output.geometry_velocity.shape == off_output.geometry_velocity.shape
    with pytest.raises(ValueError):
        serial_euler_step(
            serial,
            event,
            coordinates,
            node,
            mask,
            tau=0.9,
            dt=0.2,
            condition_features=condition,
        )


def test_joint_loss_masks_nan_unobserved_labels_before_arithmetic():
    torch.manual_seed(10)
    event, coordinates, node, mask, condition = inputs()
    flow, _ = model()
    output = flow(event, coordinates, node, mask, tau=0.4, condition_features=condition)
    target_event = torch.zeros_like(output.event_velocity)
    target_event[:, 1] = float("nan")
    event_mask = torch.ones_like(target_event, dtype=torch.bool)
    event_mask[:, 1] = False
    losses = joint_flow_loss(
        output,
        target_event,
        torch.zeros_like(output.geometry_velocity),
        event_mask=event_mask,
        atom_mask=mask,
    )
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in flow.parameters())


def test_padding_keeps_inactive_event_features_and_coordinate_gradients_zero_or_finite():
    torch.manual_seed(14)
    dtype = torch.float64
    event_matrix = torch.zeros((1, 3, 3), dtype=dtype)
    event_matrix[:, 0, 0] = 4.0
    event = pack_be(event_matrix)
    coordinates = torch.zeros((1, 3, 3), dtype=dtype, requires_grad=True)
    coordinates.data[0, 0] = torch.tensor([0.4, -0.2, 0.1], dtype=dtype)
    node = torch.randn((1, 3, 4), dtype=dtype)
    mask = torch.tensor([[True, False, False]])
    condition = torch.tensor([[0.2, 0.1]], dtype=dtype)
    flow, projector = model()
    output = flow(event, coordinates, node, mask, tau=0.3, condition_features=condition)
    losses = joint_flow_loss(
        output,
        torch.zeros_like(output.event_velocity),
        torch.zeros_like(output.geometry_velocity),
        atom_mask=mask,
    )
    losses["total"].backward()
    rows, cols = upper_triangle_indices(3)
    active = mask[:, rows] & mask[:, cols]
    assert torch.count_nonzero(output.event_velocity.masked_select(~active)) == 0
    assert torch.isfinite(coordinates.grad).all()
    torch.testing.assert_close(
        projector.residual(output.event_velocity),
        torch.zeros(1, 1, dtype=dtype),
        atol=1e-7,
        rtol=0,
    )
    invalid = event.clone()
    invalid[:, 1] = 1.0
    with pytest.raises(ValueError, match="padded event-state"):
        flow(invalid, coordinates.detach(), node, mask, tau=0.3, condition_features=condition)


def test_joint_model_is_atom_permutation_equivariant_for_event_and_geometry_states():
    torch.manual_seed(19)
    event, coordinates, node, mask, condition = inputs()
    flow, _ = model()
    output = flow(event, coordinates, node, mask, tau=0.35, condition_features=condition)
    order = torch.tensor([2, 0, 1])
    inverse = torch.argsort(order)
    event_matrix = unpack_be(event, 3)
    permuted_event = pack_be(event_matrix[:, order][:, :, order])
    permuted = flow(
        permuted_event,
        coordinates[:, order],
        node[:, order],
        mask[:, order],
        tau=0.35,
        condition_features=condition,
    )
    restored_event = unpack_be(permuted.event_velocity, 3)[:, inverse][:, :, inverse]
    restored_event_message = unpack_be(permuted.event_message, 3)[:, inverse][:, :, inverse]
    torch.testing.assert_close(restored_event, unpack_be(output.event_velocity, 3), atol=1e-9, rtol=1e-8)
    torch.testing.assert_close(
        restored_event_message,
        unpack_be(output.event_message, 3),
        atol=1e-9,
        rtol=1e-8,
    )
    torch.testing.assert_close(permuted.geometry_velocity[:, inverse], output.geometry_velocity, atol=1e-9, rtol=1e-8)
    torch.testing.assert_close(permuted.geometry_message[:, inverse], output.geometry_message, atol=1e-9, rtol=1e-8)


def test_control_paths_expose_bidirectional_local_state_dependence():
    torch.manual_seed(23)
    event, coordinates, node, mask, condition = inputs()
    flow, _ = model()
    moved = coordinates.clone()
    moved[:, 2, 0] += 0.5
    joint = flow(event, coordinates, node, mask, tau=0.4, condition_features=condition)
    joint_moved = flow(event, moved, node, mask, tau=0.4, condition_features=condition)
    off = flow.forward_control(
        "both_off", event, coordinates, node, mask, tau=0.4, dt=0.2, condition_features=condition
    )
    off_moved = flow.forward_control(
        "both_off", event, moved, node, mask, tau=0.4, dt=0.2, condition_features=condition
    )
    assert not torch.allclose(joint.event_velocity, joint_moved.event_velocity)
    torch.testing.assert_close(off.event_velocity, off_moved.event_velocity)

    changed_event = event.clone()
    changed_event[:, 1] += 0.4
    joint_changed = flow(changed_event, coordinates, node, mask, tau=0.4, condition_features=condition)
    off_changed = flow.forward_control(
        "both_off", changed_event, coordinates, node, mask, tau=0.4, dt=0.2, condition_features=condition
    )
    assert not torch.allclose(joint.geometry_velocity, joint_changed.geometry_velocity)
    torch.testing.assert_close(off.geometry_velocity, off_changed.geometry_velocity)


def test_unconstrained_control_only_disables_event_projection():
    torch.manual_seed(29)
    event, coordinates, node, mask, condition = inputs()
    flow, projector = model()
    constrained = flow.forward_control(
        "joint_bidirectional",
        event,
        coordinates,
        node,
        mask,
        tau=0.4,
        condition_features=condition,
        conservation_projection=True,
    )
    unconstrained = flow.forward_control(
        "joint_bidirectional",
        event,
        coordinates,
        node,
        mask,
        tau=0.4,
        condition_features=condition,
        conservation_projection=False,
    )
    torch.testing.assert_close(unconstrained.geometry_velocity, constrained.geometry_velocity)
    torch.testing.assert_close(projector.residual(constrained.event_velocity), torch.zeros_like(projector.residual(constrained.event_velocity)), atol=1e-7, rtol=0)
    assert torch.linalg.vector_norm(projector.residual(unconstrained.event_velocity)) > 1e-8
