from __future__ import annotations

import pytest
import torch

from xtbflow.models import DualClockJointFlow, pack_be, total_electron_projector
from xtbflow.sampling import DualClockSchedule, dual_clock_step, dual_clock_trajectory


def setup():
    dtype = torch.float64
    projector = total_electron_projector(("C", "O", "H"), dtype=dtype)
    model = DualClockJointFlow(projector, 4, hidden_dim=8, radial_features=4).to(dtype=dtype)
    event = pack_be(torch.tensor([[[4., .2, .1], [.2, 6., .3], [.1, .3, 1.]]], dtype=dtype))
    coords = torch.tensor([[[0., 0., 0.], [.8, .1, 0.], [-.2, .7, .1]]], dtype=dtype)
    nodes = torch.randn(1, 3, 4, dtype=dtype)
    mask = torch.ones(1, 3, dtype=torch.bool)
    return model, event, coords, nodes, mask


def test_schedule_is_monotone_and_explicitly_reports_freeze_window():
    schedule = DualClockSchedule(event_power=2, geometry_power=1, event_finish=.75, coupling_start=.2, coupling_end=.8)
    assert schedule.at(.5) == ((.5 / .75) ** 2, .5)
    grid = schedule.grid(4)
    assert grid == sorted(grid) and {.2, .75, .8}.issubset(grid)
    assert schedule.coupled(.2) and not schedule.coupled(.8)
    with pytest.raises(ValueError):
        schedule.at(1.1)


def test_synchronous_dual_clock_reduces_to_parent_field_without_local_gate():
    model, event, coords, nodes, mask = setup()
    parent = model(event, coords, nodes, mask, tau=.4)
    dual = model.forward_dual(event, coords, nodes, mask, event_tau=.4, geometry_tau=.4)
    torch.testing.assert_close(parent.event_velocity, dual.event_velocity)
    torch.testing.assert_close(parent.geometry_velocity, dual.geometry_velocity)


def test_independent_clocks_and_local_gate_change_current_state_only():
    model, event, coords, nodes, mask = setup()
    global_out = model.forward_dual(event, coords, nodes, mask, event_tau=.8, geometry_tau=.2)
    local_out = model.forward_dual(event, coords, nodes, mask, event_tau=.8, geometry_tau=.2,
                                   reactant_event=event, gate="local", radius_angstrom=1.5, event_scale=.5)
    assert not torch.allclose(global_out.event_velocity, local_out.event_velocity)
    assert local_out.pair_gate.shape == (1, 3, 3)
    assert torch.isfinite(local_out.pair_gate).all()


def test_dual_clock_step_preserves_event_constraint_and_rejects_boundary_crossing():
    model, event, coords, nodes, mask = setup()
    schedule = DualClockSchedule(event_finish=.5, coupling_start=.25, coupling_end=.9)
    new_event, new_coords, output, row = dual_clock_step(model, event, coords, nodes, mask,
        schedule=schedule, start=.25, end=.5, reactant_event=event, gate="local")
    assert new_event.shape == event.shape and new_coords.shape == coords.shape
    assert row["event_editable"] is True and row["coupling_active"] is True
    assert row["conservation_residual_max"] < 1e-7
    with pytest.raises(ValueError, match="crosses"):
        dual_clock_step(model, event, coords, nodes, mask, schedule=schedule, start=.2, end=.6)


def test_trajectory_is_bounded_and_does_not_call_calculator():
    model, event, coords, nodes, mask = setup()
    final_event, final_coords, report = dual_clock_trajectory(model, event, coords, nodes, mask,
        schedule=DualClockSchedule(event_finish=.6, coupling_start=.2, coupling_end=.9), steps=5,
        max_evaluations=20, reactant_event=event, gate="local")
    assert report["model_evaluations"] == len(report["steps"])
    assert report["calculator_calls"] == 0
    assert torch.isfinite(final_event).all() and torch.isfinite(final_coords).all()
