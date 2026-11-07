"""Bounded explicit Euler updates in each branch's own generation coordinate."""
from __future__ import annotations

from dataclasses import asdict
import math

import torch
from torch import Tensor

from .clocks import DualClockSchedule


def dual_clock_step(model, event: Tensor, coordinates: Tensor, nodes: Tensor, mask: Tensor,
                    *, schedule: DualClockSchedule, start: float, end: float,
                    geometry_to_event_strength: float = 1.0, event_to_geometry_strength: float = 1.0,
                    **kwargs):
    tb, tx = schedule.at(start)
    next_b, next_x = schedule.at(end)
    if end <= start:
        raise ValueError("strictly positive generation progress step required")
    if any(start < boundary < end for boundary in (schedule.event_finish, schedule.coupling_start, schedule.coupling_end)):
        raise ValueError("step crosses a freeze or coupling boundary; use schedule.grid")
    for value in (geometry_to_event_strength, event_to_geometry_strength):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("coupling strengths must be finite and nonnegative")
    active = float(schedule.coupled(start))
    output = model.forward_dual(event, coordinates, nodes, mask, event_tau=tb, geometry_tau=tx,
        geometry_to_event_strength=active * geometry_to_event_strength,
        event_to_geometry_strength=active * event_to_geometry_strength, **kwargs)
    result_b = event + (next_b - tb) * output.event_velocity
    result_x = coordinates + (next_x - tx) * output.geometry_velocity
    if not bool(torch.isfinite(result_b).all()) or not bool(torch.isfinite(result_x).all()):
        raise FloatingPointError("dual clock step produced a nonfinite state")
    diagnostics = dict(progress=start, next_progress=end, event_tau=tb, geometry_tau=tx,
        event_increment=next_b-tb, geometry_increment=next_x-tx,
        event_editable=next_b > tb, coupling_active=bool(active),
        event_update_norm=float(torch.linalg.vector_norm((result_b-event).detach())),
        geometry_update_norm_angstrom=float(torch.linalg.vector_norm((result_x-coordinates).detach())),
        conservation_residual_max=float(model.event_projector.residual(result_b-event).detach().abs().max()),
        event_message_norm=float(torch.linalg.vector_norm(output.event_message.detach())),
        geometry_message_norm=float(torch.linalg.vector_norm(output.geometry_message.detach())))
    return result_b, result_x, output, diagnostics


def dual_clock_trajectory(model, event: Tensor, coordinates: Tensor, nodes: Tensor, mask: Tensor,
                          *, schedule: DualClockSchedule, steps: int, max_evaluations: int, **kwargs):
    grid = schedule.grid(steps)
    if type(max_evaluations) is not int or max_evaluations < len(grid)-1:
        raise ValueError("complete trajectory exceeds model evaluation budget")
    records = []
    with torch.no_grad():
        for start, end in zip(grid, grid[1:]):
            event, coordinates, _, row = dual_clock_step(model, event, coordinates, nodes, mask,
                schedule=schedule, start=start, end=end, **kwargs)
            # Full state remains in the private run. Public evidence uses hashes
            # and scalar summaries, never inferred endpoints or raw data.
            records.append(dict(**row, event=event.detach().cpu().tolist(), coordinates=coordinates.detach().cpu().tolist()))
    return event, coordinates, dict(schema="xtbflow-dual-clock-trajectory/v1", schedule=asdict(schedule),
        model_evaluations=len(records), steps=records, calculator_calls=0,
        claim="generation trajectory only; no chemical certification")
