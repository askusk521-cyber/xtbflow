"""Explicit flow integrators with auditable normalized-time steps."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor


def euler_step(
    model: Any,
    event_state: Tensor,
    coordinates: Tensor,
    node_features: Tensor,
    atom_mask: Tensor,
    *,
    tau: float | Tensor,
    dt: float,
    coupling_strength: float = 1.0,
):
    """Advance one joint-flow step with an explicit positive ``dt``.

    ``tau`` is the normalized generation coordinate, not physical time.  The
    step is deliberately bounded to ``tau + dt <= 1`` so a caller cannot
    silently integrate beyond the declared flow domain.
    """

    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(float(dt)) or dt <= 0:
        raise ValueError("dt must be a finite positive number")
    if isinstance(tau, Tensor):
        if tau.numel() != 1:
            raise ValueError("euler_step requires a scalar tau")
        tau_value = float(tau.detach().item())
    elif isinstance(tau, (int, float)) and not isinstance(tau, bool):
        tau_value = float(tau)
    else:
        raise ValueError("tau must be a finite scalar")
    if not math.isfinite(tau_value) or not 0 <= tau_value <= 1 or tau_value + float(dt) > 1 + 1e-12:
        raise ValueError("tau and dt must remain within the normalized flow domain")
    output = model(event_state, coordinates, node_features, atom_mask, coupling_strength=coupling_strength, tau=tau)
    return (
        event_state + float(dt) * output.event_velocity,
        coordinates + float(dt) * output.geometry_velocity,
        output,
    )
