"""Explicit flow integrators with auditable normalized-time steps."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from xtbflow.models.joint_flow import validate_normalized_step


def _validated_step(tau: float | Tensor, dt: float) -> float:
    return validate_normalized_step(tau, dt)


def control_euler_step(
    model: Any,
    event_state: Tensor,
    coordinates: Tensor,
    node_features: Tensor,
    atom_mask: Tensor,
    *,
    mode: str,
    tau: float | Tensor,
    dt: float,
    coupling_strength: float = 1.0,
    condition_features: Tensor | None = None,
    conservation_projection: bool = True,
):
    """Advance one registered shared-weight control by explicit Euler."""

    _validated_step(tau, dt)
    output = model.forward_control(
        mode,
        event_state,
        coordinates,
        node_features,
        atom_mask,
        tau=tau,
        dt=dt,
        coupling_strength=coupling_strength,
        condition_features=condition_features,
        conservation_projection=conservation_projection,
    )
    return (
        event_state + float(dt) * output.event_velocity,
        coordinates + float(dt) * output.geometry_velocity,
        output,
    )


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
    condition_features: Tensor | None = None,
    conservation_projection: bool = True,
):
    """Advance the bidirectional joint control by explicit Euler."""

    return control_euler_step(
        model,
        event_state,
        coordinates,
        node_features,
        atom_mask,
        mode="joint_bidirectional",
        tau=tau,
        dt=dt,
        coupling_strength=coupling_strength,
        condition_features=condition_features,
        conservation_projection=conservation_projection,
    )


def serial_euler_step(
    model: Any,
    event_state: Tensor,
    coordinates: Tensor,
    node_features: Tensor,
    atom_mask: Tensor,
    *,
    tau: float | Tensor,
    dt: float,
    coupling_strength: float = 1.0,
    condition_features: Tensor | None = None,
    conservation_projection: bool = True,
):
    """Advance the serial wrapper with its explicit ``dt`` argument."""

    _validated_step(tau, dt)
    output = model(
        event_state,
        coordinates,
        node_features,
        atom_mask,
        tau=tau,
        dt=dt,
        coupling_strength=coupling_strength,
        condition_features=condition_features,
        conservation_projection=conservation_projection,
    )
    return (
        event_state + float(dt) * output.event_velocity,
        coordinates + float(dt) * output.geometry_velocity,
        output,
    )
