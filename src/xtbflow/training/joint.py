"""Masked objectives for comparable serial and joint flow controls."""
from __future__ import annotations

import torch
from torch import Tensor

from xtbflow.models.joint_flow import JointFlowOutput


def _masked_mse(predicted: Tensor, target: Tensor, mask: Tensor | None, *, name: str) -> Tensor:
    if predicted.shape != target.shape:
        raise ValueError(f"{name} prediction and target shapes must match")
    if mask is None:
        mask = torch.ones(predicted.shape[:-1] if predicted.ndim > 1 else predicted.shape, dtype=torch.bool, device=predicted.device)
    if predicted.ndim == 2:
        if mask.shape != predicted.shape:
            raise ValueError(f"{name} mask must match [B,F]")
    elif predicted.ndim == 3:
        if mask.shape != predicted.shape[:2] or mask.dtype is not torch.bool:
            raise ValueError(f"{name} mask must be boolean [B,N]")
        mask = mask[..., None].expand_as(predicted)
    else:
        raise ValueError(f"{name} tensors must be [B,F] or [B,N,3]")
    if mask.dtype is not torch.bool:
        raise ValueError(f"{name} mask must be boolean")
    if bool(mask.any()) and (not bool(torch.isfinite(predicted[mask]).all()) or not bool(torch.isfinite(target[mask]).all())):
        raise ValueError(f"observed {name} values must be finite")
    count = mask.sum().clamp_min(1)
    return torch.where(mask, (predicted - target).square(), torch.zeros_like(predicted)).sum() / count


def joint_flow_loss(output: JointFlowOutput, target_event_velocity: Tensor, target_geometry_velocity: Tensor, *, event_mask: Tensor | None = None, atom_mask: Tensor | None = None, event_weight: float = 1.0, geometry_weight: float = 1.0) -> dict[str, Tensor]:
    """Compute separately masked event and geometry losses."""

    if event_weight < 0 or geometry_weight < 0:
        raise ValueError("loss weights must be nonnegative")
    event = _masked_mse(output.event_velocity, target_event_velocity, event_mask, name="event")
    geometry = _masked_mse(output.geometry_velocity, target_geometry_velocity, atom_mask, name="geometry")
    return {"event": event, "geometry": geometry, "total": event_weight * event + geometry_weight * geometry}


def coupled_control_manifest(*, modes: tuple[str, ...] = ("serial_independent", "joint_bidirectional", "both_off"), coupling_strength: float = 1.0, guidance_strength: float = 0.0) -> dict[str, object]:
    """Return a run-ready control declaration; no mode is silently omitted."""

    allowed = {"serial_independent", "joint_bidirectional", "both_off", "terminal_optimization"}
    if not modes or any(mode not in allowed for mode in modes):
        raise ValueError("unsupported coupling control")
    if coupling_strength < 0 or guidance_strength < 0:
        raise ValueError("control strengths must be nonnegative")
    return {"schema": "xtbflow-joint-control/v1", "modes": list(modes), "coupling_strength": float(coupling_strength), "guidance_strength": float(guidance_strength), "same_generation_weights": True, "same_physical_budget": True}

