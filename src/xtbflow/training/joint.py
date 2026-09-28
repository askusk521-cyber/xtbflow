"""Masked objectives for comparable serial and joint flow controls."""
from __future__ import annotations

import torch
from torch import Tensor

from xtbflow.models.joint_flow import JointFlowOutput

from .checkpoint import load_joint_checkpoint, save_joint_checkpoint

__all__ = ["joint_flow_loss", "coupled_control_manifest", "save_joint_checkpoint", "load_joint_checkpoint"]


def _masked_mse(predicted: Tensor, target: Tensor, mask: Tensor | None, *, name: str) -> Tensor:
    if predicted.shape != target.shape:
        raise ValueError(f"{name} prediction and target shapes must match")
    if mask is None:
        if predicted.ndim == 2:
            mask = torch.ones_like(predicted, dtype=torch.bool)
        elif predicted.ndim == 3:
            mask = torch.ones(predicted.shape[:2], dtype=torch.bool, device=predicted.device)
        else:
            raise ValueError(f"{name} tensors must be [B,F] or [B,N,3]")
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
    # Mask before arithmetic.  An unobserved NaN target must not enter a
    # subtraction and later be hidden by ``where``; doing so can still poison
    # autograd in some backends even when the selected loss is finite.
    safe_predicted = torch.where(mask, predicted, torch.zeros_like(predicted))
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    squared_error = (safe_predicted - safe_target).square()
    return torch.where(mask, squared_error, torch.zeros_like(squared_error)).sum() / count


def joint_flow_loss(output: JointFlowOutput, target_event_velocity: Tensor, target_geometry_velocity: Tensor, *, event_mask: Tensor | None = None, atom_mask: Tensor | None = None, event_weight: float = 1.0, geometry_weight: float = 1.0, require_nonzero_target: bool = False) -> dict[str, Tensor]:
    """Compute separately masked event and geometry losses.

    ``require_nonzero_target`` is an opt-in guard for real flow-matching
    experiments; synthetic compatibility tests may continue to use zero
    targets explicitly by leaving it disabled.
    """

    if event_weight < 0 or geometry_weight < 0:
        raise ValueError("loss weights must be nonnegative")
    if require_nonzero_target:
        event_observed = target_event_velocity if event_mask is None else target_event_velocity[event_mask]
        geometry_observed = target_geometry_velocity if atom_mask is None else target_geometry_velocity[atom_mask]
        observed = torch.cat((event_observed.reshape(-1), geometry_observed.reshape(-1)))
        if observed.numel() == 0 or not bool(torch.isfinite(observed).all()) or not bool((observed.abs() > 0).any()):
            raise ValueError("flow targets must contain a finite nonzero observed velocity")
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
    return {"schema": "xtbflow-joint-control/v1", "modes": list(modes), "coupling_strength": float(coupling_strength), "guidance_strength": float(guidance_strength), "same_generation_weights_for_controls": True, "same_physical_budget_for_controls": True}
