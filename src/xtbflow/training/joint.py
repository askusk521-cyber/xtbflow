"""Masked objectives and auditable serial/joint control manifests."""
from __future__ import annotations

import hashlib
import math
from typing import Mapping

import torch
from torch import Tensor, nn

from xtbflow.models.joint_flow import CONTROL_MODES, JointFlowOutput

from .checkpoint import load_joint_checkpoint, save_joint_checkpoint

__all__ = [
    "joint_flow_loss",
    "model_state_identity",
    "coupled_control_manifest",
    "save_joint_checkpoint",
    "load_joint_checkpoint",
]


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
    if bool(mask.any()) and (
        not bool(torch.isfinite(predicted[mask]).all())
        or not bool(torch.isfinite(target[mask]).all())
    ):
        raise ValueError(f"observed {name} values must be finite")
    count = mask.sum().clamp_min(1)
    # Mask before arithmetic so an unobserved NaN never enters autograd.
    safe_predicted = torch.where(mask, predicted, torch.zeros_like(predicted))
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    squared_error = (safe_predicted - safe_target).square()
    return torch.where(mask, squared_error, torch.zeros_like(squared_error)).sum() / count


def joint_flow_loss(
    output: JointFlowOutput,
    target_event_velocity: Tensor,
    target_geometry_velocity: Tensor,
    *,
    event_mask: Tensor | None = None,
    atom_mask: Tensor | None = None,
    event_weight: float = 1.0,
    geometry_weight: float = 1.0,
    require_nonzero_target: bool = False,
) -> dict[str, Tensor]:
    """Compute separately masked event and geometry losses."""

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
    return {
        "event": event,
        "geometry": geometry,
        "total": event_weight * event + geometry_weight * geometry,
    }


def model_state_identity(model: nn.Module) -> dict[str, object]:
    """Hash the exact model state and report real parameter counts."""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return {
        "class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "state_sha256": digest.hexdigest(),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def coupled_control_manifest(
    *,
    modes: tuple[str, ...] = CONTROL_MODES,
    models: Mapping[str, nn.Module] | None = None,
    physical_call_budgets: Mapping[str, int] | None = None,
    coupling_strength: float = 1.0,
    guidance_strength: float = 0.0,
    dataset_fingerprint: str | None = None,
    split_fingerprint: str | None = None,
    input_view_fingerprint: str | None = None,
) -> dict[str, object]:
    """Build a control manifest from measured identities, never assertions.

    Equality fields are ``None`` until concrete model objects or call budgets
    are supplied.  This prevents a configuration declaration from being
    mistaken for evidence that weights or physical costs actually matched.
    """

    allowed = set(CONTROL_MODES) | {"terminal_optimization"}
    if not modes or len(set(modes)) != len(modes) or any(mode not in allowed for mode in modes):
        raise ValueError("unsupported or duplicate coupling control")
    for value, name in ((coupling_strength, "coupling_strength"), (guidance_strength, "guidance_strength")):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")

    identities: dict[str, dict[str, object]] | None = None
    same_weights: bool | None = None
    if models is not None:
        if set(models) != set(modes) or any(not isinstance(model, nn.Module) for model in models.values()):
            raise ValueError("models must provide one torch module for every requested mode")
        identities = {mode: model_state_identity(models[mode]) for mode in modes}
        signatures = {
            (
                identity["state_sha256"],
                identity["total_parameters"],
                identity["trainable_parameters"],
            )
            for identity in identities.values()
        }
        same_weights = len(signatures) == 1

    budgets: dict[str, int] | None = None
    same_budget: bool | None = None
    if physical_call_budgets is not None:
        if set(physical_call_budgets) != set(modes):
            raise ValueError("physical_call_budgets must cover every requested mode")
        if any(type(value) is not int or value < 0 for value in physical_call_budgets.values()):
            raise ValueError("physical call budgets must be nonnegative integers")
        budgets = {mode: physical_call_budgets[mode] for mode in modes}
        same_budget = len(set(budgets.values())) == 1

    return {
        "schema": "xtbflow-joint-control/v2",
        "modes": list(modes),
        "coupling_strength": float(coupling_strength),
        "guidance_strength": float(guidance_strength),
        "dataset_fingerprint": dataset_fingerprint,
        "split_fingerprint": split_fingerprint,
        "input_view_fingerprint": input_view_fingerprint,
        "model_identity_status": "measured" if identities is not None else "unverified",
        "model_identities": identities,
        "same_generation_weights_for_controls": same_weights,
        "physical_budget_status": "measured" if budgets is not None else "unverified",
        "physical_call_budgets": budgets,
        "same_physical_budget_for_controls": same_budget,
    }
