"""Task-separated geometry-flow losses and result metadata."""
from __future__ import annotations

import torch
from torch import Tensor

from xtbflow.data.task_views import EndpointGeometryInput, GeometryFlowResult, GeometryTarget, ReactantEventGeometryInput
from xtbflow.models.reaction_direction import sign_invariant_direction_loss


def masked_coordinate_loss(predicted: Tensor, target: Tensor, atom_mask: Tensor, observed_mask: Tensor | None = None) -> Tensor:
    if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[-1] != 3:
        raise ValueError("predicted and target coordinates must have shape [B,N,3]")
    if atom_mask.shape != predicted.shape[:2] or atom_mask.dtype is not torch.bool:
        raise ValueError("atom_mask must be boolean [B,N]")
    observed = atom_mask if observed_mask is None else atom_mask & observed_mask
    if observed.shape != atom_mask.shape or observed.dtype is not torch.bool:
        raise ValueError("observed_mask must be boolean [B,N]")
    if not bool(torch.isfinite(predicted[observed]).all()) or not bool(torch.isfinite(target[observed]).all()):
        raise ValueError("observed coordinates must be finite")
    count = observed.sum().clamp_min(1)
    return torch.where(observed[..., None], (predicted - target).square(), torch.zeros_like(predicted)).sum() / (3 * count)


def endpoint_geometry_loss(predicted: Tensor, view: EndpointGeometryInput, target: GeometryTarget) -> Tensor:
    view.validate()
    target.validate(batch=view.coordinates.shape[0], atoms=view.coordinates.shape[1], device=view.coordinates.device, dtype=view.coordinates.dtype)
    return masked_coordinate_loss(predicted, target.coordinates, view.atom_mask, target.observed_mask)


def reactant_direction_loss(predicted_direction: Tensor, target_direction: Tensor, view: ReactantEventGeometryInput) -> Tensor:
    view.validate()
    if predicted_direction.shape != view.coordinates.shape or target_direction.shape != view.coordinates.shape:
        raise ValueError("direction tensors must match the reactant geometry")
    return sign_invariant_direction_loss(predicted_direction, target_direction, view.atom_mask)


def result_metadata(view: EndpointGeometryInput | ReactantEventGeometryInput, *, generated_from: str = "model") -> tuple[str, ...]:
    """Make information access explicit in serialized result metadata."""

    view.validate()
    if isinstance(view, EndpointGeometryInput):
        return ("task_mode:endpoint_geometry", "information:declared_endpoint", f"generated_from:{generated_from}")
    return ("task_mode:reactant_event_geometry", "information:reactant_only", f"generated_from:{generated_from}")
