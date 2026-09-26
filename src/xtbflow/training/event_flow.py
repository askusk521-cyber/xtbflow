"""Loss and reporting helpers for conserved event-flow experiments."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from xtbflow.models.conserved_be import ConservationProjector
from xtbflow.models.event_decoder import DecodeBatch


def conservation_penalty(vector: torch.Tensor, projector: ConservationProjector) -> torch.Tensor:
    residual = projector.residual(vector)
    return residual.square().mean()


def flow_matching_loss(
    predicted_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    *,
    mode: str = "constructive",
    projector: ConservationProjector | None = None,
    penalty_weight: float = 1.0,
) -> torch.Tensor:
    """Compare equal-capacity flow modes while making conservation explicit."""

    if predicted_velocity.shape != target_velocity.shape:
        raise ValueError("predicted and target velocities must have the same shape")
    if mode not in {"constructive", "soft", "unconstrained"}:
        raise ValueError("mode must be constructive, soft, or unconstrained")
    if mode == "constructive":
        if projector is None:
            raise ValueError("constructive mode requires a conservation projector")
        predicted_velocity = projector.project(predicted_velocity)
    loss = F.mse_loss(predicted_velocity, target_velocity)
    if mode == "soft":
        if projector is None:
            raise ValueError("soft mode requires a conservation projector")
        if penalty_weight < 0:
            raise ValueError("penalty_weight must be nonnegative")
        loss = loss + penalty_weight * conservation_penalty(predicted_velocity, projector)
    return loss


def decode_summary(batch: DecodeBatch) -> dict[str, int | float]:
    """Return machine-readable rejection and endpoint-diversity metrics."""

    return {
        "attempted": batch.attempted,
        "accepted": batch.accepted,
        "rejected": batch.rejected,
        "rejection_rate": batch.rejection_rate,
        "unique_endpoints": batch.unique_endpoints,
    }
