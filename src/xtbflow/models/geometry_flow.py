"""Translation/rotation/permutation equivariant coordinate flow baseline."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _mask(value: Tensor, shape: tuple[int, int], device: torch.device) -> None:
    if value.shape != shape or value.dtype is not torch.bool or value.device != device:
        raise ValueError(f"atom_mask must be boolean {shape} on the coordinate device")


class EquivariantGeometryFlow(nn.Module):
    """A bounded pairwise scalar message field with vector coordinate updates.

    Pair messages depend only on invariant distances and node/event scalars;
    multiplying each scalar by ``x_j-x_i`` makes the output transform as a
    vector under translations and proper rotations.  No reflection alignment,
    atom IDs, target coordinates, or physical force interpretation is used.
    """

    def __init__(self, node_feature_dim: int, *, hidden_dim: int = 64, radial_features: int = 16, distance_scale: float = 8.0):
        super().__init__()
        if any(type(value) is not int or value < 1 for value in (node_feature_dim, hidden_dim, radial_features)):
            raise ValueError("feature widths must be positive integers")
        if not isinstance(distance_scale, (int, float)) or not math.isfinite(distance_scale) or distance_scale <= 0:
            raise ValueError("distance_scale must be finite and positive")
        self.node_feature_dim = node_feature_dim
        self.radial_features = radial_features
        self.register_buffer("radial_centers", torch.linspace(0.0, float(distance_scale), radial_features))
        width = float(distance_scale) / max(radial_features - 1, 1)
        self.register_buffer("radial_width", torch.tensor(max(width, 1e-6)))
        self.message = nn.Sequential(
            nn.Linear(2 * node_feature_dim + radial_features + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, coordinates: Tensor, node_features: Tensor, atom_mask: Tensor, *, pair_features: Tensor | None = None) -> Tensor:
        if coordinates.ndim != 3 or coordinates.shape[-1] != 3 or not coordinates.is_floating_point():
            raise ValueError("coordinates must be floating [B,N,3]")
        b, n, _ = coordinates.shape
        if node_features.shape != (b, n, self.node_feature_dim) or node_features.dtype != coordinates.dtype or node_features.device != coordinates.device:
            raise ValueError("node_features must be [B,N,F] matching coordinates")
        _mask(atom_mask, (b, n), coordinates.device)
        if not bool(torch.isfinite(coordinates).all()) or not bool(torch.isfinite(node_features).all()):
            raise ValueError("valid geometry inputs must be finite")
        if pair_features is not None:
            if pair_features.shape != (b, n, n) or pair_features.dtype != coordinates.dtype or pair_features.device != coordinates.device:
                raise ValueError("pair_features must be [B,N,N] matching coordinates")
            if not torch.allclose(pair_features, pair_features.transpose(-1, -2)):
                raise ValueError("pair_features must be symmetric")
            if not bool(torch.isfinite(pair_features).all()):
                raise ValueError("pair_features must be finite")
        displacement = coordinates[:, :, None, :] - coordinates[:, None, :, :]
        distance = torch.sqrt(displacement.square().sum(dim=-1) + 1e-12)
        centers = self.radial_centers.to(dtype=coordinates.dtype, device=coordinates.device)
        width = self.radial_width.to(dtype=coordinates.dtype, device=coordinates.device)
        radial = torch.exp(-((distance[..., None] - centers) / width).square())
        sender = node_features[:, :, None, :].expand(-1, -1, n, -1)
        receiver = node_features[:, None, :, :].expand(-1, n, -1, -1)
        pair_scalar = torch.zeros_like(distance) if pair_features is None else pair_features
        message_input = torch.cat((sender, receiver, radial, pair_scalar[..., None]), dim=-1)
        weights = self.message(message_input).squeeze(-1)
        valid_pairs = atom_mask[:, :, None] & atom_mask[:, None, :]
        valid_pairs = valid_pairs & ~torch.eye(n, dtype=torch.bool, device=coordinates.device)[None]
        weights = torch.where(valid_pairs, weights, torch.zeros_like(weights))
        updates = (weights[..., None] * (coordinates[:, None, :, :] - coordinates[:, :, None, :])).sum(dim=2)
        return torch.where(atom_mask[..., None], updates, torch.zeros_like(updates))


class ReactantEventGeometryFlow(nn.Module):
    """Reactant/event-conditioned wrapper with separate event pooling."""

    def __init__(self, node_feature_dim: int, event_feature_dim: int, *, hidden_dim: int = 64, radial_features: int = 16):
        super().__init__()
        if type(event_feature_dim) is not int or event_feature_dim < 1:
            raise ValueError("event_feature_dim must be positive")
        self.event_feature_dim = event_feature_dim
        self.event_pool = nn.Sequential(nn.Linear(event_feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, node_feature_dim))
        self.flow = EquivariantGeometryFlow(node_feature_dim, hidden_dim=hidden_dim, radial_features=radial_features)

    def forward(self, coordinates: Tensor, node_features: Tensor, atom_mask: Tensor, event_features: Tensor, pair_edits: Tensor) -> Tensor:
        if event_features.ndim != 3 or event_features.shape[0] != coordinates.shape[0] or event_features.shape[2] != self.event_feature_dim:
            raise ValueError("event_features must be [B,E,event_feature_dim]")
        pooled = self.event_pool(event_features).mean(dim=1)
        conditioned = node_features + pooled[:, None, :]
        if pair_edits.shape != (coordinates.shape[0], event_features.shape[1], coordinates.shape[1], coordinates.shape[1]):
            raise ValueError("pair_edits must align with event and atom dimensions")
        pair = pair_edits.abs().sum(dim=1)
        return self.flow(coordinates, conditioned, atom_mask, pair_features=pair)
