"""Sign-invariant normalized reaction-direction helpers."""
from __future__ import annotations

import torch
from torch import Tensor, nn


def normalize_direction(direction: Tensor, atom_mask: Tensor, *, eps: float = 1e-8) -> Tensor:
    if direction.ndim != 3 or direction.shape[-1] != 3 or not direction.is_floating_point():
        raise ValueError("direction must be floating [B,N,3]")
    if atom_mask.shape != direction.shape[:2] or atom_mask.dtype is not torch.bool or atom_mask.device != direction.device:
        raise ValueError("atom_mask must be boolean [B,N] on direction device")
    if eps <= 0:
        raise ValueError("eps must be positive")
    masked = torch.where(atom_mask[..., None], direction, torch.zeros_like(direction))
    norm = masked.square().sum(dim=(-1, -2), keepdim=True).sqrt().clamp_min(eps)
    return masked / norm


def sign_invariant_direction_loss(predicted: Tensor, target: Tensor, atom_mask: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Use the lower MSE of ``target`` and ``-target`` without choosing a sign."""

    pred = normalize_direction(predicted, atom_mask, eps=eps)
    truth = normalize_direction(target, atom_mask, eps=eps)
    mask = atom_mask[..., None].expand_as(pred)
    direct = torch.where(mask, (pred - truth).square(), torch.zeros_like(pred)).sum(dim=(-1, -2))
    flipped = torch.where(mask, (pred + truth).square(), torch.zeros_like(pred)).sum(dim=(-1, -2))
    return torch.minimum(direct, flipped).mean()


class ReactionDirectionHead(nn.Module):
    """Predict one vector per atom from scalar node features."""

    def __init__(self, node_feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        if any(type(value) is not int or value < 1 for value in (node_feature_dim, hidden_dim)):
            raise ValueError("feature widths must be positive integers")
        self.readout = nn.Sequential(nn.Linear(node_feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3))

    def forward(self, node_features: Tensor, coordinates: Tensor, atom_mask: Tensor) -> Tensor:
        if node_features.ndim != 3 or not node_features.is_floating_point():
            raise ValueError("node_features must be floating [B,N,F]")
        if coordinates.ndim != 3 or coordinates.shape != (*node_features.shape[:2], 3) or coordinates.dtype != node_features.dtype or coordinates.device != node_features.device:
            raise ValueError("coordinates must be floating [B,N,3] matching node_features")
        if atom_mask.shape != node_features.shape[:2] or atom_mask.dtype is not torch.bool or atom_mask.device != node_features.device:
            raise ValueError("atom_mask must be boolean [B,N]")
        center = (torch.where(atom_mask[..., None], coordinates, torch.zeros_like(coordinates)).sum(dim=1, keepdim=True) /
                 atom_mask.sum(dim=1, keepdim=True).clamp_min(1)[..., None])
        gate = self.readout(node_features).mean(dim=-1, keepdim=True)
        direction = gate * (coordinates - center)
        return torch.where(atom_mask[..., None], direction, torch.zeros_like(direction))
