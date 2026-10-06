"""Flow-matching states, noise, interpolation and losses for M0."""
from __future__ import annotations

import math

import torch
from torch import Tensor

from .numerics import remove_masked_com

ROLES = ("joint", "event", "geometry")


def active_pairs(atom_mask: Tensor) -> Tensor:
    """[B,N] -> [B,N,N] bool, diagonal included."""
    return atom_mask[:, :, None] & atom_mask[:, None, :]


def project_conserved(v: Tensor, atom_mask: Tensor) -> Tensor:
    """Orthogonal projection onto {sum_ij v_ij = 0} over the active block.

    Electron count of a BE matrix b is sum_ij b_ij, so b + dt*v keeps it
    exactly.  Symmetry is preserved; padded entries stay 0.
    """
    m = active_pairs(atom_mask).to(v.dtype)
    v = v * m
    count = m.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
    return (v - v.sum(dim=(1, 2), keepdim=True) / count) * m


def symmetric_noise(atom_mask: Tensor, generator: torch.Generator | None = None) -> Tensor:
    b, n = atom_mask.shape
    g = torch.randn(b, n, n, generator=generator, device=atom_mask.device)
    eps = (g + g.transpose(1, 2)) / math.sqrt(2.0)
    eye = torch.eye(n, dtype=torch.bool, device=atom_mask.device)[None]
    eps = torch.where(eye, g, eps)  # 对角元方差也为 1
    return project_conserved(eps, atom_mask)


def geometry_noise(atom_mask: Tensor, generator: torch.Generator | None = None) -> Tensor:
    b, n = atom_mask.shape
    eps = torch.randn(b, n, 3, generator=generator, device=atom_mask.device)
    return remove_masked_com(eps, atom_mask)


def interpolate(a0: Tensor, a1: Tensor, t: Tensor) -> Tensor:
    tt = t.view((-1,) + (1,) * (a0.ndim - 1))
    return (1.0 - tt) * a0 + tt * a1


def training_inputs(role: str, batch: dict, sigma_b: float, sigma_x: float,
                    generator: torch.Generator | None = None) -> tuple[dict, dict]:
    """Model inputs and velocity targets for one training step.

    joint:    b_cur = b_t, x_cur = x_t, shared t          -> b1-b0 and x1-x0
    event:    b_cur = b_t, x_cur = x_r (no geometry state) -> b1-b0
    geometry: b_cur = b_p (clean label), x_cur = x_t       -> x1-x0
    """
    mask = batch["atom_mask"]
    t = torch.rand(mask.shape[0], generator=generator, device=mask.device)
    b0 = batch["b_r"] + sigma_b * symmetric_noise(mask, generator)
    x0 = batch["x_r"] + sigma_x * geometry_noise(mask, generator)
    b1, x1 = batch["b_p"], batch["x_ts"]
    inputs = {"z": batch["z"], "atom_mask": mask, "x_r": batch["x_r"], "b_r": batch["b_r"], "t": t}
    if role == "joint":
        inputs.update(b_cur=interpolate(b0, b1, t), x_cur=interpolate(x0, x1, t))
        targets = {"b_vel": b1 - b0, "x_vel": x1 - x0}
    elif role == "event":
        inputs.update(b_cur=interpolate(b0, b1, t), x_cur=batch["x_r"])
        targets = {"b_vel": b1 - b0}
    elif role == "geometry":
        inputs.update(b_cur=b1, x_cur=interpolate(x0, x1, t))
        targets = {"x_vel": x1 - x0}
    else:
        raise ValueError(f"unknown role {role!r}")
    return inputs, targets


def event_loss(pred: Tensor, target: Tensor, atom_mask: Tensor) -> Tensor:
    m = active_pairs(atom_mask).to(pred.dtype)
    return ((pred - target).square() * m).sum() / m.sum().clamp_min(1.0)


def geometry_loss(pred: Tensor, target: Tensor, atom_mask: Tensor) -> Tensor:
    m = atom_mask.to(pred.dtype)[..., None]
    return ((pred - target).square() * m).sum() / (3.0 * m.sum()).clamp_min(1.0)


def flow_loss(out: dict, targets: dict, atom_mask: Tensor, geometry_weight: float = 1.0) -> dict:
    losses: dict[str, Tensor] = {}
    total = 0.0
    if "b_vel" in targets:
        losses["event"] = event_loss(out["b_vel"], targets["b_vel"], atom_mask)
        total = total + losses["event"]
    if "x_vel" in targets:
        losses["geometry"] = geometry_loss(out["x_vel"], targets["x_vel"], atom_mask)
        total = total + geometry_weight * losses["geometry"]
    losses["total"] = total
    return losses
