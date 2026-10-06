"""NaN-safe pair geometry shared by every M0 module."""
from __future__ import annotations

import torch
from torch import Tensor

EPS_DIST2 = 1.0e-8  # Å^2，距离下限约 1e-4 Å


def pair_mask_from_atoms(atom_mask: Tensor) -> Tensor:
    """[B,N] bool -> [B,N,N] bool, True 表示 i != j 且两者都是真实原子。"""
    n = atom_mask.shape[1]
    eye = torch.eye(n, dtype=torch.bool, device=atom_mask.device)[None]
    return atom_mask[:, :, None] & atom_mask[:, None, :] & ~eye


def pair_geometry(x: Tensor, atom_mask: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """x: [B,N,3] float; atom_mask: [B,N] bool.

    Returns (rel, dist, unit, pair_mask):
      rel[b,i,j]  = x[b,j] - x[b,i]                 [B,N,N,3]
      dist[b,i,j] = |rel|，被屏蔽的位置为 0          [B,N,N]
      unit[b,i,j] = rel/|rel|，被屏蔽的位置为 0       [B,N,N,3]
      pair_mask                                      [B,N,N] bool
    """
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError("x must be [B,N,3]")
    pair_mask = pair_mask_from_atoms(atom_mask)
    rel = x[:, None, :, :] - x[:, :, None, :]
    d2 = rel.square().sum(dim=-1)
    d2 = torch.where(pair_mask, d2, torch.ones_like(d2))  # 先替换，再开方
    dist = torch.sqrt(d2 + EPS_DIST2)
    unit = rel / dist[..., None]
    zero = torch.zeros_like(dist)
    dist = torch.where(pair_mask, dist, zero)
    unit = torch.where(pair_mask[..., None], unit, torch.zeros_like(unit))
    return rel, dist, unit, pair_mask


def masked_mean(x: Tensor, mask: Tensor, dim: int) -> Tensor:
    """对 mask 为 True 的元素求平均；mask 会广播到 x 的形状。"""
    m = mask.to(x.dtype)
    while m.ndim < x.ndim:
        m = m[..., None]
    return (x * m).sum(dim=dim) / m.sum(dim=dim).clamp_min(1.0)


def remove_masked_com(x: Tensor, atom_mask: Tensor) -> Tensor:
    """减去真实原子的质心（几何中心），填充原子保持为 0。x: [B,N,3]。"""
    com = masked_mean(x, atom_mask, dim=1)  # [B,3]
    return (x - com[:, None, :]) * atom_mask[..., None].to(x.dtype)
