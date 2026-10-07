"""ReactionFlowNet: one class, three roles (joint / event / geometry)."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .backbone import ReactionTrunk
from .flow import ROLES, project_conserved
from .numerics import remove_masked_com


class EventHead(nn.Module):
    def __init__(self, fs: int, fv: int, fe: int):
        super().__init__()
        self.pair = nn.Sequential(
            nn.Linear(2 * fs + fe + fv, fs), nn.SiLU(), nn.Linear(fs, fs), nn.SiLU(), nn.Linear(fs, 1)
        )
        self.diag = nn.Sequential(nn.Linear(fs, fs), nn.SiLU(), nn.Linear(fs, 1))

    def forward(self, s, v, edge, pair_mask, atom_mask) -> Tensor:
        si, sj = s[:, :, None, :], s[:, None, :, :]
        vv = (v[:, :, None, :, :] * v[:, None, :, :, :]).sum(dim=3)  # <v_i, v_j> [B,N,N,fv]
        feat = torch.cat([si + sj, si * sj, edge, vv], dim=-1)
        off = self.pair(feat).squeeze(-1)
        off = 0.5 * (off + off.transpose(1, 2)) * pair_mask.to(off.dtype)
        dg = self.diag(s).squeeze(-1) * atom_mask.to(s.dtype)
        return off + torch.diag_embed(dg)


class GeometryHead(nn.Module):
    def __init__(self, fs: int, fv: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(fs, fs), nn.SiLU(), nn.Linear(fs, fv))
        self.out = nn.Linear(fv, 1, bias=False)

    def forward(self, s, v, atom_mask) -> Tensor:
        vel = self.out(v * self.gate(s)[:, :, None, :]).squeeze(-1)  # [B,N,3]
        return remove_masked_com(vel, atom_mask)


class ReactionFlowNet(nn.Module):
    def __init__(self, role: str, scalar_dim: int = 128, vector_dim: int = 32, edge_dim: int = 64,
                 n_layers: int = 6, n_rbf: int = 32, n_rbf_reactant: int = 16, r_cut: float = 10.0,
                 dual_time: bool = False):
        super().__init__()
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        self.role = role
        if dual_time and role != 'joint':
            raise ValueError('only the joint generator uses two clocks')
        self.trunk = ReactionTrunk(scalar_dim, vector_dim, edge_dim, n_layers, n_rbf,
                                   n_rbf_reactant, r_cut, dual_time=dual_time)
        self.event_head = EventHead(scalar_dim, vector_dim, edge_dim) if role in ("joint", "event") else None
        self.geometry_head = GeometryHead(scalar_dim, vector_dim) if role in ("joint", "geometry") else None

    def forward(self, z, atom_mask, x_cur, x_r, b_cur, b_r, t, t_x=None) -> dict[str, Tensor]:
        if self.role == "event" and not torch.equal(x_cur, x_r):
            raise ValueError("event role must be called with x_cur = x_r")
        if self.role == "geometry" and not torch.equal(b_cur, torch.round(b_cur)):
            raise ValueError("geometry role needs an integer b_cur (label or decoded event)")
        h = self.trunk(z, atom_mask, x_cur, x_r, b_cur, b_r, t, t_x=t_x)
        out: dict[str, Tensor] = {}
        if self.event_head is not None:
            raw = self.event_head(h["s"], h["v"], h["edge"], h["pair_mask"], atom_mask)
            out["b_vel"] = project_conserved(raw, atom_mask)
        if self.geometry_head is not None:
            out["x_vel"] = self.geometry_head(h["s"], h["v"], atom_mask)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def match_cascade_width(model_cfg: dict, tol: float = 0.05) -> dict:
    """Pick the A-arm width so event+geometry params match the joint net within tol."""
    target = count_parameters(ReactionFlowNet("joint", **model_cfg))
    best = None
    for fs in range(32, model_cfg["scalar_dim"] + 1, 4):
        fv = max(8, fs // 4)
        cfg = dict(model_cfg, scalar_dim=fs, vector_dim=fv)
        total = count_parameters(ReactionFlowNet("event", **cfg)) + count_parameters(ReactionFlowNet("geometry", **cfg))
        rel = abs(total - target) / target
        if best is None or rel < best["rel_diff"]:
            best = {"scalar_dim": fs, "vector_dim": fv, "cascade_params": total,
                    "joint_params": target, "rel_diff": rel}
    if best["rel_diff"] > tol:
        raise RuntimeError(f"no width within {tol:.0%}: {best}")
    return best
