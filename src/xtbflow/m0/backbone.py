"""Equivariant scalar-vector trunk (PaiNN-style) with symmetric pair features."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .batching import N_ELEMENT_TYPES
from .numerics import pair_geometry

UPDATE_VEC_EPS = 1.0e-6  # 更新块向量归一化的分母下限（特征尺度，无物理单位）


def gaussian_rbf(d: Tensor, n: int, r_max: float) -> Tensor:
    centers = torch.linspace(0.0, r_max, n, device=d.device, dtype=d.dtype)
    width = r_max / (n - 1)
    return torch.exp(-(((d[..., None] - centers) / width) ** 2))


def cosine_envelope(d: Tensor, r_cut: float) -> Tensor:
    env = 0.5 * (torch.cos(math.pi * d.clamp(max=r_cut) / r_cut) + 1.0)
    return env * (d < r_cut).to(d.dtype)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, n_freq: int = 16):
        super().__init__()
        self.register_buffer("freq", torch.arange(1, n_freq + 1, dtype=torch.float32) * math.pi)
        self.mlp = nn.Sequential(nn.Linear(2 * n_freq, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: Tensor) -> Tensor:  # t: [B] -> [B,dim]
        arg = t[:, None] * self.freq[None, :]
        return self.mlp(torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1))


class PaiNNLayer(nn.Module):
    def __init__(self, fs: int, fv: int, fe: int):
        super().__init__()
        self.fs, self.fv = fs, fv
        self.norm = nn.LayerNorm(fs)
        self.time = nn.Linear(fs, fs)
        self.phi = nn.Sequential(nn.Linear(fs, fs), nn.SiLU(), nn.Linear(fs, fs + 2 * fv))
        self.filter = nn.Linear(fe, fs + 2 * fv)
        self.U = nn.Linear(fv, fv, bias=False)
        self.V = nn.Linear(fv, fv, bias=False)
        self.update = nn.Sequential(nn.Linear(fs + fv, fs), nn.SiLU(), nn.Linear(fs, fv + 2 * fs))
        self.inner_to_s = nn.Linear(fv, fs)

    def forward(self, s, v, edge, unit, env, pair_mask, atom_mask, t_emb):
        # s [B,N,fs]  v [B,N,3,fv]  edge [B,N,N,fe]  unit [B,N,N,3]  env [B,N,N]
        fs, fv = self.fs, self.fv
        h = self.norm(s) + self.time(t_emb)[:, None, :]
        w = self.filter(edge) * (env * pair_mask.to(env.dtype))[..., None]
        x = self.phi(h)[:, None, :, :] * w                    # 下标 [b, i, j]，j 是发送方
        scale = pair_mask.sum(dim=2).clamp_min(1).to(s.dtype).rsqrt()[..., None]  # [B,N,1]
        ds, dvv, dvr = torch.split(x, [fs, fv, fv], dim=-1)
        s = s + ds.sum(dim=2) * scale
        dv = (v[:, None, :, :, :] * dvv[:, :, :, None, :]).sum(dim=2) \
            + (unit[..., None] * dvr[:, :, :, None, :]).sum(dim=2)
        v = v + dv * scale[..., None]
        # 更新块的输入先做无参数归一化（2026-10-07 负责人批准的稳定化修改）。
        # 原式 a_vv·U(v) 与 a_sv·<Uv,Vv> 对 |v| 是二次的，孤立且位移大的原子
        # （训练样本 7451 的 H7，位移约 8.8 Å）会逐层双指数放大直至 float32 溢出。
        # s_hat 用不带仿射参数的 LayerNorm；v_hat 按原子除以各通道向量模平方均值的
        # 平方根，是旋转不变的标量缩放，故保持等变；被屏蔽原子 v=0 时 v_hat=0。
        # 残差流 s、v 本身不归一化，位移幅度仍经残差流传给输出头。不新增参数。
        s_hat = F.layer_norm(s, (fs,))
        v_hat = v * torch.rsqrt(v.square().sum(dim=2).mean(dim=-1) + UPDATE_VEC_EPS)[..., None, None]
        uv, vv = self.U(v_hat), self.V(v_hat)
        vv_norm = torch.sqrt(vv.square().sum(dim=2) + 1e-8)  # [B,N,fv]，在 0 处梯度安全
        a = self.update(torch.cat([s_hat, vv_norm], dim=-1))
        a_vv, a_sv, a_ss = torch.split(a, [fv, fs, fs], dim=-1)
        s = s + a_ss + a_sv * self.inner_to_s((uv * vv).sum(dim=2))
        v = v + a_vv[:, :, None, :] * uv
        m = atom_mask.to(s.dtype)
        return s * m[..., None], v * m[..., None, None]


class ReactionTrunk(nn.Module):
    def __init__(self, scalar_dim: int = 128, vector_dim: int = 32, edge_dim: int = 64,
                 n_layers: int = 6, n_rbf: int = 32, n_rbf_reactant: int = 16, r_cut: float = 10.0,
                 dual_time: bool = False):
        super().__init__()
        self.n_rbf, self.n_rbf_r, self.r_cut = n_rbf, n_rbf_reactant, r_cut
        self.embed = nn.Embedding(N_ELEMENT_TYPES, scalar_dim)
        self.node_b = nn.Linear(3, scalar_dim)
        self.time = TimeEmbedding(scalar_dim)
        self.time_x = TimeEmbedding(scalar_dim) if dual_time else None
        self.disp = nn.Linear(1, vector_dim, bias=False)
        self.edge_mlp = nn.Sequential(
            nn.Linear(n_rbf + n_rbf_reactant + 4, edge_dim), nn.SiLU(), nn.Linear(edge_dim, edge_dim)
        )
        self.layers = nn.ModuleList(PaiNNLayer(scalar_dim, vector_dim, edge_dim) for _ in range(n_layers))

    def forward(self, z, atom_mask, x_cur, x_r, b_cur, b_r, t, t_x=None) -> dict:
        n = z.shape[1]
        _, d_cur, unit, pair_mask = pair_geometry(x_cur, atom_mask)
        _, d_r, _, _ = pair_geometry(x_r, atom_mask)
        eye = torch.eye(n, dtype=torch.bool, device=z.device)[None]
        off = (~eye).to(x_cur.dtype)
        db = b_cur - b_r
        edge_in = torch.cat([
            gaussian_rbf(d_cur, self.n_rbf, self.r_cut),
            gaussian_rbf(d_r, self.n_rbf_r, self.r_cut),
            (b_cur * off)[..., None] / 3.0,
            (b_r * off)[..., None] / 3.0,
            (db * off)[..., None] / 3.0,
            ((b_r > 0.5) & ~eye).to(x_cur.dtype)[..., None],
        ], dim=-1)
        edge = self.edge_mlp(edge_in) * pair_mask[..., None].to(x_cur.dtype)
        diag = lambda m: torch.diagonal(m, dim1=1, dim2=2)
        node_b = torch.stack([diag(b_cur), diag(b_r), diag(db)], dim=-1) / 4.0
        t_emb = self.time(t)
        if self.time_x is not None:
            if t_x is None:
                raise ValueError('dual-time trunk requires an explicit geometry clock')
            t_emb = (t_emb + self.time_x(t_x)) / math.sqrt(2.0)
        elif t_x is not None:
            raise ValueError('single-time trunk cannot consume a second clock')
        mask = atom_mask.to(x_cur.dtype)
        s = (self.embed(z) + self.node_b(node_b) + t_emb[:, None, :]) * mask[..., None]
        disp = (x_cur - x_r) * mask[..., None]
        v = disp[..., None] * self.disp.weight.view(1, 1, 1, -1)  # [B,N,3,fv]
        env = cosine_envelope(d_cur, self.r_cut)
        for layer in self.layers:
            s, v = layer(s, v, edge, unit, env, pair_mask, atom_mask, t_emb)
        return {"s": s, "v": v, "edge": edge, "pair_mask": pair_mask}
