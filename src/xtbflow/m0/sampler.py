"""Generation for both arms.  Product/TS fields are forbidden at sampling."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from xtbflow.models.conserved_be import pack_be
from xtbflow.models.event_decoder import DecodeError, decode_endpoint

from .flow import geometry_noise, symmetric_noise

REACTANT_KEYS = ("z", "atom_mask", "x_r", "b_r", "index")
FORBIDDEN = {"x_ts", "x_p", "b_p"}
INDEX_TO_SYMBOL = {1: "H", 2: "C", 3: "N", 4: "O"}


def reactant_view(batch: dict) -> dict:
    return {k: batch[k] for k in REACTANT_KEYS}


def _repeat(rv: dict, n: int) -> dict:
    if FORBIDDEN & set(rv):
        raise ValueError(f"forbidden inputs at sampling: {sorted(FORBIDDEN & set(rv))}")
    return {k: v.repeat_interleave(n, dim=0) for k, v in rv.items()}


def decode_be(b: Tensor, z_index: Tensor, n: int) -> tuple[np.ndarray | None, np.ndarray]:
    """One padded sample -> (repaired integer BE [n,n] or None, fallback integer BE [n,n])."""
    bb = b[:n, :n].detach().double().cpu()
    bb = 0.5 * (bb + bb.T)
    fallback = np.clip(np.rint(bb.numpy()), 0, 3)
    diag = np.clip(2 * np.rint(np.diag(bb.numpy()) / 2), 0, 8)
    np.fill_diagonal(fallback, diag)
    symbols = [INDEX_TO_SYMBOL[int(v)] for v in z_index[:n]]
    try:
        decoded = decode_endpoint(pack_be(bb), symbols, charge=0, multiplicity=1)
    except (DecodeError, ValueError):
        return None, fallback.astype(np.int64)
    mat = np.array(decoded.state.be, dtype=np.int64)
    return mat, mat


def decode_batch(b: Tensor, rv: dict) -> tuple[Tensor, Tensor]:
    """[S,N,N] -> integer BE tensor (fallback where decoding failed) and ok flags [S]."""
    out = torch.zeros_like(b)
    ok = torch.zeros(b.shape[0], dtype=torch.bool, device=b.device)
    n_atoms = rv["atom_mask"].sum(dim=1).tolist()
    for k, n in enumerate(n_atoms):
        mat, fallback = decode_be(b[k], rv["z"][k], int(n))
        use = mat if mat is not None else fallback
        out[k, :n, :n] = torch.from_numpy(use).to(b)
        ok[k] = mat is not None
    return out, ok


@torch.no_grad()
def sample_joint(net, rv: dict, n_samples: int, nfe: int, sigma_b: float, sigma_x: float,
                 generator: torch.Generator) -> dict:
    rv = _repeat(rv, n_samples)
    mask = rv["atom_mask"]
    b = rv["b_r"] + sigma_b * symmetric_noise(mask, generator)
    x = rv["x_r"] + sigma_x * geometry_noise(mask, generator)
    dt = 1.0 / nfe
    for k in range(nfe):
        t = torch.full((mask.shape[0],), k * dt, device=mask.device)
        out = net(rv["z"], mask, x, rv["x_r"], b, rv["b_r"], t)
        b = b + dt * out["b_vel"]
        x = x + dt * out["x_vel"]
    return {"b_raw": b, "x": x, "rv": rv}


@torch.no_grad()
def sample_cascade(event_net, geometry_net, rv: dict, n_samples: int, nfe: int,
                   sigma_b: float, sigma_x: float, generator: torch.Generator) -> dict:
    rv = _repeat(rv, n_samples)
    mask = rv["atom_mask"]
    b = rv["b_r"] + sigma_b * symmetric_noise(mask, generator)
    dt = 1.0 / nfe
    for k in range(nfe):
        t = torch.full((mask.shape[0],), k * dt, device=mask.device)
        b = b + dt * event_net(rv["z"], mask, rv["x_r"], rv["x_r"], b, rv["b_r"], t)["b_vel"]
    b_int, _ = decode_batch(b, rv)                  # 第二步只看解码后的整数事件
    x = rv["x_r"] + sigma_x * geometry_noise(mask, generator)
    for k in range(nfe):
        t = torch.full((mask.shape[0],), k * dt, device=mask.device)
        x = x + dt * geometry_net(rv["z"], mask, x, rv["x_r"], b_int, rv["b_r"], t)["x_vel"]
    return {"b_raw": b, "x": x, "rv": rv}
