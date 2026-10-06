import math

import torch

from xtbflow.m0.numerics import pair_geometry, remove_masked_com


def _rotation(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_gradients_finite_with_coincident_and_padded_atoms():
    x = torch.randn(2, 6, 3, dtype=torch.float64)
    x[0, 1] = x[0, 0]            # 两个真实原子重合
    x[1, 4:] = 0.0               # 填充原子坐标为 0
    x.requires_grad_(True)
    mask = torch.tensor([[True] * 6, [True] * 4 + [False] * 2])
    _, dist, unit, pm = pair_geometry(x, mask)
    rbf = torch.exp(-((dist[..., None] - torch.linspace(0, 6, 16, dtype=x.dtype)) ** 2))
    loss = dist.sum() + unit.sum() + (rbf * pm[..., None]).sum()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert torch.all(x.grad[1, 4:] == 0)


def test_gradcheck_generic_positions():
    x = torch.randn(1, 5, 3, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(1, 5, dtype=torch.bool)
    f = lambda y: pair_geometry(y, mask)[1].sum() + pair_geometry(y, mask)[2].square().sum()
    assert torch.autograd.gradcheck(f, (x,))


def test_distances_invariant_and_units_equivariant_under_rotation():
    x = torch.randn(1, 7, 3, dtype=torch.float64)
    mask = torch.ones(1, 7, dtype=torch.bool)
    r = _rotation(0)
    _, d1, u1, _ = pair_geometry(x, mask)
    _, d2, u2, _ = pair_geometry(x @ r.T, mask)
    assert torch.allclose(d1, d2, atol=1e-10)
    assert torch.allclose(u1 @ r.T, u2, atol=1e-10)


def test_remove_com_keeps_padding_zero():
    x = torch.randn(1, 5, 3)
    mask = torch.tensor([[True, True, True, False, False]])
    y = remove_masked_com(x, mask)
    assert torch.allclose(y[0, :3].mean(0), torch.zeros(3), atol=1e-6)
    assert torch.all(y[0, 3:] == 0)
