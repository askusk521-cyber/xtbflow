"""Numerical-stability regressions for the M0 trunk (update-block pre-normalization).

The formal M0 runs stopped because an isolated, strongly displaced H atom
(train sample 7451, ~8.8 Å reactant-to-TS displacement) drove a double-
exponential scalar/vector blow-up in the PaiNN update block.  These tests pin
the fix at initialization on CPU; they make no claim about trained accuracy.
"""
import torch

from xtbflow.m0.model import ReactionFlowNet, count_parameters, match_cascade_width

FULL = dict(scalar_dim=128, vector_dim=32, edge_dim=64, n_layers=6, n_rbf=32, n_rbf_reactant=16, r_cut=10.0)


def isolated_atom_inputs(disp: float, same_geometry: bool = False):
    """Six real atoms + one padding slot; atom 0 is an H moved `disp` Å away."""
    torch.manual_seed(0)
    mask = torch.tensor([[True] * 6 + [False]])
    x_r = torch.randn(1, 7, 3) * mask[..., None]
    x_cur = x_r.clone()
    if not same_geometry:
        x_cur[0, 0] += torch.tensor([disp, 0.0, 0.0])
    b = torch.zeros(1, 7, 7)
    return dict(z=torch.tensor([[1, 2, 2, 4, 1, 1, 0]]), atom_mask=mask, x_r=x_r, x_cur=x_cur,
                b_r=b, b_cur=b.clone(), t=torch.tensor([0.9]))


def trunk_maxima(net, d):
    maxima = []
    hooks = [layer.register_forward_hook(lambda m, i, o: maxima.append(
        (o[0].abs().max().item(), o[1].abs().max().item()))) for layer in net.trunk.layers]
    out = net(**d)
    for h in hooks:
        h.remove()
    return maxima, out


def test_large_isolated_displacement_stays_bounded():
    # Before the fix this input reached NaN by layer 6 at 30 Å with default init.
    for disp in (8.8, 30.0, 100.0):
        torch.manual_seed(0)
        net = ReactionFlowNet("joint", **FULL)
        maxima, out = trunk_maxima(net, isolated_atom_inputs(disp))
        assert all(torch.isfinite(v).all() for v in out.values())
        s_max = [s for s, _ in maxima]
        # Scalar growth across layers must be at most additive, not multiplicative.
        assert max(s_max) < 20.0, (disp, s_max)


def test_gradients_finite_for_zero_vectors_and_padding():
    # event role: x_cur = x_r, so v starts at exactly zero; padding atom also has v = 0.
    torch.manual_seed(0)
    net = ReactionFlowNet("event", **FULL)
    d = isolated_atom_inputs(0.0, same_geometry=True)
    d["b_cur"] = d["b_cur"] + 0.1 * torch.eye(7)[None] * d["atom_mask"][..., None]
    out = net(**d)
    out["b_vel"].square().sum().backward()
    for name, p in net.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), name


def test_fix_adds_no_parameters_and_keeps_cascade_width():
    # Values recorded in PR #102 before the fix; a parameter-free change must keep them.
    assert count_parameters(ReactionFlowNet("joint", **FULL)) == 935970
    best = match_cascade_width(FULL)
    assert (best["scalar_dim"], best["vector_dim"], best["cascade_params"]) == (92, 23, 960498)
