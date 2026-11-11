"""Shared old-state observation / post-Euler injection boundary.

Coordinates are Angstrom; both clocks are dimensionless. Model/decoder and
physical service are injected explicitly. No checkpoint or target data is loaded
here. The same functions serve software fixtures and frozen trained diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import torch
from torch import Tensor


@dataclass(frozen=True)
class FlowState:
    b: Tensor
    x: Tensor
    t_b: Tensor
    t_x: Tensor


@dataclass(frozen=True)
class Injection:
    observed: FlowState
    base: FlowState
    predicted_b: Tensor
    predicted_x: Tensor
    dt_x: float


def centered(value: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(value.dtype)[..., None]
    return (value - (value * weights).sum(1, keepdim=True) /
            weights.sum(1, keepdim=True).clamp_min(1)) * weights


def model_values(model: Any, query: Any, state: FlowState) -> tuple[Tensor, Tensor, Tensor]:
    """One counted forward; return endpoint geometry, projected BE velocity, x velocity."""
    out = model(z=query.element_index, atom_mask=query.atom_mask,
                x_cur=centered(state.x, query.atom_mask), x_r=query.x_r,
                b_cur=state.b, b_r=query.b_r, t=state.t_b, t_x=state.t_x)
    if not all(torch.isfinite(v).all() for v in out.values()):
        raise ValueError("nonfinite_model_output")
    vb, vx = out["b_vel"], out["x_vel"]
    pair = query.atom_mask[:, :, None] & query.atom_mask[:, None, :]
    if (vb.shape != state.b.shape or vx.shape != state.x.shape or
            not torch.allclose(vb, vb.transpose(1, 2), atol=1e-7, rtol=0) or
            not torch.allclose(vb.sum((1, 2)), torch.zeros_like(state.t_b), atol=1e-6, rtol=0) or
            torch.any(vb.masked_select(~pair).abs() > 1e-7)):
        raise ValueError("model_mask_or_BE_conservation_failure")
    endpoint = centered(state.x + (1-state.t_x[:, None, None])*vx, query.atom_mask)
    return endpoint, vb, vx


def prepare_injection(model: Any, query: Any, state: FlowState,
                      next_t_b: float, next_t_x: float) -> Injection:
    """Observe OLD endpoint, then perform the ordinary event and geometry Euler step."""
    if not (torch.all(state.t_b <= next_t_b) and torch.all(state.t_x <= next_t_x)
            and 0 <= next_t_b <= 1 and 0 <= next_t_x <= 1):
        raise ValueError("invalid_clock_increment")
    endpoint, vb, vx = model_values(model, query, state)
    dt_b = next_t_b-state.t_b[:, None, None]
    dt_x = next_t_x-state.t_x[:, None, None]
    base = FlowState(state.b+dt_b*vb, centered(state.x+dt_x*vx, query.atom_mask),
                     torch.full_like(state.t_b, next_t_b), torch.full_like(state.t_x, next_t_x))
    return Injection(state, base, state.b+(1-state.t_b[:, None, None])*vb,
                     endpoint, float(dt_x.flatten()[0]))


def apply_update(injection: Injection, query: Any, delta: Tensor) -> FlowState:
    """Only geometry changes here; next model call consumes updated BE and clocks."""
    if delta.shape != injection.base.x.shape or not torch.isfinite(delta).all():
        raise ValueError("invalid_displacement")
    return replace(injection.base, x=centered(injection.base.x+delta, query.atom_mask))


def guided_step(model: Any, query: Any, state: FlowState, *, next_t_b: float,
                next_t_x: float, guide: Callable | None = None, alpha: float = 0.) -> FlowState:
    injection = prepare_injection(model, query, state, next_t_b, next_t_x)
    if alpha == 0 or guide is None:
        return injection.base
    return apply_update(injection, query, guide(injection, query)*alpha)


def require_settled(upstream: dict) -> None:
    """Certification cannot cross an incomplete or unmeasured guidance boundary."""
    if (upstream.get("status") != "completed" or upstream.get("termination") != "confirmed"
            or upstream.get("settlement") != "settled" or upstream.get("reservations") != {}):
        raise ValueError("uncertified_upstream_execution_or_cost")


def candidate_from_state(query_packet: dict, state: FlowState, decoder: Callable,
                         *, candidate_id: str, protocol: dict, upstream: dict):
    from xtbflow.governance.generation_bridge import certification_candidate
    require_settled(upstream)
    decoded = decoder(state.b[0])
    if decoded is None:
        raise ValueError("invalid_final_decode")
    output = dict(query_id=query_packet["query_id"], status="generated", method="integrated_guidance",
                  proposal=0, candidate_id=candidate_id, x=state.x[0].detach().tolist(),
                  b_dec=decoded.tolist())
    return certification_candidate(query_packet, output, generation_protocol=protocol)
