"""Auditable gradient and clipping controls for event/geometry training.

The joint loss can couple the two prediction heads through shared encoders.
This module records the objective-specific norms before clipping and exposes a
branchwise clipping control.  It deliberately does not decide which clipping
policy is scientifically preferable; a run must record the selected policy
and the resulting diagnostics.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping

import torch
from torch import Tensor, nn


_EVENT_ONLY_PREFIXES = ("event_base.", "geometry_to_event.")
_GEOMETRY_ONLY_PREFIXES = ("event_to_node.", "geometry.")
CLIP_MODES = ("global", "branchwise")


def _trainable_named_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]


def parameter_groups(model: nn.Module) -> dict[str, tuple[nn.Parameter, ...]]:
    """Partition trainable parameters into event, geometry, and shared groups.

    The current graph has explicit event-only and geometry-only submodules; all
    encoders and future unclassified modules are kept in ``shared`` so that a
    new architecture cannot silently disappear from the accounting.
    """

    groups: dict[str, list[nn.Parameter]] = {"event_only": [], "geometry_only": [], "shared": []}
    seen: set[int] = set()
    for name, parameter in _trainable_named_parameters(model):
        if id(parameter) in seen:
            raise ValueError(f"parameter appears more than once: {name}")
        seen.add(id(parameter))
        if name.startswith(_EVENT_ONLY_PREFIXES):
            group = "event_only"
        elif name.startswith(_GEOMETRY_ONLY_PREFIXES):
            group = "geometry_only"
        else:
            group = "shared"
        groups[group].append(parameter)
    return {name: tuple(values) for name, values in groups.items()}


def _finite_scalar(value: Tensor, name: str) -> None:
    if value.ndim != 0 or not value.requires_grad:
        raise ValueError(f"{name} must be a scalar tensor that requires gradients")
    if not bool(torch.isfinite(value.detach())):
        raise ValueError(f"{name} must be finite")


def _norm(values: Iterable[Tensor | None]) -> dict[str, float | int | bool]:
    squared = 0.0
    max_abs = 0.0
    tensor_count = 0
    nonzero_tensor_count = 0
    for value in values:
        if value is None:
            continue
        if not bool(torch.isfinite(value).all()):
            raise ValueError("gradient values must be finite")
        tensor_count += 1
        if bool(torch.count_nonzero(value).item()):
            nonzero_tensor_count += 1
        squared += float(value.detach().double().square().sum().item())
        if value.numel():
            max_abs = max(max_abs, float(value.detach().abs().max().item()))
    l2 = math.sqrt(squared)
    return {
        "l2_norm": l2,
        "max_abs": max_abs,
        "tensor_count": tensor_count,
        "nonzero_tensor_count": nonzero_tensor_count,
    }


def _clip_summary(norm: float, max_norm: float) -> dict[str, float | bool]:
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("max_norm must be finite and positive")
    triggered = norm > max_norm
    return {
        "max_norm": float(max_norm),
        "pre_clip_norm": float(norm),
        "clip_scale": float(max_norm / norm) if triggered else 1.0,
        "triggered": triggered,
    }


def _group_norms(
    groups: Mapping[str, tuple[nn.Parameter, ...]],
    gradients: tuple[Tensor | None, ...],
    parameters: tuple[nn.Parameter, ...],
) -> dict[str, dict[str, float | int | bool]]:
    by_parameter = {id(parameter): gradient for parameter, gradient in zip(parameters, gradients)}
    return {
        group_name: _norm(by_parameter.get(id(parameter)) for parameter in group_parameters)
        for group_name, group_parameters in groups.items()
    }


def gradient_diagnostics(
    model: nn.Module,
    event_loss: Tensor,
    geometry_loss: Tensor,
    *,
    total_loss: Tensor | None = None,
    max_norm: float = 5.0,
) -> dict[str, object]:
    """Measure objective and group norms before ``backward`` or clipping.

    ``torch.autograd.grad`` is used with ``retain_graph=True`` so callers may
    subsequently run their ordinary ``loss.backward()``.  The returned
    ``global_clip`` is the coefficient a global norm clip would apply; no
    gradients are modified by this function.
    """

    _finite_scalar(event_loss, "event_loss")
    _finite_scalar(geometry_loss, "geometry_loss")
    if total_loss is None:
        total_loss = event_loss + geometry_loss
    _finite_scalar(total_loss, "total_loss")
    parameters = tuple(parameter for _, parameter in _trainable_named_parameters(model))
    groups = parameter_groups(model)

    def grads(loss: Tensor) -> tuple[Tensor | None, ...]:
        if not parameters:
            return ()
        return tuple(torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True))

    event_grads = grads(event_loss)
    geometry_grads = grads(geometry_loss)
    total_grads = grads(total_loss)
    total_norm = _norm(total_grads)
    result: dict[str, object] = {
        "schema": "xtbflow-joint-gradient-diagnostics/v1",
        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
        "parameter_group_counts": {
            name: sum(parameter.numel() for parameter in values) for name, values in groups.items()
        },
        "objectives": {
            "event": _norm(event_grads),
            "geometry": _norm(geometry_grads),
            "total": total_norm,
        },
        "objective_group_norms": {
            "event": _group_norms(groups, event_grads, parameters),
            "geometry": _group_norms(groups, geometry_grads, parameters),
            "total": _group_norms(groups, total_grads, parameters),
        },
        "global_clip": _clip_summary(float(total_norm["l2_norm"]), max_norm),
    }
    return result


def _clip_parameters(parameters: Iterable[nn.Parameter], max_norm: float) -> dict[str, float | bool]:
    selected = tuple(parameter for parameter in parameters if parameter.grad is not None)
    if not selected:
        return _clip_summary(0.0, max_norm)
    pre_norm = float(torch.nn.utils.clip_grad_norm_(selected, max_norm).detach().item())
    return _clip_summary(pre_norm, max_norm)


def apply_gradient_clip(
    model: nn.Module,
    *,
    max_norm: float = 5.0,
    mode: str = "global",
) -> dict[str, object]:
    """Apply and report either global or branchwise gradient clipping.

    In ``branchwise`` mode the shared encoder parameters form their own group;
    this makes the control explicit rather than pretending shared gradients can
    be assigned to only one task.
    """

    if mode not in CLIP_MODES:
        raise ValueError(f"mode must be one of {CLIP_MODES}")
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("max_norm must be finite and positive")
    groups = parameter_groups(model)
    if mode == "global":
        return {
            "schema": "xtbflow-joint-gradient-clip/v1",
            "mode": mode,
            "max_norm": float(max_norm),
            "global": _clip_parameters(
                (parameter for values in groups.values() for parameter in values), max_norm
            ),
        }
    return {
        "schema": "xtbflow-joint-gradient-clip/v1",
        "mode": mode,
        "max_norm": float(max_norm),
        "groups": {
            name: _clip_parameters(values, max_norm) for name, values in groups.items()
        },
    }


__all__ = ["CLIP_MODES", "apply_gradient_clip", "gradient_diagnostics", "parameter_groups"]
