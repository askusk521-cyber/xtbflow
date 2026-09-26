"""Finite-difference curvature helpers using only declared calculator calls."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from xtbflow.calculators import CalculatorBackend, MolecularSystem


@dataclass(frozen=True)
class HVPResult:
    vector: tuple[tuple[float, float, float], ...]
    hessian_vector_product: tuple[tuple[float, float, float], ...] | None
    calculator_calls: int
    status: str
    error: str | None = None


def finite_difference_hvp(backend: CalculatorBackend, system: MolecularSystem, vector: Sequence[Sequence[float]] | torch.Tensor, *, step: float = 1e-4) -> HVPResult:
    """Estimate ``H v = -(F(R+h v)-F(R-h v))/(2h)`` with two E/F calls."""

    if not isinstance(step, (int, float)) or not math.isfinite(float(step)) or step <= 0:
        raise ValueError("step must be finite and positive")
    direction = vector if isinstance(vector, torch.Tensor) else torch.tensor(vector, dtype=torch.float64)
    if direction.shape != (len(system.symbols), 3) or not direction.is_floating_point() or not bool(torch.isfinite(direction).all()):
        raise ValueError("vector must be finite [N,3]")
    direction = direction.to(torch.float64)
    norm = direction.norm()
    if float(norm) == 0:
        raise ValueError("vector must be nonzero")
    direction = direction / norm
    coordinates = torch.tensor(system.coordinates, dtype=torch.float64)
    plus = system.with_coordinates((coordinates + float(step) * direction).tolist())
    minus = system.with_coordinates((coordinates - float(step) * direction).tolist())
    try:
        plus_result = backend.evaluate(plus, operation="energy_forces")
    except Exception as exc:
        return HVPResult(tuple(map(tuple, direction.tolist())), None, 1, "failed", f"{type(exc).__name__}: {exc}")
    try:
        minus_result = backend.evaluate(minus, operation="energy_forces")
    except Exception as exc:
        return HVPResult(tuple(map(tuple, direction.tolist())), None, plus_result.calculator_calls + 1, "failed", f"{type(exc).__name__}: {exc}")
    calls = plus_result.calculator_calls + minus_result.calculator_calls
    if plus_result.status != "success" or minus_result.status != "success" or plus_result.forces is None or minus_result.forces is None:
        error = plus_result.error_message or plus_result.error_category or minus_result.error_message or minus_result.error_category
        return HVPResult(tuple(map(tuple, direction.tolist())), None, calls, "failed", error)
    force_plus = torch.tensor(plus_result.forces, dtype=torch.float64)
    force_minus = torch.tensor(minus_result.forces, dtype=torch.float64)
    hvp = -(force_plus - force_minus) / (2.0 * float(step))
    if not bool(torch.isfinite(hvp).all()):
        return HVPResult(tuple(map(tuple, direction.tolist())), None, calls, "failed", "nonfinite finite-difference curvature")
    return HVPResult(tuple(map(tuple, direction.tolist())), tuple(map(tuple, hvp.tolist())), calls, "success")
