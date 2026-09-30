"""Finite-difference curvature helpers using only declared calculator calls."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from xtbflow.calculators import CalculatorBackend, MolecularSystem
from xtbflow.runtime import BudgetExceeded, BudgetTokenRequired, CalculatorCallToken


@dataclass(frozen=True)
class HVPResult:
    vector: tuple[tuple[float, float, float], ...]
    hessian_vector_product: tuple[tuple[float, float, float], ...] | None
    calculator_calls: int
    status: str
    error: str | None = None


def finite_difference_hvp(
    backend: CalculatorBackend,
    system: MolecularSystem,
    vector: Sequence[Sequence[float]] | torch.Tensor,
    *,
    step: float = 1e-4,
    budget_token: CalculatorCallToken | None = None,
) -> HVPResult:
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
    calls_before = budget_token.consumed_calls if budget_token is not None else 0
    coordinates = torch.tensor(system.coordinates, dtype=torch.float64)
    plus = system.with_coordinates((coordinates + float(step) * direction).tolist())
    minus = system.with_coordinates((coordinates - float(step) * direction).tolist())
    try:
        if budget_token is None:
            plus_result = backend.evaluate(plus, operation="energy_forces")
        else:
            plus_result = backend.evaluate(plus, operation="energy_forces", budget_token=budget_token)
    except (BudgetExceeded, BudgetTokenRequired):
        raise
    except Exception as exc:
        calls = (budget_token.consumed_calls - calls_before) if budget_token is not None else 1
        return HVPResult(tuple(map(tuple, direction.tolist())), None, calls, "failed", f"{type(exc).__name__}: {exc}")
    try:
        if budget_token is None:
            minus_result = backend.evaluate(minus, operation="energy_forces")
        else:
            minus_result = backend.evaluate(minus, operation="energy_forces", budget_token=budget_token)
    except (BudgetExceeded, BudgetTokenRequired):
        raise
    except Exception as exc:
        calls = (budget_token.consumed_calls - calls_before) if budget_token is not None else plus_result.calculator_calls + 1
        return HVPResult(tuple(map(tuple, direction.tolist())), None, calls, "failed", f"{type(exc).__name__}: {exc}")
    calls = (
        budget_token.consumed_calls - calls_before
        if budget_token is not None
        else plus_result.calculator_calls + minus_result.calculator_calls
    )
    if plus_result.status != "success" or minus_result.status != "success" or plus_result.forces is None or minus_result.forces is None:
        error = plus_result.error_message or plus_result.error_category or minus_result.error_message or minus_result.error_category
        return HVPResult(tuple(map(tuple, direction.tolist())), None, calls, "failed", error)
    force_plus = torch.tensor(plus_result.forces, dtype=torch.float64)
    force_minus = torch.tensor(minus_result.forces, dtype=torch.float64)
    hvp = -(force_plus - force_minus) / (2.0 * float(step))
    if not bool(torch.isfinite(hvp).all()):
        return HVPResult(tuple(map(tuple, direction.tolist())), None, calls, "failed", "nonfinite finite-difference curvature")
    return HVPResult(tuple(map(tuple, direction.tolist())), tuple(map(tuple, hvp.tolist())), calls, "success")


@dataclass(frozen=True)
class ProjectedHessianResult:
    """Finite-difference Hessian represented in a declared orthonormal basis."""

    basis: tuple[tuple[tuple[float, float, float], ...], ...]
    matrix: tuple[tuple[float, ...], ...] | None
    eigenvalues: tuple[float, ...] | None
    eigenvectors: tuple[tuple[float, ...], ...] | None
    calculator_calls: int
    status: str
    error: str | None = None


def projected_hessian(
    backend: CalculatorBackend,
    system: MolecularSystem,
    basis: Sequence[Sequence[Sequence[float]]] | torch.Tensor,
    *,
    step: float = 1e-3,
    orthonormal_tolerance: float = 1e-7,
    budget_token: CalculatorCallToken | None = None,
) -> ProjectedHessianResult:
    """Evaluate ``QᵀHQ`` with two force calls per basis direction."""

    directions = basis if isinstance(basis, torch.Tensor) else torch.tensor(
        basis, dtype=torch.float64
    )
    expected_tail = (len(system.symbols), 3)
    if directions.ndim != 3 or tuple(directions.shape[1:]) != expected_tail:
        raise ValueError("basis must have shape [K,N,3] matching the system")
    directions = directions.to(torch.float64)
    if not bool(torch.isfinite(directions).all()) or directions.shape[0] < 1:
        raise ValueError("basis must be a nonempty finite tensor")
    flat = directions.reshape(directions.shape[0], -1)
    norms = flat.norm(dim=1)
    if bool((norms <= 0).any()):
        raise ValueError("basis directions must be nonzero")
    flat = flat / norms[:, None]
    gram = flat @ flat.T
    eye = torch.eye(flat.shape[0], dtype=torch.float64)
    if not torch.allclose(gram, eye, atol=orthonormal_tolerance, rtol=0.0):
        raise ValueError("basis directions must be orthonormal")
    normalized = flat.reshape_as(directions)
    calls_before = budget_token.consumed_calls if budget_token is not None else 0
    products: list[torch.Tensor] = []
    calls_without_token = 0
    for direction in normalized:
        result = finite_difference_hvp(
            backend,
            system,
            direction,
            step=step,
            budget_token=budget_token,
        )
        calls_without_token += result.calculator_calls
        if result.status != "success" or result.hessian_vector_product is None:
            calls = (
                budget_token.consumed_calls - calls_before
                if budget_token is not None
                else calls_without_token
            )
            return ProjectedHessianResult(
                tuple(tuple(map(tuple, row.tolist())) for row in normalized),
                None,
                None,
                None,
                calls,
                "failed",
                result.error or "finite-difference HVP failed",
            )
        products.append(torch.tensor(result.hessian_vector_product, dtype=torch.float64).reshape(-1))
    hvp_matrix = torch.stack(products, dim=1)
    projected = flat @ hvp_matrix
    projected = 0.5 * (projected + projected.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(projected)
    calls = (
        budget_token.consumed_calls - calls_before
        if budget_token is not None
        else calls_without_token
    )
    return ProjectedHessianResult(
        tuple(tuple(map(tuple, row.tolist())) for row in normalized),
        tuple(tuple(float(value) for value in row) for row in projected.tolist()),
        tuple(float(value) for value in eigenvalues.tolist()),
        tuple(tuple(float(value) for value in row) for row in eigenvectors.tolist()),
        calls,
        "success",
    )
