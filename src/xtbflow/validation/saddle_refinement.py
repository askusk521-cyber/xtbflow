"""Bounded, fail-closed saddle refinement for the xTB validation boundary."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from xtbflow.calculators import CalculatorBackend, CalculatorError, MolecularSystem
from xtbflow.runtime import BudgetExceeded, CalculatorCallToken


def _coords(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1:] != (3,) or not np.isfinite(array).all():
        raise ValueError("coordinates must be finite [N,3]")
    return array.copy()


def _basis(coordinates: np.ndarray) -> np.ndarray:
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    columns = []
    for axis in np.eye(3):
        columns.append(np.tile(axis, (len(coordinates), 1)).reshape(-1))
    for axis in np.eye(3):
        columns.append(np.cross(np.tile(axis, (len(coordinates), 1)), centered).reshape(-1))
    left, singular, _ = np.linalg.svd(np.stack(columns, axis=1), full_matrices=False)
    rank = int(np.count_nonzero(singular > 1.0e-8))
    return left[:, :rank]


def _evaluate(backend: CalculatorBackend, system: MolecularSystem, coordinates: np.ndarray, token: CalculatorCallToken):
    result = backend.evaluate(system.with_coordinates(coordinates.tolist()), operation="energy_forces", budget_token=token)
    if result.status != "success" or result.forces is None or result.energy is None:
        raise CalculatorError(result.error_message or f"calculator returned {result.status}")
    return float(result.energy), _coords(result.forces), result


def _max_force(forces: np.ndarray) -> float:
    return float(np.max(np.abs(forces)))


@dataclass(frozen=True)
class SaddleRefinementConfig:
    hessian_step_angstrom: float = 0.003
    gradient_tolerance: float = 1.0e-4
    negative_mode_threshold: float = 1.0e-4
    trust_radius_angstrom: float = 0.03
    max_steps: int = 2
    max_calls: int = 256
    rigid_body_tolerance: float = 1.0e-8
    strategy: str = "newton"

    def __post_init__(self) -> None:
        for name in (
            "hessian_step_angstrom",
            "gradient_tolerance",
            "negative_mode_threshold",
            "trust_radius_angstrom",
            "rigid_body_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if type(self.max_calls) is not int or self.max_calls < 1:
            raise ValueError("max_calls must be a positive integer")
        if self.strategy not in {"newton", "mode_follow"}:
            raise ValueError("strategy must be 'newton' or 'mode_follow'")

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SaddleRefinementResult:
    status: str
    reason_code: str
    coordinates: tuple[tuple[float, float, float], ...]
    gradient_norm: float
    negative_mode_count: int
    calculator_calls: int
    trace: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        if self.status not in {"success", "attempted_unresolved", "budget_exhausted", "execution_failed"}:
            raise ValueError("unsupported saddle refinement status")
        if self.reason_code.strip() == "":
            raise ValueError("reason_code is required")
        if type(self.calculator_calls) is not int or self.calculator_calls < 0:
            raise ValueError("calculator_calls must be nonnegative")

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def refine_saddle(
    backend: CalculatorBackend,
    system: MolecularSystem,
    coordinates: Any,
    *,
    token: CalculatorCallToken,
    config: SaddleRefinementConfig = SaddleRefinementConfig(),
) -> SaddleRefinementResult:
    """Refine one candidate without changing thresholds or silently fallbacking."""

    if not isinstance(token, CalculatorCallToken) or not token.durable:
        raise ValueError("saddle refinement requires a durable calculator token")
    current = _coords(coordinates)
    calls_before = token.consumed_calls
    trace: list[dict[str, Any]] = []
    try:
        for iteration in range(config.max_steps + 1):
            energy, forces, _ = _evaluate(backend, system, current, token)
            gradient = _max_force(forces)
            if gradient <= config.gradient_tolerance:
                return SaddleRefinementResult(
                    "success", "STATIONARY_POINT_REACHED",
                    tuple(tuple(float(v) for v in row) for row in current),
                    gradient, 0, token.consumed_calls - calls_before, tuple(trace),
                )
            if iteration == config.max_steps:
                return SaddleRefinementResult(
                    "attempted_unresolved", "MAX_REFINEMENT_STEPS",
                    tuple(tuple(float(v) for v in row) for row in current),
                    gradient, 0, token.consumed_calls - calls_before, tuple(trace),
                )

            flat = current.reshape(-1)
            q = len(flat)
            hessian = np.zeros((q, q), dtype=float)
            for index in range(q):
                plus = flat.copy()
                minus = flat.copy()
                plus[index] += config.hessian_step_angstrom
                minus[index] -= config.hessian_step_angstrom
                _, plus_forces, _ = _evaluate(backend, system, plus.reshape(current.shape), token)
                _, minus_forces, _ = _evaluate(backend, system, minus.reshape(current.shape), token)
                hessian[:, index] = -(plus_forces.reshape(-1) - minus_forces.reshape(-1)) / (2.0 * config.hessian_step_angstrom)
            hessian = 0.5 * (hessian + hessian.T)
            rigid = _basis(current)
            projector = np.eye(q) - rigid @ rigid.T
            projected_hessian = projector @ hessian @ projector
            projected_gradient = projector @ forces.reshape(-1)
            eigenvalues, eigenvectors = np.linalg.eigh(projected_hessian)
            negative_count = int(np.count_nonzero(eigenvalues < -config.negative_mode_threshold))
            if negative_count != 1:
                return SaddleRefinementResult(
                    "attempted_unresolved", "WRONG_NEGATIVE_MODE_COUNT",
                    tuple(tuple(float(v) for v in row) for row in current),
                    gradient, negative_count, token.consumed_calls - calls_before, tuple(trace),
                )
            if config.strategy == "mode_follow":
                mode = eigenvectors[:, int(np.argmin(eigenvalues))]
                mode /= max(float(np.linalg.norm(mode)), 1.0e-12)
                positive_projector = projector - np.outer(mode, mode)
                raw_step = (positive_projector @ forces.reshape(-1)).reshape(current.shape)
                step_norm = float(np.max(np.linalg.norm(raw_step, axis=1)))
                if step_norm > config.trust_radius_angstrom:
                    raw_step *= config.trust_radius_angstrom / step_norm
                accepted = False
                trials: list[dict[str, Any]] = []
                for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
                    trial = current + fraction * raw_step
                    trial_energy, trial_forces, _ = _evaluate(backend, system, trial, token)
                    trial_gradient = _max_force(trial_forces)
                    trials.append({"fraction": fraction, "energy_hartree": trial_energy, "gradient_norm": trial_gradient})
                    if trial_gradient < gradient:
                        current = trial
                        accepted = True
                        break
                trace.append({
                    "iteration": iteration,
                    "strategy": config.strategy,
                    "energy_hartree": energy,
                    "gradient_norm": gradient,
                    "negative_mode_count": negative_count,
                    "minimum_eigenvalue": float(eigenvalues[0]),
                    "raw_step_norm_angstrom": step_norm,
                    "accepted": accepted,
                    "trials": trials,
                })
                if not accepted:
                    return SaddleRefinementResult(
                        "attempted_unresolved", "NO_TRUST_STEP",
                        tuple(tuple(float(v) for v in row) for row in current),
                        gradient, negative_count, token.consumed_calls - calls_before, tuple(trace),
                    )
                continue
            denominator = np.where(
                np.abs(eigenvalues) > 5.0e-4,
                eigenvalues,
                5.0e-4 * np.where(eigenvalues < 0.0, -1.0, 1.0),
            )
            raw_step = -(eigenvectors @ ((eigenvectors.T @ projected_gradient) / denominator))
            raw_step = projector @ raw_step
            raw_step = raw_step.reshape(current.shape)
            step_norm = float(np.max(np.linalg.norm(raw_step, axis=1)))
            if step_norm > config.trust_radius_angstrom:
                raw_step *= config.trust_radius_angstrom / step_norm

            accepted = False
            trials: list[dict[str, Any]] = []
            for fraction in (1.0, 0.5, 0.25, 0.125):
                trial = current + fraction * raw_step
                trial_energy, trial_forces, _ = _evaluate(backend, system, trial, token)
                trial_gradient = _max_force(trial_forces)
                trials.append({"fraction": fraction, "energy_hartree": trial_energy, "gradient_norm": trial_gradient})
                if trial_gradient < gradient:
                    current = trial
                    accepted = True
                    break
            trace.append({
                "iteration": iteration,
                "energy_hartree": energy,
                "gradient_norm": gradient,
                "negative_mode_count": negative_count,
                "minimum_eigenvalue": float(eigenvalues[0]),
                "raw_step_norm_angstrom": step_norm,
                "accepted": accepted,
                "trials": trials,
            })
            if not accepted:
                return SaddleRefinementResult(
                    "attempted_unresolved", "NO_TRUST_STEP",
                    tuple(tuple(float(v) for v in row) for row in current),
                    gradient, negative_count, token.consumed_calls - calls_before, tuple(trace),
                )
    except BudgetExceeded:
        return SaddleRefinementResult(
            "budget_exhausted", "BUDGET_EXCEEDED",
            tuple(tuple(float(v) for v in row) for row in current),
            _max_force(forces) if "forces" in locals() else float("inf"),
            0, token.consumed_calls - calls_before, tuple(trace),
        )
    except Exception as exc:
        return SaddleRefinementResult(
            "execution_failed", "REFINEMENT_EXECUTION",
            tuple(tuple(float(v) for v in row) for row in current),
            _max_force(forces) if "forces" in locals() else float("inf"),
            0, token.consumed_calls - calls_before, tuple(trace),
        )


__all__ = ["SaddleRefinementConfig", "SaddleRefinementResult", "refine_saddle"]
