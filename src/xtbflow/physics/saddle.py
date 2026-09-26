"""Physical-force guidance and common post-processing controls."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

import torch

from xtbflow.calculators import CalculatorBackend, CalculationResult, MolecularSystem


class GuidanceMode(str, Enum):
    NONE = "none"
    ENERGY_DESCENT = "energy_descent"
    SADDLE = "saddle"


class FailurePolicy(str, Enum):
    REJECT = "reject"
    SKIP = "skip"
    STOP = "stop"


@dataclass(frozen=True)
class GuidanceConfig:
    """Finite guidance schedule; strength is applied exactly once."""

    mode: GuidanceMode = GuidanceMode.SADDLE
    strength: float = 0.0
    max_step: float = 0.1
    trust_radius: float = 0.25
    force_clip: float | None = None
    collision_distance: float = 0.5
    failure_policy: FailurePolicy = FailurePolicy.REJECT
    stage: str = "late_generation"

    def __post_init__(self) -> None:
        mode = self.mode if isinstance(self.mode, GuidanceMode) else GuidanceMode(self.mode)
        policy = self.failure_policy if isinstance(self.failure_policy, FailurePolicy) else FailurePolicy(self.failure_policy)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "failure_policy", policy)
        if not isinstance(self.stage, str) or not self.stage.strip():
            raise ValueError("stage must be a nonempty string")
        for name in ("strength", "max_step", "trust_radius", "collision_distance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        if self.max_step <= 0 or self.trust_radius <= 0:
            raise ValueError("max_step and trust_radius must be positive")
        if self.force_clip is not None and (isinstance(self.force_clip, bool) or not isinstance(self.force_clip, (int, float)) or not math.isfinite(float(self.force_clip)) or self.force_clip <= 0):
            raise ValueError("force_clip must be positive or None")
        if self.mode is GuidanceMode.NONE and self.strength != 0:
            raise ValueError("none guidance must have zero strength")


@dataclass(frozen=True)
class GuidanceResult:
    status: str
    mode: GuidanceMode
    coordinates: tuple[tuple[float, float, float], ...]
    forces: tuple[tuple[float, float, float], ...] | None
    guidance: tuple[tuple[float, float, float], ...]
    calculator_calls: int
    error: str | None = None
    collision_rejected: bool = False


def _tensor_coordinates(system: MolecularSystem) -> torch.Tensor:
    return torch.tensor(system.coordinates, dtype=torch.float64)


def _validate_direction(direction: Sequence[Sequence[float]] | torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    tensor = direction if isinstance(direction, torch.Tensor) else torch.tensor(direction, dtype=torch.float64)
    if tensor.shape != shape or not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
        raise ValueError("reaction direction must be finite [N,3] matching the system")
    return tensor.to(dtype=torch.float64)


def normalize_direction(direction: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    norm = direction.square().sum().sqrt()
    if not math.isfinite(float(norm)) or float(norm) <= eps:
        raise ValueError("reaction direction must have a nonzero finite norm")
    return direction / norm


def saddle_guidance(forces: torch.Tensor, direction: torch.Tensor, *, strength: float = 1.0) -> torch.Tensor:
    """Return ``strength * (I - 2 u u^T) F`` using physical forces F."""

    if forces.shape != direction.shape or forces.ndim != 2 or forces.shape[-1] != 3:
        raise ValueError("forces and direction must both have shape [N,3]")
    if not bool(torch.isfinite(forces).all()) or not bool(torch.isfinite(direction).all()):
        raise ValueError("forces and direction must be finite")
    if not isinstance(strength, (int, float)) or not math.isfinite(float(strength)) or strength < 0:
        raise ValueError("strength must be finite and nonnegative")
    u = normalize_direction(direction.reshape(-1))
    force = forces.reshape(-1)
    reflected = force - 2.0 * torch.dot(u, force) * u
    return (float(strength) * reflected).reshape_as(forces)


def energy_descent_guidance(forces: torch.Tensor, *, strength: float = 1.0) -> torch.Tensor:
    if not isinstance(strength, (int, float)) or not math.isfinite(float(strength)) or strength < 0:
        raise ValueError("strength must be finite and nonnegative")
    if not bool(torch.isfinite(forces).all()):
        raise ValueError("forces must be finite")
    return float(strength) * forces


def _clip_and_trust(vector: torch.Tensor, config: GuidanceConfig) -> torch.Tensor:
    flat = vector.reshape(-1)
    if config.force_clip is not None:
        flat = flat.clamp(min=-config.force_clip, max=config.force_clip)
    norm = flat.norm()
    if float(norm) > config.trust_radius:
        flat = flat * (config.trust_radius / norm)
    return flat.reshape_as(vector)


def _collision(coordinates: torch.Tensor, minimum: float) -> bool:
    if coordinates.shape[0] < 2:
        return False
    distances = torch.cdist(coordinates, coordinates)
    distances = distances + torch.eye(coordinates.shape[0], dtype=distances.dtype) * 1e6
    return bool((distances < minimum).any())


def apply_guidance(
    backend: CalculatorBackend,
    system: MolecularSystem,
    config: GuidanceConfig,
    *,
    reaction_direction: Sequence[Sequence[float]] | torch.Tensor | None = None,
) -> GuidanceResult:
    """Evaluate physical forces and take one bounded coordinate guidance step.

    ``GuidanceMode.NONE`` is an exact zero-call baseline. Physical forces are
    only used for coordinates; they are never treated as gradients in event BE
    variables. A failed calculation returns a recorded status rather than a
    guessed force, according to the declared failure policy.
    """

    coordinates = _tensor_coordinates(system)
    zero = torch.zeros_like(coordinates)
    if config.mode is GuidanceMode.NONE or config.strength == 0:
        return GuidanceResult("unchanged", config.mode, tuple(map(tuple, coordinates.tolist())), None, tuple(map(tuple, zero.tolist())), 0)
    if _collision(coordinates, config.collision_distance):
        return GuidanceResult("collision_rejected", config.mode, tuple(map(tuple, coordinates.tolist())), None, tuple(map(tuple, zero.tolist())), 0, "input geometry is inside collision distance", True)
    try:
        result = backend.evaluate(system, operation="energy_forces")
    except Exception as exc:
        status = "stopped" if config.failure_policy is FailurePolicy.STOP else "skipped" if config.failure_policy is FailurePolicy.SKIP else "rejected"
        return GuidanceResult(status, config.mode, tuple(map(tuple, coordinates.tolist())), None, tuple(map(tuple, zero.tolist())), 1, f"{type(exc).__name__}: {exc}")
    if result.status != "success" or result.forces is None:
        status = "stopped" if config.failure_policy is FailurePolicy.STOP else "skipped" if config.failure_policy is FailurePolicy.SKIP else "rejected"
        return GuidanceResult(status, config.mode, tuple(map(tuple, coordinates.tolist())), None, tuple(map(tuple, zero.tolist())), result.calculator_calls, result.error_message or result.error_category)
    forces = torch.tensor(result.forces, dtype=torch.float64)
    if forces.shape != coordinates.shape:
        raise ValueError("calculator force shape does not match molecular coordinates")
    if config.mode is GuidanceMode.SADDLE:
        if reaction_direction is None:
            raise ValueError("saddle guidance requires a reaction direction")
        direction = _validate_direction(reaction_direction, tuple(coordinates.shape))
        raw = saddle_guidance(forces, direction, strength=config.strength)
    else:
        raw = energy_descent_guidance(forces, strength=config.strength)
    step = _clip_and_trust(raw, config)
    if step.norm() > 0:
        step = step * min(1.0, config.max_step / float(step.norm()))
    updated = coordinates + step
    if _collision(updated, config.collision_distance):
        return GuidanceResult("collision_rejected", config.mode, tuple(map(tuple, coordinates.tolist())), tuple(map(tuple, forces.tolist())), tuple(map(tuple, step.tolist())), result.calculator_calls, "guided geometry is inside collision distance", True)
    return GuidanceResult("updated", config.mode, tuple(map(tuple, updated.tolist())), tuple(map(tuple, forces.tolist())), tuple(map(tuple, step.tolist())), result.calculator_calls)


class PhysicalPostProcessor:
    """Shared bounded post-processing loop for all baseline arms."""

    def __init__(self, backend: CalculatorBackend, config: GuidanceConfig, *, max_calls: int = 8, stop_force_norm: float = 1e-4):
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError("max_calls must be a positive integer")
        if stop_force_norm <= 0 or not math.isfinite(stop_force_norm):
            raise ValueError("stop_force_norm must be finite and positive")
        self.backend = backend
        self.config = config
        self.max_calls = max_calls
        self.stop_force_norm = stop_force_norm

    def run(self, system: MolecularSystem, *, reaction_direction: Sequence[Sequence[float]] | torch.Tensor | None = None) -> tuple[GuidanceResult, ...]:
        results: list[GuidanceResult] = []
        current = system
        for _ in range(self.max_calls):
            result = apply_guidance(self.backend, current, self.config, reaction_direction=reaction_direction)
            results.append(result)
            if result.status != "updated" or result.forces is None:
                break
            if torch.tensor(result.forces, dtype=torch.float64).norm().item() <= self.stop_force_norm:
                break
            current = current.with_coordinates(result.coordinates)
        return tuple(results)
