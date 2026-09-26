"""Normal-mode checks kept separate from stationary-point checks."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class ModeEvidence:
    """Hessian eigenvalue evidence in the declared frequency/eigenvalue unit."""

    eigenvalues: tuple[float, ...]
    unit: str = "hessian_eigenvalue"

    def __post_init__(self) -> None:
        if not self.eigenvalues or any(not math.isfinite(float(value)) for value in self.eigenvalues):
            raise ValueError("eigenvalues must be a nonempty finite sequence")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("mode unit must be explicit")


def validate_mode(evidence: ModeEvidence, *, negative_threshold: float = 1e-4, max_negative_modes: int = 1) -> tuple[bool, str]:
    """Require a declared number of sufficiently negative modes.

    A small negative value inside the threshold is treated as numerical noise,
    so a low-gradient minimum cannot be labelled a transition state.
    """

    if negative_threshold <= 0 or not math.isfinite(float(negative_threshold)):
        raise ValueError("negative_threshold must be finite and positive")
    if type(max_negative_modes) is not int or max_negative_modes < 1:
        raise ValueError("max_negative_modes must be a positive integer")
    negative = [value for value in evidence.eigenvalues if value < -negative_threshold]
    if len(negative) != max_negative_modes:
        return False, f"expected {max_negative_modes} negative modes below -{negative_threshold:g}; observed {len(negative)}"
    return True, "mode_count_pass"

