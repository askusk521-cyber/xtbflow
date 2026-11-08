"""Normal-mode checks kept separate from stationary-point checks."""
from __future__ import annotations

from dataclasses import dataclass
import math


# Thresholds are defined in this canonical representation.  Hessian values
# are commonly emitted in Hartree/Bohr², so accepting that representation
# without conversion would make the same physical mode pass or fail solely
# because a producer changed its coordinate unit.
CANONICAL_MODE_UNIT = "hartree_per_angstrom2"
BOHR_TO_ANGSTROM = 0.529177210903
MODE_EVIDENCE_CONTRACT = "cartesian-unweighted-hartree_per_angstrom2/v1"

_UNIT_FACTORS = {
    CANONICAL_MODE_UNIT: 1.0,
    "hartree_per_bohr2": 1.0 / BOHR_TO_ANGSTROM**2,
}
_UNIT_ALIASES = {
    "hartree/angstrom^2": CANONICAL_MODE_UNIT,
    "hartree/angstrom2": CANONICAL_MODE_UNIT,
    "eh/angstrom^2": CANONICAL_MODE_UNIT,
    "eh/angstrom2": CANONICAL_MODE_UNIT,
    "hartree/bohr^2": "hartree_per_bohr2",
    "hartree/bohr2": "hartree_per_bohr2",
    "eh/bohr^2": "hartree_per_bohr2",
    "eh/bohr2": "hartree_per_bohr2",
}


def normalize_mode_unit(unit: str) -> str:
    """Return a supported mode unit or fail closed for an ambiguous label."""

    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("mode unit must be explicit")
    normalized = (
        unit.strip()
        .lower()
        .replace(" ", "")
        .replace("å", "angstrom")
        .replace("²", "2")
    )
    normalized = _UNIT_ALIASES.get(normalized, normalized)
    if normalized not in _UNIT_FACTORS:
        supported = ", ".join(sorted(_UNIT_FACTORS))
        raise ValueError(f"unsupported mode unit {unit!r}; use one of: {supported}")
    return normalized


@dataclass(frozen=True)
class ModeEvidence:
    """Unweighted Cartesian Hessian eigenvalues, never frequencies.

    Units must be declared at the boundary. Mass-weighted Hessians, internal
    coordinates, and frequencies require different thresholds and are rejected.
    """

    eigenvalues: tuple[float, ...]
    unit: str
    coordinate_system: str = "cartesian"
    mass_weighted: bool = False

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.eigenvalues)
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError("eigenvalues must be a nonempty finite sequence")
        object.__setattr__(self, "eigenvalues", values)
        object.__setattr__(self, "unit", normalize_mode_unit(self.unit))
        if self.coordinate_system != "cartesian" or self.mass_weighted is not False:
            raise ValueError("modes must be unweighted Cartesian Hessian eigenvalues")
        if any(not math.isfinite(value) for value in self.canonical_eigenvalues):
            raise ValueError("converted eigenvalues must be finite")

    @property
    def canonical_eigenvalues(self) -> tuple[float, ...]:
        factor = _UNIT_FACTORS[self.unit]
        return tuple(value * factor for value in self.eigenvalues)


def validate_mode(evidence: ModeEvidence, *, negative_threshold: float = 1e-4, max_negative_modes: int = 1) -> tuple[bool, str]:
    """Require a declared number of sufficiently negative modes.

    The threshold is in Hartree/angstrom² for an unweighted Cartesian Hessian.
    A small negative value inside the threshold is treated as numerical noise,
    so a low-gradient minimum cannot be labelled a transition state.
    """

    if negative_threshold <= 0 or not math.isfinite(float(negative_threshold)):
        raise ValueError("negative_threshold must be finite and positive")
    if type(max_negative_modes) is not int or max_negative_modes < 1:
        raise ValueError("max_negative_modes must be a positive integer")
    negative = [value for value in evidence.canonical_eigenvalues if value < -negative_threshold]
    if len(negative) != max_negative_modes:
        return False, f"expected {max_negative_modes} negative modes below -{negative_threshold:g}; observed {len(negative)}"
    return True, "mode_count_pass"
