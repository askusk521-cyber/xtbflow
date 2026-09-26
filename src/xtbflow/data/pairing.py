"""Immutable, same-geometry energy/force pairing contracts.

Pair construction never optimizes or reorders coordinates. A pair is admitted
only when both calculators report the identical molecular input hash and
explicit charge/multiplicity. Failed or missing labels remain machine-readable
failure rows so coverage cannot be inflated by dropping hard systems.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

from xtbflow.calculators.base import CalculationResult, MolecularSystem

from .records import canonical_hash


PAIR_STATUSES = frozenset({"success", "failure"})


def geometry_hash(system: MolecularSystem) -> str:
    """Hash element order and exact coordinate values, excluding labels."""

    return canonical_hash({"symbols": list(system.symbols), "coordinates": [list(row) for row in system.coordinates]})


def _force_rows(value: Sequence[Sequence[float]] | None, *, atoms: int, name: str) -> tuple[tuple[float, float, float], ...] | None:
    if value is None:
        return None
    rows = tuple(tuple(float(component) for component in row) for row in value)
    if len(rows) != atoms or any(len(row) != 3 or any(not math.isfinite(component) for component in row) for row in rows):
        raise ValueError(f"{name} must contain one finite xyz row per atom")
    return rows


@dataclass(frozen=True)
class EFPair:
    """One immutable semi-empirical/reference E/F pair or failure row."""

    pair_id: str
    source_record_id: str
    geometry_hash: str
    input_hash: str
    symbols: tuple[str, ...]
    coordinates: tuple[tuple[float, float, float], ...]
    charge: int
    multiplicity: int
    environment: Mapping[str, Any]
    semi_empirical_protocol_id: str
    reference_protocol_id: str
    units: Mapping[str, str]
    status: str = "success"
    semi_empirical_energy: float | None = None
    semi_empirical_forces: tuple[tuple[float, float, float], ...] | None = None
    reference_energy: float | None = None
    reference_forces: tuple[tuple[float, float, float], ...] | None = None
    delta_energy: float | None = None
    delta_forces: tuple[tuple[float, float, float], ...] | None = None
    error_category: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        for name in ("pair_id", "source_record_id", "geometry_hash", "input_hash", "semi_empirical_protocol_id", "reference_protocol_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        if self.status not in PAIR_STATUSES:
            raise ValueError(f"unsupported pair status: {self.status}")
        if type(self.charge) is not int or type(self.multiplicity) is not int or self.multiplicity < 1:
            raise ValueError("charge and multiplicity must be explicit")
        if len(self.symbols) != len(self.coordinates) or not self.symbols:
            raise ValueError("symbols and coordinates must describe a nonempty system")
        if any(len(row) != 3 or any(not math.isfinite(float(value)) for value in row) for row in self.coordinates):
            raise ValueError("coordinates must be finite xyz rows")
        if not isinstance(self.units, Mapping) or not self.units:
            raise ValueError("units must be explicit")
        for name in ("semi_empirical_energy", "reference_energy", "delta_energy"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite when present")
        atoms = len(self.symbols)
        _force_rows(self.semi_empirical_forces, atoms=atoms, name="semi_empirical_forces")
        _force_rows(self.reference_forces, atoms=atoms, name="reference_forces")
        _force_rows(self.delta_forces, atoms=atoms, name="delta_forces")
        if self.status == "success":
            required = (self.semi_empirical_energy, self.semi_empirical_forces, self.reference_energy, self.reference_forces, self.delta_energy, self.delta_forces)
            if any(value is None for value in required):
                raise ValueError("successful pairs require complete E/F and delta labels")
            if self.error_category or self.error_message:
                raise ValueError("successful pairs cannot carry failure details")
        elif not self.error_category or not self.error_message:
            raise ValueError("failure pairs require category and message")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["symbols"] = list(self.symbols)
        payload["coordinates"] = [list(row) for row in self.coordinates]
        for key in ("semi_empirical_forces", "reference_forces", "delta_forces"):
            value = payload[key]
            payload[key] = None if value is None else [list(row) for row in value]
        payload["environment"] = dict(self.environment)
        payload["units"] = dict(self.units)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EFPair":
        value = dict(payload)
        value["symbols"] = tuple(value["symbols"])
        value["coordinates"] = tuple(tuple(float(component) for component in row) for row in value["coordinates"])
        for key in ("semi_empirical_forces", "reference_forces", "delta_forces"):
            if value.get(key) is not None:
                value[key] = tuple(tuple(float(component) for component in row) for row in value[key])
        return cls(**value)


def _result_ok(result: CalculationResult, name: str) -> None:
    if result.status != "success" or not result.converged:
        raise ValueError(f"{name} result is not a converged success: {result.status}")
    if result.energy is None or result.forces is None:
        raise ValueError(f"{name} result is missing energy or forces")


def build_ef_pair(system: MolecularSystem, semi_empirical: CalculationResult, reference: CalculationResult, *, source_record_id: str, semi_empirical_protocol_id: str | None = None, reference_protocol_id: str | None = None, units: Mapping[str, str] | None = None) -> EFPair:
    """Build a pair without altering coordinates or calculator labels."""

    if semi_empirical.input_hash != system.input_hash or reference.input_hash != system.input_hash:
        raise ValueError("calculator inputs must match the exact system input hash")
    if (semi_empirical.charge, semi_empirical.multiplicity) != (system.charge, system.multiplicity) or (reference.charge, reference.multiplicity) != (system.charge, system.multiplicity):
        raise ValueError("calculator charge/multiplicity must match the system")
    _result_ok(semi_empirical, "semi_empirical")
    _result_ok(reference, "reference")
    se_forces = _force_rows(semi_empirical.forces, atoms=len(system.symbols), name="semi_empirical_forces")
    ref_forces = _force_rows(reference.forces, atoms=len(system.symbols), name="reference_forces")
    delta_forces = tuple(tuple(ref_forces[i][j] - se_forces[i][j] for j in range(3)) for i in range(len(system.symbols)))
    se_protocol = semi_empirical_protocol_id or semi_empirical.protocol_id
    ref_protocol = reference_protocol_id or reference.protocol_id
    pair_key = {"source_record_id": source_record_id, "input_hash": system.input_hash, "semi_empirical_protocol_id": se_protocol, "reference_protocol_id": ref_protocol}
    return EFPair(
        pair_id=canonical_hash(pair_key), source_record_id=source_record_id, geometry_hash=geometry_hash(system), input_hash=system.input_hash,
        symbols=system.symbols, coordinates=system.coordinates, charge=system.charge, multiplicity=system.multiplicity, environment=dict(system.environment),
        semi_empirical_protocol_id=se_protocol, reference_protocol_id=ref_protocol, units=dict(units or {"energy": "hartree", "force": "hartree/angstrom", "coordinate": "angstrom"}),
        semi_empirical_energy=float(semi_empirical.energy), semi_empirical_forces=se_forces, reference_energy=float(reference.energy), reference_forces=ref_forces,
        delta_energy=float(reference.energy - semi_empirical.energy), delta_forces=delta_forces,
    )


def failure_pair(system: MolecularSystem, *, source_record_id: str, semi_empirical_protocol_id: str, reference_protocol_id: str, category: str, message: str) -> EFPair:
    """Create a retained failure row so coverage and exclusions stay auditable."""

    pair_key = {"source_record_id": source_record_id, "input_hash": system.input_hash, "semi_empirical_protocol_id": semi_empirical_protocol_id, "reference_protocol_id": reference_protocol_id}
    return EFPair(pair_id=canonical_hash(pair_key), source_record_id=source_record_id, geometry_hash=geometry_hash(system), input_hash=system.input_hash, symbols=system.symbols, coordinates=system.coordinates, charge=system.charge, multiplicity=system.multiplicity, environment=dict(system.environment), semi_empirical_protocol_id=semi_empirical_protocol_id, reference_protocol_id=reference_protocol_id, units={"energy": "hartree", "force": "hartree/angstrom", "coordinate": "angstrom"}, status="failure", error_category=category, error_message=message)

