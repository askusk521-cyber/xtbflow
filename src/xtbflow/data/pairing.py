"""Immutable, same-geometry energy/force pairing contracts.

Pair construction never optimizes or reorders coordinates. A pair is admitted
only when both calculators report the identical molecular input hash and
explicit charge/multiplicity. Failed or missing labels remain machine-readable
failure rows so coverage cannot be inflated by dropping hard systems.  Loaded
rows are treated as untrusted serialized data: their hashes, deltas, units, and
pair identity are recomputed before admission.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Any, Mapping, Sequence

from xtbflow.calculators.base import CalculationResult, MolecularSystem

from .records import canonical_hash


PAIR_CANONICAL_UNITS = {
    "coordinate": "angstrom",
    "energy": "hartree",
    "force": "hartree/angstrom",
}
PAIR_STATUSES = frozenset({"success", "failure"})


def geometry_hash(system: MolecularSystem) -> str:
    """Hash element order and canonical coordinate values, excluding labels.

    ``MolecularSystem`` accepts integer-valued coordinates as valid numeric
    input.  Normalize them to floats here so the hash is identical before and
    after a JSON round trip (where ``0`` is normalized to ``0.0`` by
    :class:`EFPair`).
    """

    return canonical_hash({
        "symbols": list(system.symbols),
        "coordinates": [[float(value) for value in row] for row in system.coordinates],
    })


def _strict_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _force_rows(value: Sequence[Sequence[float]] | None, *, atoms: int, name: str) -> tuple[tuple[float, float, float], ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must contain one finite xyz row per atom")
    rows: list[tuple[float, float, float]] = []
    for row in value:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or len(row) != 3:
            raise ValueError(f"{name} must contain one finite xyz row per atom")
        rows.append(tuple(_strict_float(component, name) for component in row))
    result = tuple(rows)
    if len(result) != atoms:
        raise ValueError(f"{name} must contain one finite xyz row per atom")
    return result


def _coordinates(value: Any, symbols: Sequence[str]) -> tuple[tuple[float, float, float], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("coordinates must contain one finite xyz row per atom")
    rows: list[tuple[float, float, float]] = []
    for row in value:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or len(row) != 3:
            raise ValueError("coordinates must contain one finite xyz row per atom")
        rows.append(tuple(_strict_float(component, "coordinate") for component in row))
    result = tuple(rows)
    if not symbols or len(result) != len(symbols):
        raise ValueError("symbols and coordinates must describe a nonempty system")
    return result


def _symbols(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError("symbols must be a nonempty sequence of element names")
    result = tuple(value)
    if any(not isinstance(symbol, str) or not symbol.strip() for symbol in result):
        raise ValueError("symbols must be a nonempty sequence of element names")
    return result


def _json_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    try:
        canonical_hash(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON-compatible") from exc
    return dict(value)


def _pair_identity(*, source_record_id: str, input_hash: str, semi_empirical_protocol_id: str, reference_protocol_id: str) -> str:
    return canonical_hash({
        "source_record_id": source_record_id,
        "input_hash": input_hash,
        "semi_empirical_protocol_id": semi_empirical_protocol_id,
        "reference_protocol_id": reference_protocol_id,
    })


def _expected_input_hash(symbols: tuple[str, ...], coordinates: tuple[tuple[float, float, float], ...], charge: int, multiplicity: int, environment: Mapping[str, Any]) -> str:
    system = MolecularSystem(symbols, coordinates, charge, multiplicity, environment)
    return system.input_hash


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
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if self.status not in PAIR_STATUSES:
            raise ValueError(f"unsupported pair status: {self.status}")
        if type(self.charge) is not int or type(self.multiplicity) is not int or self.multiplicity < 1:
            raise ValueError("charge and multiplicity must be explicit strict integers")

        symbols = _symbols(self.symbols)
        coordinates = _coordinates(self.coordinates, symbols)
        environment = _json_mapping(self.environment, "environment")
        units = _json_mapping(self.units, "units")
        if units != PAIR_CANONICAL_UNITS:
            raise ValueError(f"units must be canonical: {PAIR_CANONICAL_UNITS}")
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "environment", environment)
        object.__setattr__(self, "units", dict(PAIR_CANONICAL_UNITS))

        # Recompute all identity material from the serialized molecular state.
        expected_geometry_hash = geometry_hash(MolecularSystem(symbols, coordinates, self.charge, self.multiplicity, environment))
        if self.geometry_hash != expected_geometry_hash:
            raise ValueError("geometry_hash does not match symbols and coordinates")
        expected_input_hash = _expected_input_hash(symbols, coordinates, self.charge, self.multiplicity, environment)
        if self.input_hash != expected_input_hash:
            raise ValueError("input_hash does not match molecular state")
        expected_pair_id = _pair_identity(
            source_record_id=self.source_record_id,
            input_hash=expected_input_hash,
            semi_empirical_protocol_id=self.semi_empirical_protocol_id,
            reference_protocol_id=self.reference_protocol_id,
        )
        if self.pair_id != expected_pair_id:
            raise ValueError("pair_id does not match source/input/protocol identity")

        for name in ("semi_empirical_energy", "reference_energy", "delta_energy"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _strict_float(value, name))
        atoms = len(symbols)
        for name in ("semi_empirical_forces", "reference_forces", "delta_forces"):
            value = _force_rows(getattr(self, name), atoms=atoms, name=name)
            object.__setattr__(self, name, value)

        if self.status == "success":
            required = (
                self.semi_empirical_energy,
                self.semi_empirical_forces,
                self.reference_energy,
                self.reference_forces,
                self.delta_energy,
                self.delta_forces,
            )
            if any(value is None for value in required):
                raise ValueError("successful pairs require complete E/F and delta labels")
            if self.error_category is not None or self.error_message is not None:
                raise ValueError("successful pairs cannot carry failure details")
            expected_delta_energy = self.reference_energy - self.semi_empirical_energy
            if self.delta_energy != expected_delta_energy:
                raise ValueError("delta_energy does not equal reference minus semi-empirical energy")
            expected_delta_forces = tuple(
                tuple(self.reference_forces[i][j] - self.semi_empirical_forces[i][j] for j in range(3))
                for i in range(atoms)
            )
            if self.delta_forces != expected_delta_forces:
                raise ValueError("delta_forces do not equal reference minus semi-empirical forces")
        else:
            if not isinstance(self.error_category, str) or not self.error_category.strip():
                raise ValueError("failure pairs require a nonempty error category")
            if not isinstance(self.error_message, str) or not self.error_message.strip():
                raise ValueError("failure pairs require a nonempty error message")
            if any(value is not None for value in (
                self.semi_empirical_energy,
                self.semi_empirical_forces,
                self.reference_energy,
                self.reference_forces,
                self.delta_energy,
                self.delta_forces,
            )):
                raise ValueError("failure pairs cannot carry numerical labels")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["symbols"] = list(self.symbols)
        payload["coordinates"] = [list(row) for row in self.coordinates]
        for key in ("semi_empirical_forces", "reference_forces", "delta_forces"):
            value = payload[key]
            payload[key] = None if value is None else [list(row) for row in value]
        payload["environment"] = dict(self.environment)
        payload["units"] = dict(PAIR_CANONICAL_UNITS)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EFPair":
        """Load and integrity-check an untrusted serialized pair payload."""

        if not isinstance(payload, Mapping):
            raise ValueError("E/F pair payload must be a mapping")
        value = dict(payload)
        required = {
            "pair_id", "source_record_id", "geometry_hash", "input_hash", "symbols", "coordinates",
            "charge", "multiplicity", "environment", "semi_empirical_protocol_id",
            "reference_protocol_id", "units", "status",
        }
        missing = required - set(value)
        if missing:
            raise ValueError(f"E/F pair payload is missing fields: {sorted(missing)}")
        allowed = required | {
            "semi_empirical_energy", "semi_empirical_forces", "reference_energy", "reference_forces",
            "delta_energy", "delta_forces", "error_category", "error_message",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unsupported E/F pair fields: {sorted(unknown)}")

        # Normalize JSON arrays to immutable tuples only after checking their
        # shape and numeric domain.  This prevents strings/bools from becoming
        # plausible coordinates or force values through float coercion.
        symbols = _symbols(value["symbols"])
        coordinates = _coordinates(value["coordinates"], symbols)
        for key in ("semi_empirical_forces", "reference_forces", "delta_forces"):
            if value.get(key) is not None:
                value[key] = _force_rows(value[key], atoms=len(symbols), name=key)
        for key in ("semi_empirical_energy", "reference_energy", "delta_energy"):
            if value.get(key) is not None:
                value[key] = _strict_float(value[key], key)
        value["symbols"] = symbols
        value["coordinates"] = coordinates
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
    requested_units = dict(units or PAIR_CANONICAL_UNITS)
    if requested_units != PAIR_CANONICAL_UNITS:
        raise ValueError(f"units must be canonical: {PAIR_CANONICAL_UNITS}")
    return EFPair(
        pair_id=canonical_hash(pair_key), source_record_id=source_record_id, geometry_hash=geometry_hash(system), input_hash=system.input_hash,
        symbols=system.symbols, coordinates=system.coordinates, charge=system.charge, multiplicity=system.multiplicity, environment=dict(system.environment),
        semi_empirical_protocol_id=se_protocol, reference_protocol_id=ref_protocol, units=dict(PAIR_CANONICAL_UNITS),
        semi_empirical_energy=float(semi_empirical.energy), semi_empirical_forces=se_forces, reference_energy=float(reference.energy), reference_forces=ref_forces,
        delta_energy=float(reference.energy - semi_empirical.energy), delta_forces=delta_forces,
    )


def failure_pair(system: MolecularSystem, *, source_record_id: str, semi_empirical_protocol_id: str, reference_protocol_id: str, category: str, message: str) -> EFPair:
    """Create a retained failure row so coverage and exclusions stay auditable."""

    pair_key = {"source_record_id": source_record_id, "input_hash": system.input_hash, "semi_empirical_protocol_id": semi_empirical_protocol_id, "reference_protocol_id": reference_protocol_id}
    return EFPair(pair_id=canonical_hash(pair_key), source_record_id=source_record_id, geometry_hash=geometry_hash(system), input_hash=system.input_hash, symbols=system.symbols, coordinates=system.coordinates, charge=system.charge, multiplicity=system.multiplicity, environment=dict(system.environment), semi_empirical_protocol_id=semi_empirical_protocol_id, reference_protocol_id=reference_protocol_id, units=dict(PAIR_CANONICAL_UNITS), status="failure", error_category=category, error_message=message)
