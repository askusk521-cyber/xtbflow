"""Immutable parallel-guidance contracts and deterministic merge gate.

The first ``#166`` layer is intentionally calculator-free.  It defines the
state, lane output, merge, freeze, conservation, and accounting contracts that
later physical and policy providers must implement.  A merge never mutates
the input snapshot; it returns a new snapshot, a decision record, and an
append-only ledger.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


Geometry = tuple[tuple[float, float, float], ...]
Event = tuple[float, ...]
FrozenValue = Any


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _freeze(value: Any, *, name: str = "value") -> FrozenValue:
    """Convert JSON-compatible values to immutable tuples."""

    if isinstance(value, Mapping):
        pairs: list[tuple[str, FrozenValue]] = []
        for key, nested in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{name} mapping keys must be non-empty strings")
            pairs.append((key, _freeze(nested, name=f"{name}.{key}")))
        return tuple(sorted(pairs))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, name=f"{name}[{index}]") for index, item in enumerate(value))
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if type(value) is int:
        return value
    if isinstance(value, float) and math.isfinite(value):
        return float(value)
    raise ValueError(f"{name} must be JSON-compatible and finite")


def _thaw(value: FrozenValue) -> Any:
    if isinstance(value, tuple):
        if all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value):
            return {key: _thaw(nested) for key, nested in value}
        return [_thaw(item) for item in value]
    return value


_TARGET_PARTS = frozenset(
    {
        "product", "mapped_product", "ts", "reference", "target", "sealed",
        "label", "barrier", "energy", "forces", "irc", "answer",
        "ground_truth", "gold", "event",
    }
)
_TARGET_EXACT = frozenset(
    {
        "productcoordinates", "productgeometry", "tsgeometry", "referencets",
        "referencemode", "activewaterfromts", "targetactivewater", "sealedtestlabel",
        "eventlabel", "eventgeometry", "eventbonds",
    }
)


def _normalise_key(key: str) -> str:
    camel = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key.strip())
    return re.sub(r"[^A-Za-z0-9]+", "_", camel).strip("_").lower()


def _target_derived_key(key: str) -> bool:
    normalized = _normalise_key(key)
    if not normalized:
        return False
    return bool(set(normalized.split("_")) & _TARGET_PARTS) or normalized.replace("_", "") in _TARGET_EXACT


def reject_target_derived_fields(value: Any, *, name: str = "guidance input") -> None:
    """Recursively reject target/reference/TS-derived mapping keys."""

    def walk(current: Any, path: str) -> None:
        if isinstance(current, Mapping):
            for key, nested in current.items():
                if not isinstance(key, str):
                    raise ValueError(f"{name} contains a non-string key at {path}")
                if _target_derived_key(key):
                    raise ValueError(f"target-derived field {path}.{key} is forbidden in {name}")
                walk(nested, f"{path}.{key}")
        elif isinstance(current, (list, tuple)):
            for index, nested in enumerate(current):
                walk(nested, f"{path}[{index}]")

    walk(value, name)


def _finite_number(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    number = float(value)
    if nonnegative and number < 0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _geometry(value: Any, name: str) -> Geometry:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"{name} must be a non-empty [N,3] sequence")
    rows: list[tuple[float, float, float]] = []
    for index, row in enumerate(value):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or len(row) != 3:
            raise ValueError(f"{name}[{index}] must have exactly three coordinates")
        rows.append(tuple(_finite_number(component, f"{name}[{index}]") for component in row))  # type: ignore[arg-type]
    return tuple(rows)


def _event(value: Any, name: str) -> Event:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"{name} must be a non-empty sequence")
    return tuple(_finite_number(item, name) for item in value)


def _mask(value: Any, name: str) -> tuple[bool, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"{name} must be a non-empty boolean sequence")
    if any(type(item) is not bool for item in value):
        raise ValueError(f"{name} must contain only booleans")
    return tuple(value)


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(sum(float(item) * float(item) for item in value))


def _add_geometry(left: Geometry, right: Geometry) -> Geometry:
    if len(left) != len(right):
        raise ValueError("geometry shapes must match")
    return tuple(tuple(a + b for a, b in zip(row_left, row_right)) for row_left, row_right in zip(left, right))


def _scale_geometry(value: Geometry, factor: float) -> Geometry:
    return tuple(tuple(component * factor for component in row) for row in value)


def _zero_geometry(n_atoms: int) -> Geometry:
    return tuple((0.0, 0.0, 0.0) for _ in range(n_atoms))


def _collision(geometry: Geometry, minimum_distance: float) -> bool:
    if minimum_distance <= 0 or len(geometry) < 2:
        return False
    threshold = minimum_distance * minimum_distance
    for index, left in enumerate(geometry):
        for right in geometry[index + 1 :]:
            squared = sum((a - b) ** 2 for a, b in zip(left, right))
            if squared < threshold:
                return True
    return False


class GuidanceLane(str, Enum):
    BASE = "base"
    GEOMETRY = "geometry"
    EVENT = "event"


class GuidanceStatus(str, Enum):
    ACCEPTED = "accepted"
    INVALID = "invalid"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"


@dataclass(frozen=True)
class GuidanceCost:
    """Bounded cost vector for one guidance action."""

    model_forwards: int = 0
    calculator_calls: int = 0
    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    gpu_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name in ("model_forwards", "calculator_calls"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("wall_seconds", "cpu_seconds", "gpu_seconds"):
            _finite_number(getattr(self, name), name, nonnegative=True)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "model_forwards": self.model_forwards,
            "calculator_calls": self.calculator_calls,
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "gpu_seconds": self.gpu_seconds,
        }

    @classmethod
    def zero(cls) -> "GuidanceCost":
        return cls()


@dataclass(frozen=True)
class GenerationSnapshot:
    """Immutable input state shared by every lane at one generation step."""

    parent_id: str
    attempt_id: str
    generation_step: int
    geometry: Geometry
    event: Event
    event_mask: tuple[bool, ...]
    random_seed: int
    metadata: FrozenValue = field(default_factory=tuple)
    provenance: FrozenValue = field(default_factory=tuple)
    snapshot_id: str = ""
    event_frozen: bool = False

    def __post_init__(self) -> None:
        for name in ("parent_id", "attempt_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        if type(self.generation_step) is not int or self.generation_step < 0:
            raise ValueError("generation_step must be a nonnegative integer")
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise ValueError("random_seed must be a nonnegative integer")
        geometry = _geometry(self.geometry, "geometry")
        event = _event(self.event, "event")
        mask = _mask(self.event_mask, "event_mask")
        if len(event) != len(mask):
            raise ValueError("event and event_mask must have the same length")
        metadata = _freeze(self.metadata, name="metadata")
        provenance = _freeze(self.provenance, name="provenance")
        reject_target_derived_fields(_thaw(metadata), name="snapshot metadata")
        reject_target_derived_fields(_thaw(provenance), name="snapshot provenance")
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "event", event)
        object.__setattr__(self, "event_mask", mask)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "provenance", provenance)
        computed = _canonical_hash(self.to_mapping(include_snapshot_id=False))
        if self.snapshot_id and self.snapshot_id != computed:
            raise ValueError("snapshot_id does not match immutable snapshot contents")
        object.__setattr__(self, "snapshot_id", computed)

    def to_mapping(self, *, include_snapshot_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "parent_id": self.parent_id,
            "attempt_id": self.attempt_id,
            "generation_step": self.generation_step,
            "geometry": [list(row) for row in self.geometry],
            "event": list(self.event),
            "event_mask": list(self.event_mask),
            "random_seed": self.random_seed,
            "metadata": _thaw(self.metadata),
            "provenance": _thaw(self.provenance),
            "event_frozen": self.event_frozen,
        }
        if include_snapshot_id:
            payload["snapshot_id"] = self.snapshot_id
        return payload


@dataclass(frozen=True)
class GuidanceOutput:
    """One bounded result returned by a guidance lane."""

    lane: GuidanceLane | str
    provider: str
    input_fingerprint: str
    geometry_delta: Geometry | None = None
    direction_delta: Geometry | None = None
    event_delta: Event | None = None
    ranking: tuple[tuple[str, float], ...] | None = None
    trust_radius: float = 1.0
    estimated_cost: GuidanceCost = field(default_factory=GuidanceCost.zero)
    measured_cost: GuidanceCost = field(default_factory=GuidanceCost.zero)
    status: GuidanceStatus | str = GuidanceStatus.ACCEPTED
    failure_stage: str | None = None
    rejection_reason: str | None = None
    provenance: FrozenValue = field(default_factory=tuple)
    target_free: bool = True
    output_fingerprint: str = ""

    def __post_init__(self) -> None:
        lane = self.lane if isinstance(self.lane, GuidanceLane) else GuidanceLane(self.lane)
        status = self.status if isinstance(self.status, GuidanceStatus) else GuidanceStatus(self.status)
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider must be a non-empty string")
        if not isinstance(self.input_fingerprint, str) or len(self.input_fingerprint) != 64:
            raise ValueError("input_fingerprint must be a SHA-256 string")
        if not isinstance(self.estimated_cost, GuidanceCost) or not isinstance(self.measured_cost, GuidanceCost):
            raise ValueError("estimated_cost and measured_cost must be GuidanceCost values")
        if type(self.target_free) is not bool:
            raise ValueError("target_free must be a boolean")
        radius = _finite_number(self.trust_radius, "trust_radius", nonnegative=True)
        if radius <= 0:
            raise ValueError("trust_radius must be positive")
        geometry_delta = None if self.geometry_delta is None else _geometry(self.geometry_delta, "geometry_delta")
        direction_delta = None if self.direction_delta is None else _geometry(self.direction_delta, "direction_delta")
        event_delta = None if self.event_delta is None else _event(self.event_delta, "event_delta")
        ranking = None if self.ranking is None else tuple((str(key), _finite_number(score, "ranking score")) for key, score in self.ranking)
        if lane is GuidanceLane.GEOMETRY and event_delta is not None:
            raise ValueError("geometry lane cannot return an event delta")
        if lane is GuidanceLane.EVENT and (geometry_delta is not None or direction_delta is not None):
            raise ValueError("event lane cannot return geometry guidance")
        if lane is GuidanceLane.BASE and ranking is not None:
            raise ValueError("base lane cannot return an event ranking")
        if ranking is not None:
            if lane is not GuidanceLane.EVENT:
                raise ValueError("only event lane may return a ranking")
        if status in {GuidanceStatus.FAILED, GuidanceStatus.BUDGET_EXHAUSTED} and not self.failure_stage:
            raise ValueError("failed outputs require failure_stage")
        if status is GuidanceStatus.REJECTED and not self.rejection_reason:
            raise ValueError("rejected outputs require rejection_reason")
        if not self.target_free:
            raise ValueError("guidance output must declare target_free=true")
        provenance = _freeze(self.provenance, name="provenance")
        reject_target_derived_fields(_thaw(provenance), name="guidance provenance")
        object.__setattr__(self, "lane", lane)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "trust_radius", radius)
        object.__setattr__(self, "geometry_delta", geometry_delta)
        object.__setattr__(self, "direction_delta", direction_delta)
        object.__setattr__(self, "event_delta", event_delta)
        object.__setattr__(self, "ranking", ranking)
        object.__setattr__(self, "provenance", provenance)
        payload = self.to_mapping(include_output_fingerprint=False)
        for key in ("estimated_cost", "measured_cost", "status", "failure_stage", "rejection_reason"):
            payload.pop(key, None)
        computed = _canonical_hash(payload)
        if self.output_fingerprint and self.output_fingerprint != computed:
            raise ValueError("output_fingerprint does not match output contents")
        object.__setattr__(self, "output_fingerprint", computed)

    def to_mapping(self, *, include_output_fingerprint: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "lane": self.lane.value,
            "provider": self.provider,
            "input_fingerprint": self.input_fingerprint,
            "geometry_delta": None if self.geometry_delta is None else [list(row) for row in self.geometry_delta],
            "direction_delta": None if self.direction_delta is None else [list(row) for row in self.direction_delta],
            "event_delta": None if self.event_delta is None else list(self.event_delta),
            "ranking": None if self.ranking is None else [list(item) for item in self.ranking],
            "trust_radius": self.trust_radius,
            "estimated_cost": self.estimated_cost.to_mapping(),
            "measured_cost": self.measured_cost.to_mapping(),
            "status": self.status.value,
            "failure_stage": self.failure_stage,
            "rejection_reason": self.rejection_reason,
            "provenance": _thaw(self.provenance),
            "target_free": self.target_free,
        }
        if include_output_fingerprint:
            payload["output_fingerprint"] = self.output_fingerprint
        return payload


@dataclass(frozen=True)
class MergeConfig:
    """Deterministic merge policy shared by all lanes."""

    lane_priority: tuple[GuidanceLane | str, ...] = (
        GuidanceLane.BASE,
        GuidanceLane.GEOMETRY,
        GuidanceLane.EVENT,
    )
    geometry_trust_radius: float = 0.25
    collision_distance: float = 0.0
    event_freeze_step: int | None = None
    projection_tolerance: float = 1e-7

    def __post_init__(self) -> None:
        priority = tuple(item if isinstance(item, GuidanceLane) else GuidanceLane(item) for item in self.lane_priority)
        if set(priority) != set(GuidanceLane) or len(priority) != len(GuidanceLane):
            raise ValueError("lane_priority must contain base, geometry, and event exactly once")
        trust = _finite_number(self.geometry_trust_radius, "geometry_trust_radius", nonnegative=True)
        collision = _finite_number(self.collision_distance, "collision_distance", nonnegative=True)
        if trust <= 0:
            raise ValueError("geometry_trust_radius must be positive")
        if self.event_freeze_step is not None and (type(self.event_freeze_step) is not int or self.event_freeze_step < 0):
            raise ValueError("event_freeze_step must be None or a nonnegative integer")
        tolerance = _finite_number(self.projection_tolerance, "projection_tolerance", nonnegative=True)
        if tolerance <= 0:
            raise ValueError("projection_tolerance must be positive")
        object.__setattr__(self, "lane_priority", priority)
        object.__setattr__(self, "geometry_trust_radius", trust)
        object.__setattr__(self, "collision_distance", collision)
        object.__setattr__(self, "projection_tolerance", tolerance)

    def priority(self, lane: GuidanceLane) -> int:
        return self.lane_priority.index(lane)


@dataclass(frozen=True)
class GuidanceLedgerEntry:
    snapshot_id: str
    generation_step: int
    lane: GuidanceLane
    provider: str
    input_fingerprint: str
    output_fingerprint: str
    status: GuidanceStatus
    applied: bool
    rejection_reason: str | None
    failure_stage: str | None
    estimated_cost: GuidanceCost
    measured_cost: GuidanceCost
    charged_cost: GuidanceCost
    target_free: bool
    provenance: FrozenValue

    def to_mapping(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "generation_step": self.generation_step,
            "lane": self.lane.value,
            "provider": self.provider,
            "input_fingerprint": self.input_fingerprint,
            "output_fingerprint": self.output_fingerprint,
            "status": self.status.value,
            "applied": self.applied,
            "rejection_reason": self.rejection_reason,
            "failure_stage": self.failure_stage,
            "estimated_cost": self.estimated_cost.to_mapping(),
            "measured_cost": self.measured_cost.to_mapping(),
            "charged_cost": self.charged_cost.to_mapping(),
            "target_free": self.target_free,
            "provenance": _thaw(self.provenance),
        }


@dataclass(frozen=True)
class GuidanceLedger:
    """Immutable append-only action ledger."""

    entries: tuple[GuidanceLedgerEntry, ...] = ()

    def append(self, entry: GuidanceLedgerEntry) -> "GuidanceLedger":
        return GuidanceLedger(self.entries + (entry,))

    def to_mapping(self) -> dict[str, Any]:
        return {"schema": "xtbflow-guidance-ledger/v1", "entries": [entry.to_mapping() for entry in self.entries]}


@dataclass(frozen=True)
class MergeDecision:
    snapshot_id: str
    lane_order: tuple[str, ...]
    applied_lanes: tuple[str, ...]
    rejected: tuple[tuple[str, str], ...]
    geometry_before: Geometry
    geometry_after: Geometry
    event_before: Event
    event_after: Event
    event_velocity: Event
    conservation_residual: tuple[float, ...]
    event_frozen: bool

    def to_mapping(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "lane_order": list(self.lane_order),
            "applied_lanes": list(self.applied_lanes),
            "rejected": [list(item) for item in self.rejected],
            "geometry_before": [list(row) for row in self.geometry_before],
            "geometry_after": [list(row) for row in self.geometry_after],
            "event_before": list(self.event_before),
            "event_after": list(self.event_after),
            "event_velocity": list(self.event_velocity),
            "conservation_residual": list(self.conservation_residual),
            "event_frozen": self.event_frozen,
        }


@dataclass(frozen=True)
class MergeResult:
    snapshot: GenerationSnapshot
    decision: MergeDecision
    ledger: GuidanceLedger


def project_masked_event_delta(
    delta: Sequence[float],
    event_mask: Sequence[bool],
    constraint_matrix: Sequence[Sequence[float]],
    *,
    tolerance: float = 1e-7,
) -> tuple[Event, tuple[float, ...]]:
    """Project an event delta into ``A delta = 0`` while zeroing padding."""

    vector = np.asarray(_event(delta, "event delta"), dtype=float)
    mask = np.asarray(_mask(event_mask, "event_mask"), dtype=bool)
    if vector.shape != mask.shape:
        raise ValueError("event delta and event_mask must have the same length")
    matrix = np.asarray(constraint_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != vector.size or matrix.shape[0] == 0 or not np.isfinite(matrix).all():
        raise ValueError("constraint_matrix must be finite [constraints, features]")
    active = np.flatnonzero(mask)
    projected = np.zeros_like(vector)
    if active.size:
        active_matrix = matrix[:, active]
        active_values = vector[active]
        gram = active_matrix @ active_matrix.T
        projected[active] = active_values - active_matrix.T @ np.linalg.pinv(gram) @ active_matrix @ active_values
    residual = matrix @ projected
    if not np.all(np.isfinite(residual)) or float(np.max(np.abs(residual), initial=0.0)) > tolerance:
        raise ValueError("event projection residual exceeds tolerance")
    return tuple(float(value) for value in projected), tuple(float(value) for value in residual)


def _clip_geometry(delta: Geometry, radius: float) -> Geometry:
    flat = tuple(component for row in delta for component in row)
    norm = _norm(flat)
    return delta if norm <= radius else _scale_geometry(delta, radius / norm)


def _zero_cost() -> GuidanceCost:
    return GuidanceCost.zero()


def _ledger_entry(
    snapshot: GenerationSnapshot,
    output: GuidanceOutput,
    *,
    status: GuidanceStatus,
    applied: bool,
    rejection_reason: str | None,
    charged_cost: GuidanceCost,
) -> GuidanceLedgerEntry:
    return GuidanceLedgerEntry(
        snapshot_id=snapshot.snapshot_id,
        generation_step=snapshot.generation_step,
        lane=output.lane,
        provider=output.provider,
        input_fingerprint=output.input_fingerprint,
        output_fingerprint=output.output_fingerprint,
        status=status,
        applied=applied,
        rejection_reason=rejection_reason,
        failure_stage=output.failure_stage,
        estimated_cost=output.estimated_cost,
        measured_cost=output.measured_cost,
        charged_cost=charged_cost,
        target_free=output.target_free,
        provenance=output.provenance,
    )


EventVelocityFn = Callable[[Geometry, Event, GenerationSnapshot], Sequence[float]]


def merge_guidance(
    snapshot: GenerationSnapshot,
    outputs: Iterable[GuidanceOutput],
    *,
    config: MergeConfig = MergeConfig(),
    constraint_matrix: Sequence[Sequence[float]] | None = None,
    event_velocity_fn: EventVelocityFn | None = None,
) -> MergeResult:
    """Apply deterministic lane outputs to one immutable snapshot.

    Outputs are sorted by the declared lane priority and then provider and
    fingerprint.  A duplicate output is logged with zero charged cost.  A
    failed or invalid lane never prevents other valid lanes from being
    processed.  Event projection is applied once after all accepted event
    deltas, and event velocity is recomputed only after the geometry state is
    final.
    """

    ordered = sorted(
        tuple(outputs),
        key=lambda item: (config.priority(item.lane), item.provider, item.output_fingerprint),
    )
    geometry = snapshot.geometry
    raw_event = snapshot.event
    event_frozen = snapshot.event_frozen or (
        config.event_freeze_step is not None and snapshot.generation_step >= config.event_freeze_step
    )
    event_delta = [0.0] * len(snapshot.event)
    seen: set[str] = set()
    ledger = GuidanceLedger()
    rejected: list[tuple[str, str]] = []
    applied_lanes: list[str] = []
    lane_order = tuple(f"{item.lane.value}:{item.provider}" for item in ordered)

    for output in ordered:
        if output.output_fingerprint in seen:
            ledger = ledger.append(_ledger_entry(snapshot, output, status=GuidanceStatus.DUPLICATE, applied=False, rejection_reason="duplicate_guidance", charged_cost=_zero_cost()))
            rejected.append((output.lane.value, "duplicate_guidance"))
            continue
        seen.add(output.output_fingerprint)
        if output.input_fingerprint != snapshot.snapshot_id:
            ledger = ledger.append(_ledger_entry(snapshot, output, status=GuidanceStatus.REJECTED, applied=False, rejection_reason="snapshot_mismatch", charged_cost=output.measured_cost))
            rejected.append((output.lane.value, "snapshot_mismatch"))
            continue
        if output.status is not GuidanceStatus.ACCEPTED:
            ledger = ledger.append(_ledger_entry(snapshot, output, status=output.status, applied=False, rejection_reason=output.rejection_reason or output.status.value, charged_cost=output.measured_cost))
            rejected.append((output.lane.value, output.rejection_reason or output.status.value))
            continue
        if output.lane is GuidanceLane.EVENT and event_frozen:
            ledger = ledger.append(_ledger_entry(snapshot, output, status=GuidanceStatus.REJECTED, applied=False, rejection_reason="event_frozen", charged_cost=output.measured_cost))
            rejected.append((output.lane.value, "event_frozen"))
            continue
        candidate_geometry = geometry
        geometry_delta = output.geometry_delta
        if output.direction_delta is not None:
            geometry_delta = _add_geometry(geometry_delta or _zero_geometry(len(geometry)), output.direction_delta)
        if geometry_delta is not None:
            if len(geometry_delta) != len(geometry):
                raise ValueError("geometry guidance shape does not match snapshot")
            bounded = _clip_geometry(geometry_delta, min(config.geometry_trust_radius, output.trust_radius))
            candidate_geometry = _add_geometry(geometry, bounded)
            if _collision(candidate_geometry, config.collision_distance):
                ledger = ledger.append(_ledger_entry(snapshot, output, status=GuidanceStatus.REJECTED, applied=False, rejection_reason="collision_guard", charged_cost=output.measured_cost))
                rejected.append((output.lane.value, "collision_guard"))
                continue
            geometry = candidate_geometry
        if output.event_delta is not None:
            if len(output.event_delta) != len(raw_event):
                raise ValueError("event guidance shape does not match snapshot")
            for index, value in enumerate(output.event_delta):
                event_delta[index] += value
        ledger = ledger.append(_ledger_entry(snapshot, output, status=GuidanceStatus.ACCEPTED, applied=True, rejection_reason=None, charged_cost=output.measured_cost))
        applied_lanes.append(output.lane.value)

    has_event_delta = any(value != 0.0 for value in event_delta)
    if has_event_delta and constraint_matrix is None:
        raise ValueError("constraint_matrix is required whenever an event delta is applied")
    if constraint_matrix is None:
        projected_delta = tuple(event_delta)
        residual = tuple()
    else:
        projected_delta, residual = project_masked_event_delta(event_delta, snapshot.event_mask, constraint_matrix, tolerance=config.projection_tolerance)
    event = tuple(value + delta for value, delta in zip(raw_event, projected_delta))
    if event_velocity_fn is None:
        velocity = tuple(value - before for value, before in zip(event, raw_event))
    else:
        velocity = _event(event_velocity_fn(geometry, event, snapshot), "event_velocity")
        if len(velocity) != len(event):
            raise ValueError("event_velocity_fn returned the wrong feature dimension")
    decision = MergeDecision(
        snapshot_id=snapshot.snapshot_id,
        lane_order=lane_order,
        applied_lanes=tuple(applied_lanes),
        rejected=tuple(rejected),
        geometry_before=snapshot.geometry,
        geometry_after=geometry,
        event_before=snapshot.event,
        event_after=event,
        event_velocity=velocity,
        conservation_residual=residual,
        event_frozen=event_frozen,
    )
    next_snapshot = GenerationSnapshot(
        parent_id=snapshot.parent_id,
        attempt_id=snapshot.attempt_id,
        generation_step=snapshot.generation_step + 1,
        geometry=geometry,
        event=event,
        event_mask=snapshot.event_mask,
        random_seed=snapshot.random_seed,
        metadata=_thaw(snapshot.metadata),
        provenance=_thaw(snapshot.provenance),
        event_frozen=event_frozen,
    )
    return MergeResult(next_snapshot, decision, ledger)


__all__ = [
    "Event", "Geometry", "GuidanceLane", "GuidanceStatus", "GuidanceCost",
    "GenerationSnapshot", "GuidanceOutput", "MergeConfig", "GuidanceLedgerEntry",
    "GuidanceLedger", "MergeDecision", "MergeResult", "reject_target_derived_fields",
    "project_masked_event_delta", "merge_guidance",
]
