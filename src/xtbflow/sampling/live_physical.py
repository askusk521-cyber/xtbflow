"""Live xTB physical guidance for the first bounded feedback loop.

This module is intentionally small: one immutable generation snapshot is
converted into one explicit xTB ``energy_forces`` call, and the observed
forces are returned as bounded geometry guidance.  A second, zero-calculator
event lane derives a deterministic ranking delta from the same observed
feedback.  Providers never receive mutable state and never read products, TS
labels, endpoints, or sealed labels.

The provider proves a software/runtime closed loop only.  TS/IRC certification
is a separate downstream gate and must remain ``blocked`` when its required
evidence is absent.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from time import perf_counter, process_time
from typing import Any, Mapping, Sequence

from xtbflow.calculators import CalculatorBackend, CalculatorProtocol, MolecularSystem

from .adapters import ProviderRequest, fingerprint_text
from .parallel_guidance import (
    Event,
    GenerationSnapshot,
    Geometry,
    GuidanceCost,
    GuidanceLane,
    GuidanceOutput,
    reject_target_derived_fields,
)


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _geometry(value: Sequence[Sequence[float]], name: str) -> Geometry:
    rows: list[tuple[float, float, float]] = []
    if not value:
        raise ValueError(f"{name} must be nonempty")
    for index, row in enumerate(value):
        if len(row) != 3:
            raise ValueError(f"{name}[{index}] must have three components")
        rows.append(tuple(_finite(component, f"{name}[{index}]") for component in row))
    return tuple(rows)


def _force_norm(forces: Geometry) -> float:
    return math.sqrt(sum(component * component for row in forces for component in row))


def _scale(value: Geometry, factor: float) -> Geometry:
    return tuple(tuple(component * factor for component in row) for row in value)


@dataclass(frozen=True)
class LiveXTBFeedback:
    """Observed feedback from one current-snapshot xTB call."""

    snapshot_id: str
    input_hash: str
    protocol_id: str
    calculator_build_hash: str
    forces: Geometry
    scalar_observable: float
    force_norm: float
    calculator_calls: int
    wall_seconds: float
    cpu_seconds: float
    feedback_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_id, str) or len(self.snapshot_id) != 64:
            raise ValueError("snapshot_id must be a SHA-256 string")
        if not isinstance(self.input_hash, str) or len(self.input_hash) != 64:
            raise ValueError("input_hash must be a SHA-256 string")
        if not self.protocol_id.strip() or not self.calculator_build_hash.strip():
            raise ValueError("protocol_id and calculator_build_hash are required")
        forces = _geometry(self.forces, "forces")
        scalar = _finite(self.scalar_observable, "scalar_observable")
        norm = _finite(self.force_norm, "force_norm")
        if norm < 0 or type(self.calculator_calls) is not int or self.calculator_calls < 0:
            raise ValueError("force_norm/call count are invalid")
        wall = _finite(self.wall_seconds, "wall_seconds")
        cpu = _finite(self.cpu_seconds, "cpu_seconds")
        if wall < 0 or cpu < 0:
            raise ValueError("cost values must be nonnegative")
        payload = {
            "snapshot_id": self.snapshot_id,
            "input_hash": self.input_hash,
            "protocol_id": self.protocol_id,
            "calculator_build_hash": self.calculator_build_hash,
            "forces": [list(row) for row in forces],
            "scalar_observable": scalar,
            "force_norm": norm,
            "calculator_calls": self.calculator_calls,
            "wall_seconds": wall,
            "cpu_seconds": cpu,
        }
        computed = _hash(payload)
        if self.feedback_fingerprint and self.feedback_fingerprint != computed:
            raise ValueError("feedback_fingerprint does not match feedback contents")
        object.__setattr__(self, "forces", forces)
        object.__setattr__(self, "scalar_observable", scalar)
        object.__setattr__(self, "force_norm", norm)
        object.__setattr__(self, "wall_seconds", wall)
        object.__setattr__(self, "cpu_seconds", cpu)
        object.__setattr__(self, "feedback_fingerprint", computed)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "input_hash": self.input_hash,
            "protocol_id": self.protocol_id,
            "calculator_build_hash": self.calculator_build_hash,
            "forces": [list(row) for row in self.forces],
            "scalar_observable": self.scalar_observable,
            "force_norm": self.force_norm,
            "calculator_calls": self.calculator_calls,
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "feedback_fingerprint": self.feedback_fingerprint,
        }


@dataclass(frozen=True)
class LiveXTBGuidanceConfig:
    geometry_scale: float = 0.15
    geometry_trust_radius: float = 0.08
    event_scale: float = 0.03
    event_trust_radius: float = 0.12
    estimated_wall_seconds: float = 5.0

    def __post_init__(self) -> None:
        for name in ("geometry_scale", "geometry_trust_radius", "event_scale", "event_trust_radius", "estimated_wall_seconds"):
            value = _finite(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    def to_mapping(self) -> dict[str, float]:
        return {
            "geometry_scale": self.geometry_scale,
            "geometry_trust_radius": self.geometry_trust_radius,
            "event_scale": self.event_scale,
            "event_trust_radius": self.event_trust_radius,
            "estimated_wall_seconds": self.estimated_wall_seconds,
        }


class LiveXTBPhysicalProvider:
    """Call xTB once for the current snapshot and return bounded geometry guidance."""

    lane = GuidanceLane.GEOMETRY

    def __init__(
        self,
        backend: CalculatorBackend,
        molecular_template: MolecularSystem,
        protocol: CalculatorProtocol,
        *,
        config: LiveXTBGuidanceConfig | None = None,
    ) -> None:
        self.backend = backend
        self.molecular_template = molecular_template
        self.protocol = protocol
        self.config = config or LiveXTBGuidanceConfig()
        self.provider_revision = fingerprint_text("live-xtb-physical-provider-v1")
        self.config_fingerprint = _hash(self.config.to_mapping())
        self.protocol_fingerprint = protocol.identity
        self.last_feedback: LiveXTBFeedback | None = None
        self.last_error: str | None = None
        self.attempted_calls = 0

    def estimate(self, request: ProviderRequest) -> GuidanceCost:
        if len(request.snapshot.geometry) != len(self.molecular_template.symbols):
            raise ValueError("snapshot geometry does not match molecular template")
        return GuidanceCost(calculator_calls=1, wall_seconds=self.config.estimated_wall_seconds)

    def invoke(self, request: ProviderRequest) -> GuidanceOutput:
        self.attempted_calls += 1
        self.last_feedback = None
        self.last_error = None
        started_wall = perf_counter()
        started_cpu = process_time()
        system = self.molecular_template.with_coordinates(request.snapshot.geometry)
        try:
            result = self.backend.evaluate(system, operation="energy_forces")
            if result.status != "success" or result.forces is None or result.energy is None:
                raise RuntimeError(f"xTB result status={result.status!r} lacks converged energy/forces")
            forces = _geometry(result.forces, "calculator forces")
            wall_seconds = perf_counter() - started_wall
            cpu_seconds = process_time() - started_cpu
            build_hash = result.calculator_build_hash or "unavailable"
            feedback = LiveXTBFeedback(
                snapshot_id=request.snapshot.snapshot_id,
                input_hash=system.input_hash,
                protocol_id=result.protocol_id,
                calculator_build_hash=build_hash,
                forces=forces,
                scalar_observable=float(result.energy),
                force_norm=_force_norm(forces),
                calculator_calls=result.calculator_calls,
                wall_seconds=wall_seconds,
                cpu_seconds=cpu_seconds,
            )
            self.last_feedback = feedback
            delta = _scale(forces, self.config.geometry_scale)
            provenance = {
                "calculator_protocol": result.protocol_id,
                "calculator_build_hash": build_hash,
                "calculator_input_hash": system.input_hash,
                "feedback_fingerprint": feedback.feedback_fingerprint,
                "observable_l2": feedback.force_norm,
            }
            reject_target_derived_fields(provenance, name="live xTB guidance provenance")
            return GuidanceOutput(
                lane=self.lane,
                provider="live-xtb-physical-v1",
                input_fingerprint=request.snapshot.snapshot_id,
                geometry_delta=delta,
                trust_radius=self.config.geometry_trust_radius,
                estimated_cost=self.estimate(request),
                measured_cost=GuidanceCost(
                    calculator_calls=result.calculator_calls,
                    wall_seconds=wall_seconds,
                    cpu_seconds=cpu_seconds,
                ),
                provenance=provenance,
            )
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise


class LiveXTBEventProvider:
    """Derive a conservation-projected event delta from the same xTB feedback."""

    lane = GuidanceLane.EVENT

    def __init__(self, feedback: LiveXTBFeedback, event_size: int, *, config: LiveXTBGuidanceConfig | None = None) -> None:
        if type(event_size) is not int or event_size < 1:
            raise ValueError("event_size must be positive")
        self.feedback = feedback
        self.event_size = event_size
        self.config = config or LiveXTBGuidanceConfig()
        self.provider_revision = fingerprint_text("live-xtb-event-ranking-v1")
        self.config_fingerprint = _hash({"event_size": event_size, **self.config.to_mapping()})
        self.protocol_fingerprint = fingerprint_text("live-xtb-event-ranking-from-force-v1")

    def estimate(self, request: ProviderRequest) -> GuidanceCost:
        if request.snapshot.snapshot_id != self.feedback.snapshot_id:
            raise ValueError("event provider feedback does not match snapshot")
        return GuidanceCost(wall_seconds=0.001)

    def invoke(self, request: ProviderRequest) -> GuidanceOutput:
        if request.snapshot.snapshot_id != self.feedback.snapshot_id:
            raise ValueError("event provider feedback does not match snapshot")
        atom_scores = tuple(_force_norm((row,)) for row in self.feedback.forces)
        raw = []
        for index in range(self.event_size):
            score = atom_scores[index % len(atom_scores)]
            raw.append(score if index % 2 == 0 else -score)
        mean = sum(raw) / len(raw)
        delta = tuple(self.config.event_scale * (value - mean) for value in raw)
        provenance = {
            "source_feedback_fingerprint": self.feedback.feedback_fingerprint,
            "ranking_fingerprint": _hash({"atom_scores": atom_scores, "event_size": self.event_size}),
            "provider_revision": self.provider_revision,
        }
        reject_target_derived_fields(provenance, name="live xTB event provenance")
        return GuidanceOutput(
            lane=self.lane,
            provider="live-xtb-event-ranking-v1",
            input_fingerprint=request.snapshot.snapshot_id,
            event_delta=delta,
            trust_radius=self.config.event_trust_radius,
            estimated_cost=self.estimate(request),
            measured_cost=GuidanceCost(wall_seconds=0.001),
            provenance=provenance,
        )


def xTB_protocol_from_mapping(settings: Mapping[str, Any]) -> CalculatorProtocol:
    """Build the explicit shared xTB protocol used by the loop."""

    return CalculatorProtocol(
        protocol_id=str(settings["protocol_id"]),
        calculator="xtb_oracle",
        method="GFN2-xTB",
        backend="cpu",
        parameters={
            "accuracy": float(settings["accuracy"]),
            "max_iterations": int(settings["max_iterations"]),
            "electronic_temperature": int(settings["electronic_temperature"]),
            "solvent": settings.get("solvent"),
        },
    )


__all__ = [
    "LiveXTBFeedback", "LiveXTBGuidanceConfig", "LiveXTBPhysicalProvider",
    "LiveXTBEventProvider", "xTB_protocol_from_mapping",
]
