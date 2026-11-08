"""Recorded physical/geometry guidance adapter for #166 PR B.

The adapter consumes only feedback visible at the current snapshot and returns
bounded geometry guidance through the PR A provider boundary.  It is a
recorded-feedback fixture: no xTB, CP2K, GPU, or live calculator is called.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping

from .adapters import (
    AdapterLedger,
    AdapterRun,
    AdapterRunner,
    GuidanceBudget,
    ProviderRequest,
    fingerprint_text,
)
from .parallel_guidance import (
    Geometry,
    GenerationSnapshot,
    GuidanceCost,
    GuidanceLane,
    GuidanceOutput,
    MergeConfig,
    MergeResult,
    merge_guidance,
    reject_target_derived_fields,
)


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _geometry(value: Any, name: str) -> Geometry:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty [N,3] sequence")
    rows = []
    for row in value:
        if isinstance(row, (str, bytes)) or not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError(f"{name} rows must have exactly three components")
        components = []
        for component in row:
            if isinstance(component, bool) or not isinstance(component, (int, float)) or not math.isfinite(float(component)):
                raise ValueError(f"{name} must be finite")
            components.append(float(component))
        rows.append(tuple(components))
    return tuple(rows)


def _zero_like(value: Geometry) -> Geometry:
    return tuple((0.0, 0.0, 0.0) for _ in value)


@dataclass(frozen=True)
class PhysicalFeedback:
    """Feedback visible at one snapshot; target-derived fields are forbidden."""

    snapshot_id: str
    forces: Geometry
    reaction_direction: Geometry | None = None
    curvature: Geometry | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    feedback_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_id, str) or len(self.snapshot_id) != 64:
            raise ValueError("snapshot_id must be a SHA-256 string")
        forces = _geometry(self.forces, "forces")
        direction = None if self.reaction_direction is None else _geometry(self.reaction_direction, "reaction_direction")
        curvature = None if self.curvature is None else _geometry(self.curvature, "curvature")
        if direction is not None and len(direction) != len(forces):
            raise ValueError("reaction_direction shape must match forces")
        if curvature is not None and len(curvature) != len(forces):
            raise ValueError("curvature shape must match forces")
        provenance = dict(self.provenance)
        reject_target_derived_fields(provenance, name="physical feedback provenance")
        payload = {
            "snapshot_id": self.snapshot_id,
            "forces": [list(row) for row in forces],
            "reaction_direction": None if direction is None else [list(row) for row in direction],
            "curvature": None if curvature is None else [list(row) for row in curvature],
            "provenance": provenance,
        }
        computed = _hash(payload)
        if self.feedback_fingerprint and self.feedback_fingerprint != computed:
            raise ValueError("feedback_fingerprint does not match feedback contents")
        object.__setattr__(self, "forces", forces)
        object.__setattr__(self, "reaction_direction", direction)
        object.__setattr__(self, "curvature", curvature)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "feedback_fingerprint", computed)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "forces": [list(row) for row in self.forces],
            "reaction_direction": None if self.reaction_direction is None else [list(row) for row in self.reaction_direction],
            "curvature": None if self.curvature is None else [list(row) for row in self.curvature],
            "provenance": dict(self.provenance),
            "feedback_fingerprint": self.feedback_fingerprint,
        }


@dataclass(frozen=True)
class PhysicalGuidanceConfig:
    force_weight: float = 1.0
    direction_weight: float = 0.2
    curvature_weight: float = 0.1
    max_step: float = 0.1
    trust_radius: float = 0.1
    estimated_cost: GuidanceCost = field(default_factory=lambda: GuidanceCost(model_forwards=1, calculator_calls=1, wall_seconds=0.05))
    measured_cost: GuidanceCost = field(default_factory=lambda: GuidanceCost(model_forwards=1, calculator_calls=1, wall_seconds=0.04))

    def __post_init__(self) -> None:
        for name in ("force_weight", "direction_weight", "curvature_weight", "max_step", "trust_radius"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.max_step <= 0 or self.trust_radius <= 0:
            raise ValueError("max_step and trust_radius must be positive")
        if not isinstance(self.estimated_cost, GuidanceCost) or not isinstance(self.measured_cost, GuidanceCost):
            raise ValueError("cost fields must be GuidanceCost")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "force_weight": self.force_weight,
            "direction_weight": self.direction_weight,
            "curvature_weight": self.curvature_weight,
            "max_step": self.max_step,
            "trust_radius": self.trust_radius,
            "estimated_cost": self.estimated_cost.to_mapping(),
            "measured_cost": self.measured_cost.to_mapping(),
        }


class RecordedPhysicalProvider:
    """Deterministic physical lane provider backed by recorded feedback."""

    lane = GuidanceLane.GEOMETRY

    def __init__(self, feedback: PhysicalFeedback, config: PhysicalGuidanceConfig | None = None, *, protocol_id: str = "recorded-physical-v1") -> None:
        self.feedback = feedback
        self.config = config or PhysicalGuidanceConfig()
        self.provider_revision = fingerprint_text("recorded-physical-provider-v1")
        self.config_fingerprint = _hash(self.config.to_mapping())
        self.protocol_fingerprint = fingerprint_text(protocol_id)
        self.invocations = 0

    def estimate(self, request: ProviderRequest) -> GuidanceCost:
        if request.snapshot.snapshot_id != self.feedback.snapshot_id:
            raise ValueError("feedback snapshot does not match provider request")
        return self.config.estimated_cost

    def invoke(self, request: ProviderRequest) -> GuidanceOutput:
        if request.snapshot.snapshot_id != self.feedback.snapshot_id:
            raise ValueError("feedback snapshot does not match provider request")
        self.invocations += 1
        direction = self.feedback.reaction_direction or _zero_like(self.feedback.forces)
        curvature = self.feedback.curvature or _zero_like(self.feedback.forces)
        raw = tuple(
            tuple(
                self.config.force_weight * force
                + self.config.direction_weight * direction_component
                + self.config.curvature_weight * curvature_component
                for force, direction_component, curvature_component in zip(force_row, direction_row, curvature_row)
            )
            for force_row, direction_row, curvature_row in zip(self.feedback.forces, direction, curvature)
        )
        norm = math.sqrt(sum(component * component for row in raw for component in row))
        radius = min(self.config.max_step, self.config.trust_radius)
        delta = raw if norm <= radius or norm == 0 else tuple(tuple(component * radius / norm for component in row) for row in raw)
        return GuidanceOutput(
            lane=self.lane,
            provider="recorded-physical-v1",
            input_fingerprint=request.snapshot.snapshot_id,
            geometry_delta=delta,
            trust_radius=self.config.trust_radius,
            estimated_cost=self.config.estimated_cost,
            measured_cost=self.config.measured_cost,
            provenance={
                "physical": {
                    "feedback_fingerprint": self.feedback.feedback_fingerprint,
                    "provider_revision": self.provider_revision,
                    "config_fingerprint": self.config_fingerprint,
                    "protocol_fingerprint": self.protocol_fingerprint,
                }
            },
        )


@dataclass(frozen=True)
class PhysicalAblationArtifact:
    enabled: bool
    candidate_pool_id: str
    initial_snapshot: GenerationSnapshot
    final_snapshot: GenerationSnapshot
    adapter_run: AdapterRun | None
    merge_result: MergeResult
    adapter_ledger: AdapterLedger
    claim_limits: tuple[str, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": "xtbflow-physical-guidance-ablation/v1",
            "enabled": self.enabled,
            "candidate_pool_id": self.candidate_pool_id,
            "initial_snapshot": self.initial_snapshot.to_mapping(),
            "final_snapshot": self.final_snapshot.to_mapping(),
            "adapter_run": None if self.adapter_run is None else {
                "status": self.adapter_run.status.value,
                "request_fingerprint": self.adapter_run.request_fingerprint,
                "failure_stage": self.adapter_run.failure_stage,
                "error": self.adapter_run.error,
                "estimated_cost": self.adapter_run.estimated_cost.to_mapping(),
                "measured_cost": self.adapter_run.measured_cost.to_mapping(),
                "charged_cost": self.adapter_run.charged_cost.to_mapping(),
            },
            "merge_decision": self.merge_result.decision.to_mapping(),
            "merge_ledger": self.merge_result.ledger.to_mapping(),
            "adapter_ledger": self.adapter_ledger.to_mapping(),
            "claim_limits": list(self.claim_limits),
        }


def run_physical_ablation(
    snapshot: GenerationSnapshot,
    feedback: PhysicalFeedback,
    *,
    enabled: bool,
    candidate_pool_id: str,
    budget: GuidanceBudget,
    merge_config: MergeConfig = MergeConfig(geometry_trust_radius=0.2, collision_distance=0.2),
    constraint_matrix: tuple[tuple[float, ...], ...] = ((1.0, 1.0, 1.0, 1.0),),
    config: PhysicalGuidanceConfig | None = None,
) -> PhysicalAblationArtifact:
    if not isinstance(candidate_pool_id, str) or not candidate_pool_id.strip():
        raise ValueError("candidate_pool_id must be non-empty")
    if feedback.snapshot_id != snapshot.snapshot_id:
        raise ValueError("feedback snapshot does not match generation snapshot")
    provider = RecordedPhysicalProvider(feedback, config=config)
    runner = AdapterRunner(budget)
    adapter_run: AdapterRun | None = None
    outputs = ()
    if enabled:
        request = ProviderRequest(
            snapshot=snapshot,
            lane=GuidanceLane.GEOMETRY,
            provider_revision=provider.provider_revision,
            config_fingerprint=provider.config_fingerprint,
            protocol_fingerprint=provider.protocol_fingerprint,
            feedback={"feedback_fingerprint": feedback.feedback_fingerprint},
        )
        adapter_run = runner.run(request, provider)
        if adapter_run.output is not None:
            outputs = (adapter_run.output,)
    merged = merge_guidance(
        snapshot,
        outputs,
        config=merge_config,
        constraint_matrix=constraint_matrix,
        event_velocity_fn=lambda geometry, event, _old: tuple(geometry[0][0] + value for value in event),
    )
    return PhysicalAblationArtifact(
        enabled=enabled,
        candidate_pool_id=candidate_pool_id,
        initial_snapshot=snapshot,
        final_snapshot=merged.snapshot,
        adapter_run=adapter_run,
        merge_result=merged,
        adapter_ledger=runner.ledger,
        claim_limits=(
            "Recorded physical feedback proves only bounded software guidance and cost accounting.",
            "It does not establish physical accuracy, chemistry, certification, performance, or generalization.",
        ),
    )


__all__ = [
    "PhysicalFeedback", "PhysicalGuidanceConfig", "RecordedPhysicalProvider",
    "PhysicalAblationArtifact", "run_physical_ablation",
]
