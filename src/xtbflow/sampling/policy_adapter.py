"""Recorded policy/event guidance adapter for #166 PR C.

The policy boundary consumes only the current snapshot, ledger/tool feedback,
and a declared action space.  It returns an event delta or ranking; geometry,
freeze state, and ledgers remain owned by the merge gate and adapter runner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping

from .adapters import AdapterLedger, AdapterRun, AdapterRunner, GuidanceBudget, ProviderRequest, fingerprint_text
from .parallel_guidance import (
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


@dataclass(frozen=True)
class PolicyFeedback:
    snapshot_id: str
    ledger: Mapping[str, Any] = field(default_factory=dict)
    tool_feedback: Mapping[str, Any] = field(default_factory=dict)
    action_id: str = "recorded-event-action"
    action_space: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    feedback_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_id, str) or len(self.snapshot_id) != 64:
            raise ValueError("snapshot_id must be a SHA-256 string")
        if not isinstance(self.action_id, str) or not self.action_id.strip():
            raise ValueError("action_id must be a non-empty string")
        values = {}
        for name, value in (("ledger", self.ledger), ("tool_feedback", self.tool_feedback), ("action_space", self.action_space), ("provenance", self.provenance)):
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be a mapping")
            try:
                json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be JSON-compatible and finite") from exc
            reject_target_derived_fields(value, name=f"policy {name}")
            values[name] = dict(value)
        payload = {
            "snapshot_id": self.snapshot_id,
            "ledger": values["ledger"],
            "tool_feedback": values["tool_feedback"],
            "action_id": self.action_id,
            "action_space": values["action_space"],
            "provenance": values["provenance"],
        }
        computed = _hash(payload)
        if self.feedback_fingerprint and self.feedback_fingerprint != computed:
            raise ValueError("feedback_fingerprint does not match policy feedback")
        object.__setattr__(self, "ledger", values["ledger"])
        object.__setattr__(self, "tool_feedback", values["tool_feedback"])
        object.__setattr__(self, "action_space", values["action_space"])
        object.__setattr__(self, "provenance", values["provenance"])
        object.__setattr__(self, "feedback_fingerprint", computed)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "ledger": dict(self.ledger),
            "tool_feedback": dict(self.tool_feedback),
            "action_id": self.action_id,
            "action_space": dict(self.action_space),
            "provenance": dict(self.provenance),
            "feedback_fingerprint": self.feedback_fingerprint,
        }


@dataclass(frozen=True)
class PolicyGuidanceConfig:
    event_trust_radius: float = 0.2
    delta_scale: float = 0.04
    estimated_cost: GuidanceCost = field(default_factory=lambda: GuidanceCost(model_forwards=1, wall_seconds=0.02))
    measured_cost: GuidanceCost = field(default_factory=lambda: GuidanceCost(model_forwards=1, wall_seconds=0.01))

    def __post_init__(self) -> None:
        for name in ("event_trust_radius", "delta_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.estimated_cost, GuidanceCost) or not isinstance(self.measured_cost, GuidanceCost):
            raise ValueError("cost fields must be GuidanceCost")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "event_trust_radius": self.event_trust_radius,
            "delta_scale": self.delta_scale,
            "estimated_cost": self.estimated_cost.to_mapping(),
            "measured_cost": self.measured_cost.to_mapping(),
        }


class RecordedPolicyProvider:
    lane = GuidanceLane.EVENT

    def __init__(self, feedback: PolicyFeedback, config: PolicyGuidanceConfig | None = None, *, protocol_id: str = "recorded-policy-v1") -> None:
        self.feedback = feedback
        self.config = config or PolicyGuidanceConfig()
        self.provider_revision = fingerprint_text("recorded-policy-provider-v1")
        self.config_fingerprint = _hash(self.config.to_mapping())
        self.protocol_fingerprint = fingerprint_text(protocol_id)
        self.invocations = 0

    def estimate(self, request: ProviderRequest) -> GuidanceCost:
        if request.snapshot.snapshot_id != self.feedback.snapshot_id:
            raise ValueError("policy feedback snapshot does not match provider request")
        if request.action_space.get("action_id") != self.feedback.action_id:
            raise ValueError("policy action identity does not match request")
        return self.config.estimated_cost

    def invoke(self, request: ProviderRequest) -> GuidanceOutput:
        if request.snapshot.snapshot_id != self.feedback.snapshot_id:
            raise ValueError("policy feedback snapshot does not match provider request")
        self.invocations += 1
        scale = self.config.delta_scale
        return GuidanceOutput(
            lane=self.lane,
            provider="recorded-policy-v1",
            input_fingerprint=request.snapshot.snapshot_id,
            event_delta=(scale, -scale / 2.0, 0.0, scale / 4.0),
            trust_radius=self.config.event_trust_radius,
            estimated_cost=self.config.estimated_cost,
            measured_cost=self.config.measured_cost,
            provenance={
                "policy": {
                    "action_id": self.feedback.action_id,
                    "feedback_fingerprint": self.feedback.feedback_fingerprint,
                    "provider_revision": self.provider_revision,
                    "config_fingerprint": self.config_fingerprint,
                    "protocol_fingerprint": self.protocol_fingerprint,
                }
            },
        )


@dataclass(frozen=True)
class PolicyAblationArtifact:
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
            "schema": "xtbflow-policy-guidance-ablation/v1",
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
                "measured_cost": None if self.adapter_run.measured_cost is None else self.adapter_run.measured_cost.to_mapping(),
                "cost_basis": self.adapter_run.ledger_entry.cost_basis,
                "backend_termination": self.adapter_run.ledger_entry.backend_termination,
                "charged_cost": self.adapter_run.charged_cost.to_mapping(),
            },
            "merge_decision": self.merge_result.decision.to_mapping(),
            "merge_ledger": self.merge_result.ledger.to_mapping(),
            "adapter_ledger": self.adapter_ledger.to_mapping(),
            "claim_limits": list(self.claim_limits),
        }


def run_policy_ablation(
    snapshot: GenerationSnapshot,
    feedback: PolicyFeedback,
    *,
    enabled: bool,
    candidate_pool_id: str,
    budget: GuidanceBudget,
    merge_config: MergeConfig = MergeConfig(geometry_trust_radius=0.2, collision_distance=0.2),
    constraint_matrix: tuple[tuple[float, ...], ...] = ((1.0, 1.0, 1.0, 1.0),),
    config: PolicyGuidanceConfig | None = None,
) -> PolicyAblationArtifact:
    if not isinstance(candidate_pool_id, str) or not candidate_pool_id.strip():
        raise ValueError("candidate_pool_id must be non-empty")
    if feedback.snapshot_id != snapshot.snapshot_id:
        raise ValueError("policy feedback snapshot does not match generation snapshot")
    provider = RecordedPolicyProvider(feedback, config=config)
    runner = AdapterRunner(budget)
    adapter_run: AdapterRun | None = None
    outputs = ()
    if enabled:
        request = ProviderRequest(
            snapshot=snapshot,
            lane=GuidanceLane.EVENT,
            provider_revision=provider.provider_revision,
            config_fingerprint=provider.config_fingerprint,
            protocol_fingerprint=provider.protocol_fingerprint,
            feedback={"feedback_fingerprint": feedback.feedback_fingerprint, "action_id": feedback.action_id},
            action_space={"action_id": feedback.action_id, **dict(feedback.action_space)},
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
    return PolicyAblationArtifact(
        enabled=enabled,
        candidate_pool_id=candidate_pool_id,
        initial_snapshot=snapshot,
        final_snapshot=merged.snapshot,
        adapter_run=adapter_run,
        merge_result=merged,
        adapter_ledger=runner.ledger,
        claim_limits=(
            "Recorded policy feedback proves only bounded software event guidance and cost accounting.",
            "It does not establish policy quality, chemistry, certification, performance, or generalization.",
        ),
    )


__all__ = [
    "PolicyFeedback", "PolicyGuidanceConfig", "RecordedPolicyProvider",
    "PolicyAblationArtifact", "run_policy_ablation",
]
