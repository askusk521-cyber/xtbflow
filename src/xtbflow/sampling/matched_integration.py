"""Matched synthetic integration comparison for #166 PR D.

The comparison fixes the parent denominator, candidate-pool identity, seed,
provider revisions, tool feedback, and total budget.  It reports software
state transitions and separated cost accounting only; no chemistry backend is
called.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Iterable, Mapping

from .adapters import AdapterLedger, AdapterRunner, GuidanceBudget, ProviderRequest
from .parallel_guidance import GenerationSnapshot, GuidanceCost, GuidanceLane, MergeConfig, merge_guidance
from .physical_adapter import PhysicalFeedback, RecordedPhysicalProvider
from .policy_adapter import PolicyFeedback, RecordedPolicyProvider


class MatchedArm(str, Enum):
    NONE = "none"
    SERIAL_PHYSICAL = "serial_physical"
    PARALLEL_GEOMETRY_EVENT = "parallel_geometry_event"
    POLICY_ON = "policy_on"
    POLICY_OFF = "policy_off"


MATCHED_ARMS = tuple(MatchedArm)


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _snapshot(parent_id: str, seed: int) -> GenerationSnapshot:
    return GenerationSnapshot(
        parent_id=parent_id,
        attempt_id=f"matched-{parent_id}",
        generation_step=0,
        geometry=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        event=(0.0, 0.0, 0.0, 0.0),
        event_mask=(True, True, False, True),
        random_seed=seed,
        metadata={"temperature": 298.15, "source": "synthetic-reactant-only"},
        provenance={"matched": "issue166-pr-d-v1"},
    )


def _feedback(snapshot: GenerationSnapshot) -> tuple[PhysicalFeedback, PolicyFeedback]:
    physical = PhysicalFeedback(
        snapshot_id=snapshot.snapshot_id,
        forces=((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        reaction_direction=((0.0, 1.0, 0.0), (0.0, 0.0, 0.0)),
        provenance={"source": "matched-recorded-feedback"},
    )
    policy = PolicyFeedback(
        snapshot_id=snapshot.snapshot_id,
        ledger={"previous_cost": 0.0},
        tool_feedback={"ranking_signal": 0.5},
        action_id="matched-policy-action-v1",
        action_space={"allowed": ["event_delta"]},
        provenance={"source": "matched-recorded-feedback"},
    )
    return physical, policy


@dataclass(frozen=True)
class MatchedComparisonConfig:
    candidate_pool_id: str = "synthetic-matched-pool-166-v1"
    random_seed: int = 166
    total_budget: GuidanceCost = GuidanceCost(model_forwards=2, calculator_calls=1, wall_seconds=1.0)
    parent_ids: tuple[str, ...] = ("parent-001", "parent-002", "parent-003")

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_pool_id, str) or not self.candidate_pool_id.strip():
            raise ValueError("candidate_pool_id must be non-empty")
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise ValueError("random_seed must be nonnegative")
        if not isinstance(self.total_budget, GuidanceCost):
            raise ValueError("total_budget must be GuidanceCost")
        if not self.parent_ids or len(set(self.parent_ids)) != len(self.parent_ids) or any(not isinstance(item, str) or not item.strip() for item in self.parent_ids):
            raise ValueError("parent_ids must be unique non-empty strings")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "candidate_pool_id": self.candidate_pool_id,
            "random_seed": self.random_seed,
            "total_budget": self.total_budget.to_mapping(),
            "parent_ids": list(self.parent_ids),
        }


@dataclass(frozen=True)
class MatchedRow:
    parent_id: str
    arm: MatchedArm
    snapshot_id: str
    seed: int
    status: str
    geometry_changed: bool
    event_changed: bool
    model_forwards: int
    calculator_calls: int
    wall_seconds: float
    rejected: tuple[tuple[str, str], ...]
    adapter_ledger: AdapterLedger

    def to_mapping(self) -> dict[str, Any]:
        return {
            "parent_id": self.parent_id,
            "arm": self.arm.value,
            "snapshot_id": self.snapshot_id,
            "seed": self.seed,
            "status": self.status,
            "geometry_changed": self.geometry_changed,
            "event_changed": self.event_changed,
            "model_forwards": self.model_forwards,
            "calculator_calls": self.calculator_calls,
            "wall_seconds": self.wall_seconds,
            "rejected": [list(item) for item in self.rejected],
            "adapter_ledger": self.adapter_ledger.to_mapping(),
        }


@dataclass(frozen=True)
class MatchedComparisonArtifact:
    config: MatchedComparisonConfig
    rows: tuple[MatchedRow, ...]
    claim_limits: tuple[str, ...]

    def to_mapping(self) -> dict[str, Any]:
        payload = {
            "schema": "xtbflow-matched-guidance-comparison/v1",
            "config": self.config.to_mapping(),
            "arms": [arm.value for arm in MATCHED_ARMS],
            "rows": [row.to_mapping() for row in self.rows],
            "parent_denominator": len(self.config.parent_ids),
            "claim_limits": list(self.claim_limits),
        }
        payload["artifact_hash"] = _hash(payload)
        return payload


def _request(snapshot: GenerationSnapshot, provider, lane: GuidanceLane, feedback: Mapping[str, Any], action_space: Mapping[str, Any] | None = None) -> ProviderRequest:
    return ProviderRequest(
        snapshot=snapshot,
        lane=lane,
        provider_revision=provider.provider_revision,
        config_fingerprint=provider.config_fingerprint,
        protocol_fingerprint=provider.protocol_fingerprint,
        feedback=feedback,
        action_space=action_space or {},
    )


def _run_arm(snapshot: GenerationSnapshot, arm: MatchedArm, config: MatchedComparisonConfig) -> MatchedRow:
    physical_feedback, policy_feedback = _feedback(snapshot)
    budget = GuidanceBudget(config.total_budget)
    physical_provider = RecordedPhysicalProvider(physical_feedback)
    policy_provider = RecordedPolicyProvider(policy_feedback)
    physical_runner = AdapterRunner(budget)
    policy_runner = AdapterRunner(budget)
    outputs = []
    ledgers: list[AdapterLedger] = []
    rejected: tuple[tuple[str, str], ...] = ()

    if arm in {MatchedArm.SERIAL_PHYSICAL, MatchedArm.PARALLEL_GEOMETRY_EVENT}:
        physical_run = physical_runner.run(
            _request(snapshot, physical_provider, GuidanceLane.GEOMETRY, {"feedback_fingerprint": physical_feedback.feedback_fingerprint}),
            physical_provider,
        )
        ledgers.append(physical_runner.ledger)
        if physical_run.output is not None:
            outputs.append(physical_run.output)
    if arm in {MatchedArm.PARALLEL_GEOMETRY_EVENT, MatchedArm.POLICY_ON}:
        policy_run = policy_runner.run(
            _request(snapshot, policy_provider, GuidanceLane.EVENT, {"feedback_fingerprint": policy_feedback.feedback_fingerprint, "action_id": policy_feedback.action_id}, {"action_id": policy_feedback.action_id}),
            policy_provider,
        )
        ledgers.append(policy_runner.ledger)
        if policy_run.output is not None:
            outputs.append(policy_run.output)

    merged = merge_guidance(
        snapshot,
        tuple(outputs),
        config=MergeConfig(geometry_trust_radius=0.2, collision_distance=0.2),
        constraint_matrix=((1.0, 1.0, 1.0, 1.0),),
        event_velocity_fn=lambda geometry, event, _old: tuple(geometry[0][0] + value for value in event),
    )
    adapter_entries = tuple(entry for ledger in ledgers for entry in ledger.entries)
    combined_ledger = AdapterLedger(adapter_entries)
    charged = budget.charged
    status = "completed" if not merged.decision.rejected else "completed_with_rejections"
    return MatchedRow(
        parent_id=snapshot.parent_id,
        arm=arm,
        snapshot_id=snapshot.snapshot_id,
        seed=snapshot.random_seed,
        status=status,
        geometry_changed=merged.snapshot.geometry != snapshot.geometry,
        event_changed=merged.snapshot.event != snapshot.event,
        model_forwards=charged.model_forwards,
        calculator_calls=charged.calculator_calls,
        wall_seconds=charged.wall_seconds,
        rejected=merged.decision.rejected,
        adapter_ledger=combined_ledger,
    )


def run_matched_comparison(config: MatchedComparisonConfig = MatchedComparisonConfig()) -> MatchedComparisonArtifact:
    rows: list[MatchedRow] = []
    for parent_id in config.parent_ids:
        snapshot = _snapshot(parent_id, config.random_seed)
        for arm in MATCHED_ARMS:
            rows.append(_run_arm(snapshot, arm, config))
    return MatchedComparisonArtifact(
        config=config,
        rows=tuple(rows),
        claim_limits=(
            "This matched comparison proves only synthetic software state transitions and separated cost accounting.",
            "It does not establish chemistry, certification, physical accuracy, performance, or generalization.",
        ),
    )


__all__ = ["MatchedArm", "MATCHED_ARMS", "MatchedComparisonConfig", "MatchedRow", "MatchedComparisonArtifact", "run_matched_comparison"]
