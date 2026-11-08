"""Deterministic synthetic replay for the #166 guidance ablations.

This module deliberately stays calculator-free.  It runs the same immutable
initial snapshot, seed, candidate-pool identity, providers, and total budget
through the five required modes so the first-phase state transition and
accounting contract can be audited before physical adapters are connected.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Iterable, Mapping

from .parallel_guidance import (
    Event,
    GenerationSnapshot,
    GuidanceCost,
    GuidanceLane,
    GuidanceLedger,
    GuidanceOutput,
    GuidanceStatus,
    MergeConfig,
    MergeDecision,
    MergeResult,
    merge_guidance,
)


class ReplayMode(str, Enum):
    NONE = "none"
    GEOMETRY_ONLY = "geometry_only"
    EVENT_ONLY = "event_only"
    PARALLEL_GEOMETRY_EVENT = "parallel_geometry_event"
    SERIAL_BASELINE = "serial_baseline"


REPLAY_MODES = tuple(ReplayMode)


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ReplayBudget:
    """One shared hard budget for every ablation mode."""

    model_forwards: int = 3
    calculator_calls: int = 0
    wall_seconds: float = 1.0

    def __post_init__(self) -> None:
        if type(self.model_forwards) is not int or self.model_forwards < 0:
            raise ValueError("model_forwards must be a nonnegative integer")
        if type(self.calculator_calls) is not int or self.calculator_calls < 0:
            raise ValueError("calculator_calls must be a nonnegative integer")
        if isinstance(self.wall_seconds, bool) or not isinstance(self.wall_seconds, (int, float)) or self.wall_seconds < 0:
            raise ValueError("wall_seconds must be nonnegative")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "model_forwards": self.model_forwards,
            "calculator_calls": self.calculator_calls,
            "wall_seconds": float(self.wall_seconds),
        }


class _BudgetTracker:
    def __init__(self, budget: ReplayBudget) -> None:
        self.budget = budget
        self.used = GuidanceCost.zero()

    def reserve(self, cost: GuidanceCost) -> bool:
        candidate = GuidanceCost(
            model_forwards=self.used.model_forwards + cost.model_forwards,
            calculator_calls=self.used.calculator_calls + cost.calculator_calls,
            wall_seconds=self.used.wall_seconds + cost.wall_seconds,
            cpu_seconds=self.used.cpu_seconds + cost.cpu_seconds,
            gpu_seconds=self.used.gpu_seconds + cost.gpu_seconds,
        )
        if candidate.model_forwards > self.budget.model_forwards:
            return False
        if candidate.calculator_calls > self.budget.calculator_calls:
            return False
        if candidate.wall_seconds > self.budget.wall_seconds:
            return False
        self.used = candidate
        return True


@dataclass(frozen=True)
class ReplayStep:
    step_index: int
    mode: ReplayMode
    input_snapshot_id: str
    outputs: tuple[GuidanceOutput, ...]
    result: MergeResult

    def to_mapping(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "mode": self.mode.value,
            "input_snapshot_id": self.input_snapshot_id,
            "outputs": [output.to_mapping() for output in self.outputs],
            "decision": self.result.decision.to_mapping(),
            "next_snapshot": self.result.snapshot.to_mapping(),
            "ledger": self.result.ledger.to_mapping(),
        }


@dataclass(frozen=True)
class ReplayArtifact:
    mode: ReplayMode
    candidate_pool_id: str
    random_seed: int
    budget: ReplayBudget
    initial_snapshot: GenerationSnapshot
    steps: tuple[ReplayStep, ...]
    final_snapshot: GenerationSnapshot
    total_cost: GuidanceCost
    status: str
    claim_limits: tuple[str, ...]

    def to_mapping(self, *, include_artifact_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": "xtbflow-guidance-replay/v1",
            "mode": self.mode.value,
            "candidate_pool_id": self.candidate_pool_id,
            "random_seed": self.random_seed,
            "budget": self.budget.to_mapping(),
            "initial_snapshot": self.initial_snapshot.to_mapping(),
            "steps": [step.to_mapping() for step in self.steps],
            "final_snapshot": self.final_snapshot.to_mapping(),
            "total_cost": self.total_cost.to_mapping(),
            "status": self.status,
            "claim_limits": list(self.claim_limits),
        }
        if include_artifact_hash:
            payload["artifact_hash"] = _hash(payload)
        return payload

    @property
    def artifact_hash(self) -> str:
        return self.to_mapping(include_artifact_hash=False) and _hash(self.to_mapping(include_artifact_hash=False))


class DeterministicSyntheticProvider:
    """Pure synthetic lane provider used only by offline replay."""

    def __init__(self, lane: GuidanceLane) -> None:
        self.lane = lane

    def __call__(self, snapshot: GenerationSnapshot, *, mode: ReplayMode) -> GuidanceOutput:
        cost = GuidanceCost(model_forwards=1, wall_seconds=0.01)
        if self.lane is GuidanceLane.BASE:
            return GuidanceOutput(
                lane=self.lane,
                provider="synthetic-base-v1",
                input_fingerprint=snapshot.snapshot_id,
                estimated_cost=cost,
                measured_cost=cost,
                provenance={"replay": {"provider": "synthetic-base-v1", "mode": mode.value}},
            )
        if self.lane is GuidanceLane.GEOMETRY:
            scale = 0.04 * (snapshot.generation_step + 1)
            delta = [(0.0, 0.0, 0.0) for _ in snapshot.geometry]
            delta[0] = (scale, -scale / 2.0, 0.0)
            return GuidanceOutput(
                lane=self.lane,
                provider="synthetic-geometry-v1",
                input_fingerprint=snapshot.snapshot_id,
                geometry_delta=tuple(delta),
                trust_radius=0.12,
                estimated_cost=cost,
                measured_cost=cost,
                provenance={"replay": {"provider": "synthetic-geometry-v1", "mode": mode.value}},
            )
        scale = 0.08 * (snapshot.generation_step + 1)
        return GuidanceOutput(
            lane=self.lane,
            provider="synthetic-event-v1",
            input_fingerprint=snapshot.snapshot_id,
            event_delta=(scale, -scale / 2.0, 0.0, scale / 4.0),
            trust_radius=0.25,
            estimated_cost=cost,
            measured_cost=cost,
            provenance={"replay": {"provider": "synthetic-event-v1", "mode": mode.value}},
        )


class OfflineGuidanceReplay:
    """Run the five deterministic guidance modes from one initial snapshot."""

    def __init__(
        self,
        initial_snapshot: GenerationSnapshot,
        *,
        candidate_pool_id: str,
        budget: ReplayBudget = ReplayBudget(),
        merge_config: MergeConfig = MergeConfig(geometry_trust_radius=0.2, collision_distance=0.2),
        constraint_matrix: tuple[tuple[float, ...], ...] = ((1.0, 1.0, 1.0, 1.0),),
    ) -> None:
        if not isinstance(candidate_pool_id, str) or not candidate_pool_id.strip():
            raise ValueError("candidate_pool_id must be a non-empty string")
        if len(initial_snapshot.event) != len(initial_snapshot.event_mask):
            raise ValueError("initial snapshot event dimensions are inconsistent")
        self.initial_snapshot = initial_snapshot
        self.candidate_pool_id = candidate_pool_id
        self.budget = budget
        self.merge_config = merge_config
        self.constraint_matrix = constraint_matrix
        self.providers = {
            lane: DeterministicSyntheticProvider(lane)
            for lane in GuidanceLane
        }

    def _output_or_budget(self, snapshot: GenerationSnapshot, lane: GuidanceLane, mode: ReplayMode, tracker: _BudgetTracker) -> GuidanceOutput:
        output = self.providers[lane](snapshot, mode=mode)
        if tracker.reserve(output.measured_cost):
            return output
        return GuidanceOutput(
            lane=lane,
            provider=output.provider,
            input_fingerprint=snapshot.snapshot_id,
            estimated_cost=output.estimated_cost,
            measured_cost=GuidanceCost.zero(),
            status=GuidanceStatus.BUDGET_EXHAUSTED,
            failure_stage="budget",
            provenance={"replay": {"provider": output.provider, "mode": mode.value}},
        )

    def _merge(self, snapshot: GenerationSnapshot, mode: ReplayMode, outputs: Iterable[GuidanceOutput]) -> MergeResult:
        return merge_guidance(
            snapshot,
            tuple(outputs),
            config=self.merge_config,
            constraint_matrix=self.constraint_matrix,
            event_velocity_fn=lambda geometry, event, _old: tuple(
                geometry[0][0] + value for value in event
            ),
        )

    def run(self, mode: ReplayMode | str) -> ReplayArtifact:
        mode = mode if isinstance(mode, ReplayMode) else ReplayMode(mode)
        tracker = _BudgetTracker(self.budget)
        state = self.initial_snapshot
        steps: list[ReplayStep] = []

        def one_step(step_mode: ReplayMode, outputs: tuple[GuidanceOutput, ...]) -> None:
            nonlocal state
            result = self._merge(state, step_mode, outputs)
            steps.append(ReplayStep(len(steps), step_mode, state.snapshot_id, outputs, result))
            state = result.snapshot

        base = self._output_or_budget(state, GuidanceLane.BASE, mode, tracker)
        if mode is ReplayMode.NONE:
            one_step(mode, (base,))
        elif mode is ReplayMode.GEOMETRY_ONLY:
            geometry = self._output_or_budget(state, GuidanceLane.GEOMETRY, mode, tracker)
            one_step(mode, (base, geometry))
        elif mode is ReplayMode.EVENT_ONLY:
            event = self._output_or_budget(state, GuidanceLane.EVENT, mode, tracker)
            one_step(mode, (base, event))
        elif mode is ReplayMode.PARALLEL_GEOMETRY_EVENT:
            geometry = self._output_or_budget(state, GuidanceLane.GEOMETRY, mode, tracker)
            event = self._output_or_budget(state, GuidanceLane.EVENT, mode, tracker)
            one_step(mode, (base, geometry, event))
        else:
            one_step(mode, (base,))
            geometry = self._output_or_budget(state, GuidanceLane.GEOMETRY, mode, tracker)
            one_step(mode, (geometry,))
            event = self._output_or_budget(state, GuidanceLane.EVENT, mode, tracker)
            one_step(mode, (event,))

        status = "completed" if all(
            entry.status in {GuidanceStatus.ACCEPTED, GuidanceStatus.DUPLICATE}
            for step in steps for entry in step.result.ledger.entries
        ) else "completed_with_rejections"
        return ReplayArtifact(
            mode=mode,
            candidate_pool_id=self.candidate_pool_id,
            random_seed=self.initial_snapshot.random_seed,
            budget=self.budget,
            initial_snapshot=self.initial_snapshot,
            steps=tuple(steps),
            final_snapshot=state,
            total_cost=tracker.used,
            status=status,
            claim_limits=(
                "Synthetic offline replay proves only deterministic software state transitions and accounting.",
                "It does not establish chemistry, physical accuracy, performance, or generalization.",
            ),
        )

    def run_all(self) -> dict[ReplayMode, ReplayArtifact]:
        artifacts = {mode: self.run(mode) for mode in REPLAY_MODES}
        identity = {
            (artifact.candidate_pool_id, artifact.random_seed, json.dumps(artifact.budget.to_mapping(), sort_keys=True))
            for artifact in artifacts.values()
        }
        if len(identity) != 1:
            raise AssertionError("ablation artifacts do not share the declared inputs and budget")
        return artifacts


def write_replay_artifacts(artifacts: Mapping[ReplayMode | str, ReplayArtifact], output_dir: str) -> list[str]:
    from pathlib import Path

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for mode, artifact in artifacts.items():
        mode = mode if isinstance(mode, ReplayMode) else ReplayMode(mode)
        path = target / f"parallel_guidance_{mode.value}.json"
        path.write_text(json.dumps(artifact.to_mapping(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        paths.append(str(path))
    return paths


__all__ = [
    "ReplayMode", "REPLAY_MODES", "ReplayBudget", "ReplayStep", "ReplayArtifact",
    "DeterministicSyntheticProvider", "OfflineGuidanceReplay", "write_replay_artifacts",
]
