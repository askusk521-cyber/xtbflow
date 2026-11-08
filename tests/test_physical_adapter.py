from __future__ import annotations

import pytest

from xtbflow.sampling import (
    GuidanceBudget,
    GuidanceCost,
    GenerationSnapshot,
    MergeConfig,
    PhysicalFeedback,
    PhysicalGuidanceConfig,
    run_physical_ablation,
)


def snapshot() -> GenerationSnapshot:
    return GenerationSnapshot(
        parent_id="physical-parent",
        attempt_id="physical-attempt",
        generation_step=0,
        geometry=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        event=(0.0, 0.0, 0.0, 0.0),
        event_mask=(True, True, False, True),
        random_seed=4,
        metadata={"temperature": 298.15},
        provenance={"fixture": "physical"},
    )


def feedback(state: GenerationSnapshot) -> PhysicalFeedback:
    return PhysicalFeedback(
        snapshot_id=state.snapshot_id,
        forces=((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        reaction_direction=((0.0, 1.0, 0.0), (0.0, 0.0, 0.0)),
        curvature=((0.0, 0.0, 0.2), (0.0, 0.0, 0.0)),
        provenance={"source": "recorded-feedback"},
    )


def budget() -> GuidanceBudget:
    return GuidanceBudget(GuidanceCost(model_forwards=1, calculator_calls=1, wall_seconds=1.0))


def test_physical_off_on_replay_uses_same_snapshot_and_changes_geometry_only():
    state = snapshot()
    off = run_physical_ablation(
        state, feedback(state), enabled=False, candidate_pool_id="pool-physical", budget=budget()
    )
    on = run_physical_ablation(
        state, feedback(state), enabled=True, candidate_pool_id="pool-physical", budget=budget()
    )
    assert off.initial_snapshot.snapshot_id == on.initial_snapshot.snapshot_id
    assert off.final_snapshot.geometry == state.geometry
    assert off.final_snapshot.event == state.event
    assert off.final_snapshot.generation_step == state.generation_step + 1
    assert on.final_snapshot.geometry != state.geometry
    assert on.final_snapshot.event == state.event
    assert on.merge_result.decision.event_velocity != (0.0, 0.0, 0.0, 0.0)
    assert on.adapter_run is not None
    assert on.adapter_run.charged_cost.calculator_calls == 1
    assert on.adapter_ledger.entries[0].status.value == "accepted"


def test_physical_guidance_is_bounded_and_collision_is_handled_by_merge_gate():
    state = snapshot()
    result = run_physical_ablation(
        state,
        feedback(state),
        enabled=True,
        candidate_pool_id="pool-physical",
        budget=budget(),
        config=PhysicalGuidanceConfig(max_step=0.01, trust_radius=0.01),
        merge_config=MergeConfig(geometry_trust_radius=0.01, collision_distance=0.2),
    )
    step = sum((a - b) ** 2 for a, b in zip(result.final_snapshot.geometry[0], state.geometry[0])) ** 0.5
    assert step <= 0.01 + 1e-9
    assert result.adapter_run is not None


def test_physical_feedback_is_target_free_and_snapshot_bound():
    state = snapshot()
    with pytest.raises(ValueError, match="target-derived"):
        PhysicalFeedback(
            snapshot_id=state.snapshot_id,
            forces=((1.0, 0.0, 0.0),),
            provenance={"nested": {"reference_ts": "hidden"}},
        )
    wrong = PhysicalFeedback(
        snapshot_id="0" * 64,
        forces=((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )
    with pytest.raises(ValueError, match="feedback snapshot"):
        run_physical_ablation(state, wrong, enabled=True, candidate_pool_id="pool", budget=budget())
