from __future__ import annotations

import math

import pytest

from xtbflow.sampling import (
    GuidanceCost,
    GuidanceLane,
    GuidanceOutput,
    GuidanceStatus,
    GenerationSnapshot,
    MergeConfig,
    merge_guidance,
    project_masked_event_delta,
)


CONSTRAINT = ((1.0, 1.0, 1.0, 1.0),)


def snapshot(*, generation_step: int = 0, event_frozen: bool = False) -> GenerationSnapshot:
    return GenerationSnapshot(
        parent_id="parent-1",
        attempt_id="attempt-1",
        generation_step=generation_step,
        geometry=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        event=(0.0, 0.0, 0.0, 0.0),
        event_mask=(True, True, False, True),
        random_seed=7,
        metadata={"temperature": 298.15, "safe": {"source": "reactant-only"}},
        provenance={"provider_revision": "fixture-v1"},
        event_frozen=event_frozen,
    )


def output(state: GenerationSnapshot, lane: GuidanceLane, provider: str, **kwargs) -> GuidanceOutput:
    return GuidanceOutput(
        lane=lane,
        provider=provider,
        input_fingerprint=state.snapshot_id,
        estimated_cost=GuidanceCost(model_forwards=1),
        measured_cost=GuidanceCost(model_forwards=1),
        **kwargs,
    )


def velocity(geometry, event, _snapshot):
    # The callback intentionally depends on the post-merge geometry so a
    # stale pre-geometry event velocity is observable in the contract test.
    return tuple(geometry[0][0] + value for value in event)


def test_snapshot_is_immutable_and_firewall_is_recursive_and_normalised():
    state = snapshot()
    original = state.snapshot_id
    assert state.to_mapping()["snapshot_id"] == original
    assert state.metadata == (("safe", (("source", "reactant-only"),)), ("temperature", 298.15))
    with pytest.raises(ValueError, match="target-derived"):
        GenerationSnapshot(
            parent_id="p", attempt_id="a", generation_step=0,
            geometry=((0.0, 0.0, 0.0),), event=(0.0,), event_mask=(True,), random_seed=1,
            metadata={"nested": [{"TS_GEOMETRY": "hidden"}]},
        )
    with pytest.raises(ValueError, match="target-derived"):
        output(state, GuidanceLane.EVENT, "policy", event_delta=(0.0, 0.0, 0.0, 0.0), provenance={"sealedTestLabel": "x"})


def test_projection_preserves_padding_and_conservation_residual():
    projected, residual = project_masked_event_delta((0.2, 0.4, 99.0, 0.0), (True, True, False, True), CONSTRAINT)
    assert projected[2] == 0.0
    assert sum(projected) == pytest.approx(0.0, abs=1e-9)
    assert max(abs(value) for value in residual) <= 1e-9


def test_parallel_merge_is_order_independent_and_recomputes_velocity_after_geometry():
    state = snapshot()
    base = output(
        state, GuidanceLane.BASE, "joint-flow",
        geometry_delta=((0.1, 0.0, 0.0), (0.0, 0.0, 0.0)),
        event_delta=(0.2, 0.0, 0.0, 0.0),
    )
    geometry = output(
        state, GuidanceLane.GEOMETRY, "physics",
        geometry_delta=((0.0, 0.1, 0.0), (0.0, 0.0, 0.0)),
    )
    event = output(
        state, GuidanceLane.EVENT, "policy",
        event_delta=(0.0, 0.3, 0.0, 0.0),
    )
    config = MergeConfig(geometry_trust_radius=1.0)
    first = merge_guidance(state, (event, geometry, base), config=config, constraint_matrix=CONSTRAINT, event_velocity_fn=velocity)
    second = merge_guidance(state, (base, geometry, event), config=config, constraint_matrix=CONSTRAINT, event_velocity_fn=velocity)
    assert first.snapshot == second.snapshot
    assert first.decision == second.decision
    assert first.decision.geometry_after != state.geometry
    assert first.decision.event_after != state.event
    assert first.decision.event_velocity == velocity(first.decision.geometry_after, first.decision.event_after, state)
    assert first.decision.conservation_residual[0] == pytest.approx(0.0, abs=1e-9)
    assert first.decision.lane_order == ("base:joint-flow", "geometry:physics", "event:policy")
    assert set(first.decision.applied_lanes) == {"base", "geometry", "event"}


def test_trust_radius_and_collision_guard_are_recorded_as_rejections_or_bounded_updates():
    state = snapshot()
    bounded = output(
        state, GuidanceLane.GEOMETRY, "physics",
        geometry_delta=((10.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        trust_radius=0.2,
    )
    result = merge_guidance(state, (bounded,), config=MergeConfig(geometry_trust_radius=0.2))
    step = math.sqrt(sum((a - b) ** 2 for a, b in zip(result.snapshot.geometry[0], state.geometry[0])))
    assert step == pytest.approx(0.2)

    colliding = output(
        state, GuidanceLane.GEOMETRY, "physics",
        geometry_delta=((0.0, 0.0, 0.0), (-0.9, 0.0, 0.0)),
    )
    rejected = merge_guidance(state, (colliding,), config=MergeConfig(geometry_trust_radius=1.0, collision_distance=0.2))
    assert rejected.snapshot.geometry == state.geometry
    assert rejected.decision.rejected == (("geometry", "collision_guard"),)
    assert rejected.ledger.entries[0].applied is False


def test_freeze_boundary_rejects_event_edit_but_keeps_other_lanes_accounted():
    state = snapshot(generation_step=3)
    geometry = output(
        state, GuidanceLane.GEOMETRY, "physics",
        geometry_delta=((0.1, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )
    event = output(state, GuidanceLane.EVENT, "policy", event_delta=(0.2, 0.0, 0.0, 0.0))
    result = merge_guidance(state, (event, geometry), config=MergeConfig(event_freeze_step=3), constraint_matrix=CONSTRAINT)
    assert result.snapshot.geometry != state.geometry
    assert result.snapshot.event == state.event
    assert result.decision.event_frozen is True
    assert ("event", "event_frozen") in result.decision.rejected
    event_entries = [entry for entry in result.ledger.entries if entry.lane is GuidanceLane.EVENT]
    assert event_entries[0].charged_cost == event.measured_cost


def test_failed_lane_does_not_block_valid_lane_and_duplicate_is_not_billed_twice():
    state = snapshot()
    failed = output(
        state, GuidanceLane.EVENT, "policy",
        status=GuidanceStatus.FAILED,
        failure_stage="provider",
    )
    geometry = output(
        state, GuidanceLane.GEOMETRY, "physics",
        geometry_delta=((0.1, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )
    duplicate = output(
        state, GuidanceLane.GEOMETRY, "physics",
        geometry_delta=((0.1, 0.0, 0.0), (0.0, 0.0, 0.0)),
        measured_cost=GuidanceCost(model_forwards=9),
    )
    result = merge_guidance(state, (duplicate, failed, geometry), config=MergeConfig())
    assert result.snapshot.geometry != state.geometry
    assert any(entry.status is GuidanceStatus.FAILED for entry in result.ledger.entries)
    duplicate_entries = [entry for entry in result.ledger.entries if entry.status is GuidanceStatus.DUPLICATE]
    assert len(duplicate_entries) == 1
    assert duplicate_entries[0].charged_cost == GuidanceCost.zero()


def test_lane_contract_rejects_cross_lane_payloads_and_requires_projection_for_event_edits():
    state = snapshot()
    with pytest.raises(ValueError, match="geometry lane cannot return an event delta"):
        output(state, GuidanceLane.GEOMETRY, "bad", geometry_delta=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), event_delta=(0.0, 0.0, 0.0, 0.0))
    event = output(state, GuidanceLane.EVENT, "policy", event_delta=(0.2, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="constraint_matrix is required"):
        merge_guidance(state, (event,))
