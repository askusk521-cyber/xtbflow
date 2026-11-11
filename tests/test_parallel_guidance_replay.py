from __future__ import annotations

import json

import pytest

from xtbflow.sampling import (
    GenerationSnapshot,
    OfflineGuidanceReplay,
    ReplayBudget,
    ReplayMode,
    REPLAY_MODES,
    write_replay_artifacts,
)


def make_replay(*, seed: int = 166, budget: ReplayBudget | None = None) -> OfflineGuidanceReplay:
    state = GenerationSnapshot(
        parent_id="synthetic-parent",
        attempt_id="synthetic-attempt",
        generation_step=0,
        geometry=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        event=(0.0, 0.0, 0.0, 0.0),
        event_mask=(True, True, False, True),
        random_seed=seed,
        metadata={"temperature": 298.15, "safe": {"source": "synthetic"}},
        provenance={"provider_revision": "test-v1"},
    )
    return OfflineGuidanceReplay(
        state,
        candidate_pool_id="pool-v1",
        budget=budget or ReplayBudget(),
    )


def test_all_modes_share_inputs_and_parallel_changes_geometry_and_event():
    replay = make_replay()
    artifacts = replay.run_all()
    assert tuple(artifacts) == REPLAY_MODES
    identities = {
        (artifact.candidate_pool_id, artifact.random_seed, artifact.budget)
        for artifact in artifacts.values()
    }
    assert len(identities) == 1
    parallel = artifacts[ReplayMode.PARALLEL_GEOMETRY_EVENT]
    assert parallel.final_snapshot.geometry != parallel.initial_snapshot.geometry
    assert parallel.final_snapshot.event != parallel.initial_snapshot.event
    assert parallel.steps[0].result.decision.applied_lanes == ("base", "geometry", "event")
    assert parallel.total_cost.model_forwards == 3
    assert parallel.total_cost.model_forwards <= parallel.budget.model_forwards
    assert parallel.total_cost.calculator_calls == 0


def test_replay_is_deterministic_and_serial_has_explicit_substeps():
    first = make_replay().run_all()
    second = make_replay().run_all()
    assert {mode: artifact.artifact_hash for mode, artifact in first.items()} == {
        mode: artifact.artifact_hash for mode, artifact in second.items()
    }
    assert len(first[ReplayMode.SERIAL_BASELINE].steps) == 3
    assert first[ReplayMode.NONE].final_snapshot.geometry == first[ReplayMode.NONE].initial_snapshot.geometry
    assert first[ReplayMode.NONE].final_snapshot.event == first[ReplayMode.NONE].initial_snapshot.event


def test_budget_exhaustion_is_recorded_without_blocking_artifact_generation():
    artifact = make_replay(budget=ReplayBudget(model_forwards=1)).run(ReplayMode.PARALLEL_GEOMETRY_EVENT)
    assert artifact.status == "completed_with_rejections"
    statuses = [entry.status.value for step in artifact.steps for entry in step.result.ledger.entries]
    assert statuses.count("accepted") == 1
    assert statuses.count("budget_exhausted") == 2
    assert artifact.total_cost.model_forwards == 1
    assert artifact.final_snapshot.geometry == artifact.initial_snapshot.geometry
    assert artifact.final_snapshot.event == artifact.initial_snapshot.event


def test_artifacts_are_json_reconstructable(tmp_path):
    artifacts = make_replay().run_all()
    paths = write_replay_artifacts(artifacts, str(tmp_path))
    assert len(paths) == 5
    for path in paths:
        payload = json.loads(open(path, encoding="utf-8").read())
        assert payload["schema"] == "xtbflow-guidance-replay/v1"
        assert len(payload["artifact_hash"]) == 64
        assert payload["claim_limits"]


def test_modes_use_same_seed_but_seed_change_changes_artifact():
    first = make_replay(seed=166).run(ReplayMode.PARALLEL_GEOMETRY_EVENT)
    second = make_replay(seed=167).run(ReplayMode.PARALLEL_GEOMETRY_EVENT)
    assert first.random_seed != second.random_seed
    assert first.artifact_hash != second.artifact_hash
