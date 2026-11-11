from __future__ import annotations

import json

from xtbflow.sampling import MATCHED_ARMS, MatchedArm, MatchedComparisonConfig, run_matched_comparison


def test_matched_comparison_preserves_parent_denominator_and_shared_inputs():
    config = MatchedComparisonConfig(parent_ids=("p1", "p2"))
    artifact = run_matched_comparison(config)
    assert len(artifact.rows) == len(config.parent_ids) * len(MATCHED_ARMS)
    assert len({row.snapshot_id for row in artifact.rows if row.parent_id == "p1"}) == 1
    assert {row.seed for row in artifact.rows} == {config.random_seed}
    assert {row.calculator_calls for row in artifact.rows} <= {0, 1}
    assert artifact.to_mapping()["parent_denominator"] == 2


def test_matched_arms_separate_geometry_event_and_policy_effects():
    artifact = run_matched_comparison(MatchedComparisonConfig(parent_ids=("p1",)))
    rows = {row.arm: row for row in artifact.rows}
    assert rows[MatchedArm.NONE].geometry_changed is False
    assert rows[MatchedArm.NONE].event_changed is False
    assert rows[MatchedArm.SERIAL_PHYSICAL].geometry_changed is True
    assert rows[MatchedArm.SERIAL_PHYSICAL].event_changed is False
    assert rows[MatchedArm.PARALLEL_GEOMETRY_EVENT].geometry_changed is True
    assert rows[MatchedArm.PARALLEL_GEOMETRY_EVENT].event_changed is True
    assert rows[MatchedArm.POLICY_ON].geometry_changed is False
    assert rows[MatchedArm.POLICY_ON].event_changed is True
    assert rows[MatchedArm.POLICY_OFF].geometry_changed is False
    assert rows[MatchedArm.POLICY_OFF].event_changed is False


def test_matched_artifact_is_deterministic_and_costs_are_separated():
    config = MatchedComparisonConfig(parent_ids=("p1", "p2"))
    first = run_matched_comparison(config).to_mapping()
    second = run_matched_comparison(config).to_mapping()
    assert first["artifact_hash"] == second["artifact_hash"]
    parallel = next(row for row in first["rows"] if row["arm"] == "parallel_geometry_event")
    assert parallel["model_forwards"] == 2
    assert parallel["calculator_calls"] == 1
    assert parallel["adapter_ledger"]["entries"]
    json.dumps(first, allow_nan=False)
