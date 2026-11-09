from __future__ import annotations

import pytest

from xtbflow.sampling import (
    AdapterRunner,
    GuidanceBudget,
    GuidanceCost,
    GuidanceLane,
    GenerationSnapshot,
    PolicyFeedback,
    RecordedPolicyProvider,
    ProviderRequest,
    run_policy_ablation,
    EventAction,
    EventEdit,
)


def snapshot(*, frozen: bool = False) -> GenerationSnapshot:
    return GenerationSnapshot(
        parent_id="policy-parent",
        attempt_id="policy-attempt",
        generation_step=0,
        geometry=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        event=(0.0, 0.0, 0.0, 0.0),
        event_mask=(True, True, False, True),
        random_seed=9,
        metadata={"temperature": 298.15},
        provenance={"fixture": "policy"},
        event_frozen=frozen,
    )


def feedback(state: GenerationSnapshot, action_id: str = "action-a") -> PolicyFeedback:
    return PolicyFeedback(
        snapshot_id=state.snapshot_id,
        ledger={"previous_cost": 0.1},
        tool_feedback={"ranking_signal": 0.5},
        action_id=action_id,
        action_space={"allowed": ["event_delta"]},
        provenance={"source": "recorded-policy-feedback"},
    )


def budget() -> GuidanceBudget:
    return GuidanceBudget(GuidanceCost(model_forwards=1, wall_seconds=1.0))


def test_policy_off_on_changes_event_only_and_preserves_geometry():
    state = snapshot()
    off = run_policy_ablation(state, feedback(state), enabled=False, candidate_pool_id="pool-policy", budget=budget())
    on = run_policy_ablation(state, feedback(state), enabled=True, candidate_pool_id="pool-policy", budget=budget())
    assert off.final_snapshot.geometry == state.geometry
    assert off.final_snapshot.event == state.event
    assert on.final_snapshot.geometry == state.geometry
    assert on.final_snapshot.event != state.event
    assert on.adapter_run is not None
    assert on.adapter_run.charged_cost.model_forwards == 1
    assert on.adapter_ledger.entries[0].status.value == "accepted"


def test_policy_freeze_is_explicitly_rejected_by_merge_gate():
    state = snapshot(frozen=True)
    result = run_policy_ablation(state, feedback(state), enabled=True, candidate_pool_id="pool-policy", budget=budget())
    assert result.final_snapshot.event == state.event
    assert ("event", "event_frozen") in result.merge_result.decision.rejected
    assert result.adapter_run is not None
    assert result.adapter_run.status.value == "accepted"


def test_policy_action_duplicate_is_not_charged_twice():
    state = snapshot()
    policy_feedback = feedback(state, "same-action")
    provider = RecordedPolicyProvider(policy_feedback)
    request = ProviderRequest(
        snapshot=state,
        lane=GuidanceLane.EVENT,
        provider_revision=provider.provider_revision,
        config_fingerprint=provider.config_fingerprint,
        protocol_fingerprint=provider.protocol_fingerprint,
        feedback={"feedback_fingerprint": policy_feedback.feedback_fingerprint, "action_id": policy_feedback.action_id},
        action_space={"action_id": policy_feedback.action_id, **dict(policy_feedback.action_space)},
    )
    runner = AdapterRunner(budget())
    first = runner.run(request, provider)
    second = runner.run(request, provider)
    assert first.status.value == "accepted"
    assert second.status.value == "duplicate"
    assert second.charged_cost == GuidanceCost.zero()
    assert provider.invocations == 1


def test_policy_feedback_firewall_and_snapshot_binding():
    state = snapshot()
    with pytest.raises(ValueError, match="target-derived"):
        PolicyFeedback(snapshot_id=state.snapshot_id, ledger={"nested": {"reference_ts": "hidden"}})
    with pytest.raises(ValueError, match="policy feedback snapshot"):
        run_policy_ablation(snapshot(), PolicyFeedback(snapshot_id="0" * 64), enabled=True, candidate_pool_id="pool", budget=budget())


def test_policy_action_supports_complete_multi_edit_event():
    state = snapshot()
    action = EventAction(
        edits=(EventEdit("add", 1, 2), EventEdit("delete", 3, 4)),
        action_id="complete-event-v1",
    )
    policy_feedback = PolicyFeedback(
        snapshot_id=state.snapshot_id,
        action_id=action.action_id,
        action_space={"allowed": ["event_action"], "packed_feature_order": ["add:1-2", "add:3-4", "delete:1-2", "delete:3-4"]},
        event_action=action,
        provenance={"source": "recorded-policy-contract-test"},
    )
    result = run_policy_ablation(state, policy_feedback, enabled=True, candidate_pool_id="pool-policy", budget=budget())
    assert result.adapter_run is not None
    assert result.adapter_run.status.value == "accepted"
    assert result.final_snapshot.event != state.event
    output = result.adapter_run.output
    assert output is not None
    assert output.event_delta == (0.04, 0.0, 0.0, -0.04)
    assert output.to_mapping()["provenance"]["provider"]["policy"]["policy_action"]["fingerprint"] == action.fingerprint


def test_event_action_rejects_duplicate_or_noncanonical_edits():
    with pytest.raises(ValueError, match="canonicalized"):
        EventEdit("add", 4, 3)
    with pytest.raises(ValueError, match="duplicate"):
        EventAction(edits=(EventEdit("add", 1, 2), EventEdit("add", 1, 2)))
