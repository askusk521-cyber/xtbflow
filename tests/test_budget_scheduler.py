from __future__ import annotations

import pytest

from xtbflow.runtime import (
    BudgetCandidate,
    BudgetScheduler,
    SchedulerBudgetExceeded,
)


def candidate(
    name: str,
    *,
    cost: int = 1,
    gain: float = 0.0,
    priority: int = 0,
    channel: str | None = None,
) -> BudgetCandidate:
    return BudgetCandidate(
        candidate_id=name,
        family_id=f"family-{name}",
        action_kind="new_event",
        estimated_cost_units=cost,
        information_gain=gain,
        priority=priority,
        channel_key=channel,
        metadata={"source": "fixture"},
    )


def test_fixed_schedule_preserves_order_and_stops_at_first_over_budget():
    scheduler = BudgetScheduler(
        [candidate("a", cost=2, channel="a"), candidate("b", cost=2, channel="b")],
        budget_units=3,
        strategy="fixed",
    )
    first = scheduler.next_action()
    assert first is not None and first.candidate.candidate_id == "a"
    scheduler.settle(first, "success", 2)
    assert scheduler.next_action() is None
    assert scheduler.status == "budget_exhausted"
    assert scheduler.pending_candidates[0].candidate_id == "b"
    assert scheduler.summary()["spent_units"] == 2
    assert scheduler.summary()["remaining_units"] == 1


def test_heuristic_prefers_information_per_cost_with_deterministic_ties():
    scheduler = BudgetScheduler(
        [
            candidate("z", cost=3, gain=3.0, priority=1),
            candidate("a", cost=1, gain=1.0, priority=0),
            candidate("low", cost=1, gain=0.1),
        ],
        budget_units=5,
        strategy="heuristic",
    )
    action = scheduler.next_action()
    assert action is not None
    assert action.candidate.candidate_id == "a"
    scheduler.settle(action, "failure", 1, reason="fixture_failure")
    next_action = scheduler.next_action()
    assert next_action is not None and next_action.candidate.candidate_id == "z"


def test_accepted_channel_is_deduplicated_without_charging_budget():
    scheduler = BudgetScheduler(
        [candidate("first", cost=2, channel="same"), candidate("repeat", cost=5, channel="same"), candidate("other", channel="other")],
        budget_units=3,
    )
    action = scheduler.next_action()
    assert action is not None
    scheduler.settle(action, "success", 2)
    other = scheduler.next_action()
    assert other is not None and other.candidate.candidate_id == "other"
    assert scheduler.records[1].status == "duplicate"
    assert scheduler.records[1].charged_cost_units == 0
    scheduler.settle(other, "success", 1)
    assert scheduler.next_action() is None
    assert scheduler.status == "completed"
    assert scheduler.summary()["duplicate_records"] == 1


def test_failures_and_unresolved_attempts_remain_in_the_cost_ledger():
    scheduler = BudgetScheduler([candidate("bad", cost=3), candidate("later")], budget_units=5)
    action = scheduler.next_action()
    assert action is not None
    scheduler.settle(action, "timeout", 2, reason="wall_time_cap")
    assert scheduler.summary()["failure_records"] == 1
    assert scheduler.summary()["failure_cost_units"] == 2
    later = scheduler.next_action()
    assert later is not None and later.candidate.candidate_id == "later"
    scheduler.settle(later, "unresolved", 1, reason="protocol_failure")
    assert scheduler.summary()["failure_records"] == 2
    assert scheduler.summary()["failure_cost_units"] == 3


def test_stop_resume_and_snapshot_preserve_an_in_flight_action(tmp_path):
    scheduler = BudgetScheduler([candidate("a"), candidate("b")], budget_units=4)
    action = scheduler.next_action()
    assert action is not None
    scheduler.stop("awaiting external result")
    assert scheduler.next_action() is None
    path = tmp_path / "scheduler.json"
    scheduler.save(path)
    restored = BudgetScheduler.load(path)
    assert restored.resume() == "running"
    resumed = restored.next_action()
    assert resumed == action
    restored.settle(resumed, "success", 1, channel_key="channel-a")
    assert restored.records[0].candidate_id == "a"


def test_settle_rejects_cost_overrun_without_mutating_in_flight_action():
    scheduler = BudgetScheduler([candidate("a", cost=2)], budget_units=2)
    action = scheduler.next_action()
    assert action is not None
    with pytest.raises(SchedulerBudgetExceeded):
        scheduler.settle(action, "failure", 3, reason="unexpected_overrun")
    assert scheduler.in_flight == action
    assert scheduler.spent_units == 0


def test_candidate_contract_rejects_invalid_cost_and_non_json_metadata():
    with pytest.raises(ValueError, match="estimated_cost_units"):
        candidate("bad", cost=0)
    with pytest.raises(ValueError, match="finite JSON"):
        BudgetCandidate("bad", "family", "new_event", 1, metadata={"x": float("nan")})
