from __future__ import annotations

import pytest

from xtbflow.evaluation.grouped import paired_group_comparison, summarize_by_group


ROWS = [
    {"parent_reaction_id": "p1", "metric": 1.0},
    {"parent_reaction_id": "p1", "metric": 3.0},
    {"parent_reaction_id": "p2", "metric": 5.0},
]


def test_summary_keeps_record_and_parent_weighting_distinct():
    summary = summarize_by_group(ROWS, metric_keys=["metric"])
    assert summary["record_count"] == 3
    assert summary["group_count"] == 2
    assert summary["record_weighted"]["metric"] == pytest.approx(3.0)
    assert summary["parent_equal"]["metric"] == pytest.approx(3.5)
    assert [(row["group_id"], row["record_count"]) for row in summary["groups"]] == [("p1", 2), ("p2", 1)]


def test_paired_comparison_bootstraps_parent_groups_deterministically():
    right = [
        {"parent_reaction_id": "p1", "metric": 2.0},
        {"parent_reaction_id": "p1", "metric": 4.0},
        {"parent_reaction_id": "p2", "metric": 4.0},
    ]
    first = paired_group_comparison(ROWS, right, metric_keys=["metric"], bootstrap_samples=128, seed=17)
    second = paired_group_comparison(ROWS, right, metric_keys=["metric"], bootstrap_samples=128, seed=17)
    assert first == second
    assert first["paired_group_count"] == 2
    # Parent means are p1=2 vs 3 and p2=5 vs 4, so left-right averages to 0.
    assert first["metrics"]["metric"]["left_minus_right_mean"] == pytest.approx(0.0)
    assert first["groups"][0]["difference"]["metric"] == pytest.approx(-1.0)


def test_missing_or_unmatched_groups_are_explicit():
    with pytest.raises(ValueError, match="no common parent groups"):
        paired_group_comparison(ROWS, [{"parent_reaction_id": "other", "metric": 1.0}], metric_keys=["metric"])
    with pytest.raises(ValueError, match="missing metric"):
        summarize_by_group([{"parent_reaction_id": "p1"}], metric_keys=["metric"])
