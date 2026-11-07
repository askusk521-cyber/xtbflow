"""Statistics/accounting fixtures; these do not establish chemical validity."""
from dataclasses import replace
from pathlib import Path
import json

import pytest

from xtbflow.evaluation.discovery import (
    COST_AXES, TRAINING_AXES, Certificate, DiscoveryAttempt, DiscoveryTrial,
    add_costs, paired_comparison, summarize_trial,
)
from xtbflow.evaluation.certified_trace import read_reference_attempt
from xtbflow.validation.certification import CertificationService
from xtbflow.evaluation.comparison_matrix import functional_dispatch, validate_matrix
from test_certification_service import FixtureBackend, budget, candidate, run


def costs(**changes):
    return {**dict.fromkeys(COST_AXES, 0), **changes}


def attempt(name="candidate", event="observed", *, status="completed", **cost):
    certificate = Certificate("a"*64, "b"*64, "valid_alternative", event) if event else None
    return DiscoveryAttempt(name, status, costs(raw_proposals=1, **cost), certificate)


def trial(*, parent="p", method="joint", attempts=(), seed=0, **changes):
    values = dict(parent_id=parent, input_sha256="c"*64, formula_group="C2H6|q0|m1", family_group="family",
        method=method, seed=seed, information_condition="reactant_only", protocol_sha256="b"*64,
        budget=dict.fromkeys(COST_AXES, 10000), status="completed", attempts=attempts,
        training_cost=dict.fromkeys(TRAINING_AXES, None))
    values.update(changes)
    return DiscoveryTrial(**values)


def test_unique_observed_channels_deduplicate_geometry_and_cache():
    t = trial(attempts=[attempt("a", event="alternative", gpu_seconds=10),
                       attempt("b", event="alternative", gpu_seconds=12),
                       attempt("cached", event="alternative", cache_hits=1)])
    row = summarize_trial(t, axis="gpu_seconds", grid=[0, 10, 22])
    assert [p["unique_certified_channels"] for p in row["curve"]] == [0, 1, 1]
    assert row["first_certification"]["cost"]["gpu_seconds"] == 10
    assert row["inference_cost"]["gpu_seconds"] == 22
    assert not row["full_cost_known"]


def test_failed_and_unattempted_trials_remain_censored_and_in_denominator():
    failed = trial(attempts=[attempt(event=None, status="execution_timeout", gpu_seconds=30)])
    unattempted = trial(parent="unattempted", status="not_started_budget_exhausted")
    for value, consumed in ((failed, 30), (unattempted, 0)):
        row = summarize_trial(value, axis="gpu_seconds", grid=[100])
        assert row["first_certification_right_censored"] and row["first_certification"] is None
        assert row["censor_at_cost"]["gpu_seconds"] == consumed
        assert row["curve"][0]["unique_certified_channels"] == 0


def test_missing_cost_propagates_and_cannot_make_a_free_success():
    row = summarize_trial(trial(attempts=[attempt(gpu_seconds=None)]), axis="gpu_seconds", grid=[100])
    assert row["curve"][0]["unique_certified_channels"] is None
    assert add_costs([costs(gpu_seconds=None), costs(gpu_seconds=10)])["gpu_seconds"] is None
    with pytest.raises(ValueError, match="every cost"):
        DiscoveryAttempt("q", "completed", {"gpu_seconds": 0}, None)


def test_certificate_after_any_budget_limit_is_not_a_budgeted_discovery():
    caps = dict.fromkeys(COST_AXES, 10000)
    caps["reference_primitive_calls"] = 3
    row = summarize_trial(trial(budget=caps, attempts=[attempt(reference_primitive_calls=4, gpu_seconds=2)]),
                          axis="gpu_seconds", grid=[10])
    assert row["observed_unique_channels_all_spending"] == 1
    assert row["unique_certified_channels"] == 0
    assert row["curve"][0]["unique_certified_channels"] == 0
    assert row["first_certification_right_censored"]
    assert row["over_budget_axes"] == ["reference_primitive_calls"]


def test_pairing_rejects_missing_cells_hidden_information_and_unequal_budgets():
    a, b = trial(method="joint"), trial(method="serial")
    opts = dict(axis="gpu_seconds", threshold=100, cluster="parent_id", bootstrap_samples=100,
                bootstrap_seed=0, minimum_practical_gain=.1)
    with pytest.raises(ValueError, match="paired cells"):
        paired_comparison([a, replace(a, seed=1)], [b], **opts)
    for changed in (replace(b, information_condition="known_product"), replace(b, input_sha256="d"*64),
                    replace(b, budget={**b.budget, "raw_proposals": 99})):
        with pytest.raises(ValueError, match="mismatch"):
            paired_comparison([a], [changed], **opts)


def test_parent_seed_average_and_whole_group_resampling():
    left, right = [], []
    for parent, successes in (("p1", [True, False]), ("p2", [True, True])):
        for seed, success in enumerate(successes):
            left.append(trial(parent=parent, seed=seed, method="joint",
                              attempts=[attempt(event="e" if success else None)]))
            right.append(trial(parent=parent, seed=seed, method="serial", attempts=[attempt(event=None)]))
    opts = dict(axis="raw_proposals", threshold=10, bootstrap_samples=200, bootstrap_seed=44, minimum_practical_gain=.1)
    result = paired_comparison(left, right, cluster="parent_id", **opts)
    assert result["mean_difference"] == .75
    assert result["parent_differences"] == {"p1": .5, "p2": 1.}
    assert result == paired_comparison(left, right, cluster="parent_id", **opts)
    # One shared family is one independent unit, not four seeds or two parents.
    family = paired_comparison(left, right, cluster="family_group", **opts)
    assert family["status"] == "descriptive_only" and family["interval95"] is None


def test_unknown_group_or_cost_never_gets_an_interval():
    opts = dict(axis="gpu_seconds", threshold=100, cluster="family_group", bootstrap_samples=100,
                bootstrap_seed=0, minimum_practical_gain=.1)
    a = trial(method="joint", family_group=None)
    b = trial(method="serial", family_group=None)
    with pytest.raises(ValueError, match="unknown"):
        paired_comparison([a], [b], **opts)
    a = trial(method="joint", attempts=[attempt(gpu_seconds=None)])
    b = trial(method="serial", attempts=[attempt(gpu_seconds=5)])
    row = paired_comparison([a], [b], **opts)
    assert row["status"] == "cost_unresolved" and row["interval95"] is None


def test_real_service_adapter_reconciles_failure_cost_and_rejects_tampering(tmp_path):
    service = CertificationService(tmp_path / "service", budget())
    result = run(service, backend=FixtureBackend(mode="failure"))
    work = service.root / result["record"]["cache_key"] / "attempt-0001"
    loaded, provenance = read_reference_attempt(work)
    assert loaded.cost["reference_primitive_calls"] == 3
    assert loaded.cost["energy_calls"] == 2 and loaded.cost["force_calls"] == 1
    assert provenance["unfinished_primitives"] == 1
    assert loaded.certificate.evidence_status == "attempted_unresolved"
    (work / "raw/calls.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact mismatch"):
        read_reference_attempt(work)


def test_method_matrix_dispatches_actual_pair_and_rejects_missing_strong_controls():
    root = Path(__file__).resolve().parents[1]
    matrix = json.loads((root / "configs/evaluation/comparison_matrix_20261107.json").read_text())
    profile = json.loads((root / "configs/research/generation_bridge_20261107.json").read_text())
    ready = validate_matrix(matrix, root)
    assert not ready["full_experiment_ready"]
    dispatch = functional_dispatch(matrix, root, profile, profile["methods"])
    assert not dispatch["equal_total_budget_comparison"]
    with pytest.raises(ValueError):
        functional_dispatch(matrix, root, profile, ["serial_physics_backtracking", "joint_generation"])
