from __future__ import annotations

import json

import pytest

from xtbflow.validation import ConnectivityEvidence, ModeEvidence, TSValidationConfig, run_resumable_validation, validate_mode, validate_ts_evidence


def connectivity(path=True):
    return ConnectivityEvidence(((0, 1, 0), (1, 0, 1), (0, 1, 0)), ((0, 0, 0), (0, 0, 1), (0, 1, 0)))


def good(path=True):
    return {"converged": True, "gradient_norm": 1e-6, "hessian_eigenvalues": (-0.5, 0.2, 0.3), "connectivity": connectivity(path), "path_connected": path}


def test_ts_success_requires_mode_and_bidirectional_connectivity():
    record = validate_ts_evidence("candidate-1", "fixture", good(), TSValidationConfig())
    assert record.status == "success"
    assert record.path_status == "validated"
    assert record.observed_event == ((0, 1, -1),)


def test_low_gradient_minimum_and_disconnected_path_fail_closed():
    minimum = good()
    minimum["hessian_eigenvalues"] = (0.01, 0.2, 0.3)
    assert validate_ts_evidence("minimum", "fixture", minimum, TSValidationConfig()).status == "failure"
    disconnected = validate_ts_evidence("disconnected", "fixture", good(False), TSValidationConfig())
    assert disconnected.status == "failure"
    assert disconnected.path_status == "not_validated"


def test_missing_connectivity_and_mode_threshold_are_explicit():
    missing = good()
    del missing["connectivity"]
    record = validate_ts_evidence("missing", "fixture", missing, TSValidationConfig())
    assert record.status == "failure"
    with pytest.raises(ValueError):
        validate_mode(ModeEvidence((-0.1, 0.1)), negative_threshold=0.0)


def test_ts_validation_rejects_string_booleans():
    evidence = good()
    evidence["converged"] = "false"
    assert validate_ts_evidence("string-bool", "fixture", evidence, TSValidationConfig()).status == "failure"
    evidence = good()
    evidence["path_connected"] = "false"
    assert validate_ts_evidence("string-path", "fixture", evidence, TSValidationConfig()).status == "failure"


def test_resumable_validation_binds_cache_to_candidate_and_checks_budget_before_search(tmp_path):
    checkpoint = tmp_path / "ts.json"
    calls = []
    order = []

    def reserve(candidate_id, candidate):
        order.append(("reserve", candidate_id))

    def search(candidate_id, candidate):
        order.append(("search", candidate_id))
        calls.append(candidate_id)
        evidence = good()
        evidence["calculator_calls"] = 1
        evidence["source"] = "fixture"
        return evidence

    candidates = {"candidate": {"coordinates": [[0.0, 0.0, 0.0]], "charge": 0, "multiplicity": 1}}
    first = run_resumable_validation(candidates, search, TSValidationConfig(), checkpoint, reserve_calls=reserve)
    assert first[0].status == "success"
    assert order == [("reserve", "candidate"), ("search", "candidate")]
    second = run_resumable_validation(candidates, search, TSValidationConfig(), checkpoint, reserve_calls=reserve)
    assert second[0].status == "success"
    assert calls == ["candidate"]
    changed = {"candidate": {"coordinates": [[0.1, 0.0, 0.0]], "charge": 0, "multiplicity": 1}}
    run_resumable_validation(changed, search, TSValidationConfig(), checkpoint, reserve_calls=reserve)
    assert calls == ["candidate", "candidate"]
