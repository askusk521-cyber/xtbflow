from __future__ import annotations

import json

import pytest

from xtbflow.validation import ConnectivityEvidence, ModeEvidence, TSValidationConfig, validate_mode, validate_ts_evidence


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

