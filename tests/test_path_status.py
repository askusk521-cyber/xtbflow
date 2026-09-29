from __future__ import annotations

import pytest

from xtbflow.runtime import RunLedger, StageBudget
from xtbflow.validation import (
    ConnectivityEvidence,
    ModeEvidence,
    TSValidationConfig,
    run_resumable_validation,
    validate_mode,
    validate_ts_evidence,
)


def connectivity(path=True):
    return ConnectivityEvidence(((0, 1, 0), (1, 0, 1), (0, 1, 0)), ((0, 0, 0), (0, 0, 1), (0, 1, 0)))


def good(path=True):
    return {
        "converged": True,
        "gradient_norm": 1e-6,
        "hessian_eigenvalues": (-0.5, 0.2, 0.3),
        "connectivity": connectivity(path),
        "path_connected": path,
    }


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


def test_strict_validation_requires_token_factory_and_binds_calls(tmp_path):
    config = TSValidationConfig(max_calls=2)
    candidate = {"coordinates": [[0.0, 0.0, 0.0]], "charge": 0, "multiplicity": 1}
    called = []
    with pytest.raises(ValueError, match="budget_token_factory"):
        run_resumable_validation(
            {"candidate": candidate},
            lambda *args: called.append(args),
            config,
            tmp_path / "strict.json",
            require_budget_token=True,
        )
    assert called == []

    ledger = RunLedger(StageBudget("P1", max_calculator_calls=2))
    checkpoint = tmp_path / "strict.json"
    ledger_path = tmp_path / "ledger.json"

    def factory(token_id, item, calls, metadata):
        assert token_id == metadata["cache_key"]
        return ledger.issue_calculator_token(token_id, calls, metadata=metadata, persist_path=ledger_path)

    def search(candidate_id, item, token):
        token.consume()
        evidence = good()
        evidence.update(source="fixture", calculator_calls=1)
        return evidence

    records = run_resumable_validation(
        {"candidate": candidate},
        search,
        config,
        checkpoint,
        budget_token_factory=factory,
        require_budget_token=True,
    )
    assert records[0].status == "success"
    assert ledger.summary()["totals"]["calculator_calls"] == 1
    assert called == []


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


def test_production_token_budget_failure_is_recorded_and_later_candidates_continue(tmp_path):
    ledger = RunLedger(StageBudget("P1", max_calculator_calls=6))
    ledger_path = tmp_path / "ledger.json"
    checkpoint = tmp_path / "ts.json"
    candidates = {
        "over-budget": {"coordinates": [[0.0, 0.0, 0.0]], "charge": 0, "multiplicity": 1},
        "after-failure": {"coordinates": [[0.2, 0.0, 0.0]], "charge": 0, "multiplicity": 1},
    }

    def factory(token_id, item, calls, metadata):
        return ledger.issue_calculator_token(token_id, calls, metadata=metadata, persist_path=ledger_path)

    seen = []

    def search(candidate_id, item, token):
        seen.append(candidate_id)
        if candidate_id == "over-budget":
            token.consume(3)
            token.consume()  # Fails before a fourth backend call can occur.
        token.consume()
        evidence = good()
        evidence.update(source="fixture", calculator_calls=token.consumed_calls)
        return evidence

    records = run_resumable_validation(
        candidates,
        search,
        TSValidationConfig(max_calls=3),
        checkpoint,
        budget_token_factory=factory,
        require_budget_token=True,
    )
    assert [record.status for record in records] == ["failure", "success"]
    assert records[0].mode_status == "call_budget"
    assert records[0].calculator_calls == 3
    assert seen == ["over-budget", "after-failure"]
    assert ledger.summary()["totals"]["calculator_calls"] == 4
    assert ledger.reservations == {}
    restored = run_resumable_validation(
        candidates,
        lambda *args: (_ for _ in ()).throw(AssertionError("cached records must not rerun")),
        TSValidationConfig(max_calls=3),
        checkpoint,
        budget_token_factory=factory,
        require_budget_token=True,
    )
    assert [record.status for record in restored] == ["failure", "success"]


def test_production_searcher_exception_settles_partial_calls_and_preserves_checkpoint(tmp_path):
    ledger = RunLedger(StageBudget("P1", max_calculator_calls=4))
    ledger_path = tmp_path / "ledger.json"
    checkpoint = tmp_path / "ts.json"

    def factory(token_id, item, calls, metadata):
        return ledger.issue_calculator_token(token_id, calls, metadata=metadata, persist_path=ledger_path)

    def search(candidate_id, item, token):
        token.consume()
        raise RuntimeError("fixture search failed after one calculation")

    records = run_resumable_validation(
        {"candidate": {"coordinates": [[0.0, 0.0, 0.0]], "charge": 0, "multiplicity": 1}},
        search,
        TSValidationConfig(max_calls=3),
        checkpoint,
        budget_token_factory=factory,
        require_budget_token=True,
    )
    assert records[0].status == "failure"
    assert records[0].mode_status == "searcher_exception"
    assert records[0].calculator_calls == 1
    assert ledger.summary()["totals"]["calculator_calls"] == 1
    assert ledger.reservations == {}
