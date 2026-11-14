"""Software accounting checks; actual force qualification runs only on n2."""
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location("gqc_pilot", Path(__file__).parents[1] / "scripts/gqc_tautomer_pilot.py")
PILOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PILOT)


def test_failed_qc_still_consumes_call_and_leaves_journal(tmp_path):
    meter = PILOT.Meter(tmp_path, 1, 60)
    def fail():
        raise RuntimeError("SCF failed")
    with pytest.raises(RuntimeError):
        meter.run("scf", fail)
    with pytest.raises(PILOT.BudgetExceeded):
        meter.run("another", lambda: 1)
    assert meter.summary()["calls_started"] == 1
    assert meter.summary()["calls_failed"] == 1
    assert meter.summary()["calls_unsettled"] == 0
    rows = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [r["status"] for r in rows] == ["started", "failed"]


def test_hessian_gate_does_not_execute_over_limit(tmp_path):
    meter = PILOT.Meter(tmp_path, 4, 60, hessian_cap=1)
    assert meter.run("hessian", lambda: 7, hessian=True) == 7
    with pytest.raises(PILOT.BudgetExceeded):
        meter.run("hessian", lambda: pytest.fail("over-budget operation executed"), hessian=True)
    assert meter.summary()["calls_started"] == 1
    assert meter.summary()["hessians_started"] == 1


def test_expired_wall_budget_prevents_new_work(tmp_path):
    meter = PILOT.Meter(tmp_path, 4, 0)
    with pytest.raises(PILOT.BudgetExceeded):
        meter.run("scf", lambda: pytest.fail("expired work executed"))
    assert meter.summary()["calls_started"] == 0
