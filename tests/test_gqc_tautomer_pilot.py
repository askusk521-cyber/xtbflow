"""Software accounting checks; actual force qualification runs only on n2."""
import importlib.util
import json
from pathlib import Path
import sys

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


def test_interrupted_continuation_preserves_later_completed_starts():
    sys.modules["gqc_tautomer_pilot"] = PILOT
    spec = importlib.util.spec_from_file_location("gqc_continue", Path(__file__).parents[1] / "scripts/continue_gqc_tautomer_minima.py")
    continuation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(continuation)
    fields = dict(id="molecule", group="group", form="form", formula="C2H4", stereo_smiles="C=C", charge=0, multiplicity=1)
    inputs = [{**fields, "start_id": str(i)} for i in range(3)]
    previous = [{**inputs[0], "status": "optimization_unresolved"}, {**inputs[1], "status": "force_converged_identity_pending", "energy_hartree": -1.0}]
    attempted = [{**inputs[0], "status": "continuation_budget_exhausted"}]
    merged = continuation.merge_rows(inputs, previous, attempted)
    assert len(merged) == 3
    assert merged[1] == previous[1]
    assert merged[2]["status"] == "not_run_budget"
