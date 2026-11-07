"""Software fixtures for live action dispatch, accounting and crash recovery."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from xtbflow.governance.generation_bridge import certification_candidate, validate_query, validate_generation_config, validate_journal
from xtbflow.runtime import BudgetExceeded, LedgerEvent, RunLedger, StageBudget
from xtbflow.runtime import generation_actions as module
from xtbflow.runtime.generation_actions import GenerationActions
from xtbflow.runtime.feedback_loop import FeedbackLoop, candidate_view, choose, initial_state
from xtbflow.runtime.live_feedback_tools import LiveGenerationTools
from xtbflow.validation.certification import CertificationService, file_hash, identity, write_json
from test_certification_service import FixtureBackend, budget, run
from test_generation_bridge import query

ROOT = Path(__file__).resolve().parents[1]


def action(name="propose", *, seed=139, sampler="joint_event_lead2"):
    value = dict(name=name, candidate_id=None, parameters=dict(sampler=sampler, sampling_seed=seed))
    return {**value, "action_id": identity(value)}


def template():
    return json.loads((ROOT / "configs/research/generation_bridge_20261107.json").read_text())


def stage(**changes):
    config = dict(phase="generation-action-software-fixture", max_gpu_hours=1, max_cpu_core_hours=8,
        max_system_evaluations=150, max_calculator_calls=1, max_storage_gb=1,
        max_concurrent_jobs=1, max_retries_per_job=0, max_wall_minutes_per_job=12)
    config.update(changes)
    return StageBudget(**config)


class FixtureGeneration(GenerationActions):
    invocations = 0
    mode = "complete"

    def _execute(self, folder, cfg, conditioned):
        self.invocations += 1
        g = folder / "generation"
        (g / "raw").mkdir(parents=True)
        q = json.loads((folder / "reactants.json").read_text())["queries"][0]
        method = cfg["methods"][0]
        roles = ["event", "geometry"] if method == "independent_event_then_geometry" else ["geometry" if conditioned else "joint"]
        rows = [dict(sequence=i+1, role=r, systems=1, status="completed", query_id=q["query_id"],
                     method=method, proposal=0) for i, r in enumerate(role for role in roles for _ in range(50))]
        if self.mode == "failed_kernel":
            rows = [{**rows[0], "status": "started"}]
        journal = dict(calls=rows, consumed_forward_evaluations=len(rows))
        write_json(g / "raw/calls.json", journal)
        if self.mode == "failed_kernel":
            raise RuntimeError("injected model failure")
        fixed = json.loads((folder / "conditioned-events.json").read_text())[q["query_id"]] if conditioned else [[1, 0], [0, 1]]
        proto = dict(role="software_fixture", config_sha256=file_hash(folder / "config.json"))
        generated = dict(query_id=q["query_id"], candidate_id="fixture-new-proposal", method=method, proposal=0,
                         status="generated", b_dec=fixed, x=[[0., 0., -.5], [0., 0., .5]])
        c = certification_candidate(q, generated, generation_protocol=proto)
        plan = dict(schema="xtbflow-generated-certification-plan/v1", source_commit="a"*40, generation_protocol=proto,
                    generator_journal_sha256=file_hash(g / "raw/calls.json"), candidates=[dict(candidate=asdict(c), method=method,
                        generation_forward_evaluations=len(rows))])
        write_json(g / "raw/certification-plan.json", plan)
        ledger = RunLedger(stage())
        ledger.append(LedgerEvent(run_id="fixture", phase=ledger.budget.phase, calculator="fixture_generator",
            protocol_id="fixture", calculator_build_hash="a"*64, system_id=q["query_id"], input_hash="a"*64,
            operation="other", status="success", calculator_calls=0, system_evaluations=len(rows)), persist_path=g / "ledger.json")
        write_json(g / "report.json", dict(status="generation_completed", settlement_status="settled",
            config_sha256=file_hash(folder / "config.json"), call_summary=validate_journal(journal, cap=100), ledger=ledger.summary(),
            certification_plan_sha256=file_hash(g / "raw/certification-plan.json"), certification_candidates=1,
            artifacts=[dict(path=str(p.relative_to(g)), sha256=file_hash(p), bytes=p.stat().st_size) for p in (g / "raw").iterdir()],
            candidates=[dict(status="generated", candidate_content_sha256=identity(c.content()))]))


def service(root, **changes):
    return FixtureGeneration(root, stage(**changes), source=ROOT, research_source=ROOT, model_root=root, template=template())


def test_fresh_generation_cache_preserves_model_calls_and_ledger_exactly(tmp_path):
    engine = service(tmp_path)
    first = engine.run(action(), query())
    ledger_before = file_hash(engine.ledger_path)
    assert first == engine.run(action(), query())
    assert engine.invocations == 1 and ledger_before == file_hash(engine.ledger_path)
    assert first["cost"]["joint_forwards"] == 50 and first["cost"]["reference_primitive_calls"] == 0
    assert len(first["candidates"]) == 1


def test_conditioned_geometry_preserves_explicit_generated_event_and_uses_no_event_forward(tmp_path):
    engine = service(tmp_path)
    fixed = [[0, 1], [1, 0]]
    result = engine.run(action("resample_geometry"), query(), fixed)
    assert result["candidates"][0]["planned_product_be"] == fixed
    assert result["cost"]["geometry_forwards"] == 50 and result["cost"]["event_forwards"] == 0
    with pytest.raises(ValueError, match="explicit generated"):
        engine.run(action("resample_geometry", seed=140), query())


def test_failed_forward_stays_charged_and_does_not_automatically_retry(tmp_path):
    engine = service(tmp_path)
    engine.mode = "failed_kernel"
    result = engine.run(action(), query())
    assert result["status"] == "execution_failed" and not result["candidates"]
    assert result["calls"]["unfinished_calls"] == 1 and result["cost"]["joint_forwards"] == 1
    assert result == engine.run(action(), query()) and engine.invocations == 1


def test_remaining_budget_must_fit_complete_next_envelope(tmp_path):
    engine = service(tmp_path, max_system_evaluations=75)
    engine.run(action(), query())
    with pytest.raises(BudgetExceeded):
        engine.run(action(seed=140), query())
    assert engine.invocations == 1
    with pytest.raises(BudgetExceeded, match="per-job"):
        service(tmp_path / "short", max_wall_minutes_per_job=1).run(action(), query())


def test_interrupted_result_write_recovers_original_calls_without_rerun(tmp_path, monkeypatch):
    engine = service(tmp_path)
    original = module.write_json

    def failed(path, value):
        if path.name == "action-result.json":
            raise OSError("injected result write failure")
        original(path, value)

    monkeypatch.setattr(module, "write_json", failed)
    with pytest.raises(OSError):
        engine.run(action(), query())
    assert json.loads(next(tmp_path.glob("*/action-report-failure.json")).read_text())["status"] == "report_write_failed"
    monkeypatch.setattr(module, "write_json", original)
    result = engine.run(action(), query())
    assert result["status"] == "execution_interrupted" and engine.invocations == 1
    assert result["cost"]["joint_forwards"] == 50 and result["cost"]["gpu_seconds"] == 130
    assert result["cost_basis"] == "conservative_interrupted_envelope"
    assert result == engine.run(action(), query())


def test_settlement_failure_does_not_drop_or_duplicate_model_calls(tmp_path, monkeypatch):
    engine = service(tmp_path)
    original = engine._settle
    monkeypatch.setattr(engine, "_settle", lambda *args: (_ for _ in ()).throw(OSError("injected settlement failure")))
    with pytest.raises(OSError):
        engine.run(action(), query())
    assert json.loads(next(tmp_path.glob("*/action-settlement.json")).read_text())["status"] == "settlement_failed"
    monkeypatch.setattr(engine, "_settle", original)
    result = engine.run(action(), query())
    assert result["status"] == "generation_completed" and engine.invocations == 1
    assert RunLedger.load(engine.ledger_path).summary()["totals"]["system_evaluations"] == 50


def test_missing_journal_is_an_explicit_conservative_bound_not_free_work(tmp_path, monkeypatch):
    engine = service(tmp_path)
    original = engine._execute

    def missing(folder, cfg, conditioned):
        original(folder, cfg, conditioned)
        (folder / "generation/raw/calls.json").unlink()

    monkeypatch.setattr(engine, "_execute", missing)
    result = engine.run(action(), query())
    assert result["status"] == "accounting_unresolved" and not result["candidates"]
    assert result["cost"]["joint_forwards"] == 50
    assert result["calls"]["kernel_invocations"] is None
    assert result["counter_basis"] == "conservative_unresolved_forward_envelope"


@pytest.mark.parametrize("change", ["artifact", "query", "action"])
def test_cached_generation_rejects_changed_evidence_input_or_action(tmp_path, change):
    engine = service(tmp_path)
    a, q = action(), query()
    engine.run(a, q)
    if change == "artifact":
        next(tmp_path.glob("*/generation/raw/calls.json")).write_text("{}")
    elif change == "query":
        q["x_r"][0][0] += .1
    else:
        a["parameters"]["sampling_seed"] += 1
    with pytest.raises(ValueError):
        engine.run(a, q)
    assert engine.invocations == 1


def test_unbounded_config_and_hidden_model_inputs_are_rejected(tmp_path):
    cfg = template()
    cfg["proposals_per_query"] = 100000
    with pytest.raises(ValueError):
        validate_generation_config(cfg)
    q = query()
    q["observed_product"] = "not allowed in reactant packet"
    with pytest.raises(ValueError, match="forbidden"):
        service(tmp_path).run(action(), q)


def test_real_tool_api_connects_observation_decision_and_executed_generation(tmp_path):
    q = query()
    c = replace(validate_query(q), candidate_id="initial-proposal", planned_product_be=[[1, 0], [0, 1]])
    ref = CertificationService(tmp_path / "reference", budget())
    previous = run(ref, item=c, backend=FixtureBackend(mode="alternative"))["record"]
    work = ref.root / previous["cache_key"] / "attempt-0001"
    protocol = identity(previous["protocol"])
    engine = service(tmp_path / "generation")
    tools = LiveGenerationTools({c.candidate_id: work}, {c.candidate_id: file_hash(work / "result.json")},
        query=q, reference_protocol_sha256=protocol, generator=engine)
    cfg = json.loads((ROOT / "configs/agent/live_generation_policy_20261107.json").read_text())
    loop = FeedbackLoop(tmp_path / "loop", [candidate_view(c, protocol)], cfg, tools)
    state = loop.run()
    assert len(state["channels"]) == 1 and len(state["pending_certification"]) == 1
    assert state["generation_spent"]["event_forwards"] == 50 and state["generation_spent"]["geometry_forwards"] == 50
    assert state["generation_spent"]["joint_forwards"] == 0 and engine.invocations == 1
    assert state["attributed_reference_cost"]["reference_primitive_calls"] == 2
    assert state == loop.run() and engine.invocations == 1
    # A native generated proposal never creates another certified channel.
    assert len(state["observations"]) == 1 and len(state["evidence"]) == 1


def test_generation_worker_parent_death_contract_terminates_descendant(tmp_path):
    pidfile = tmp_path / "child.pid"
    child = "from research_generation_worker import bind_parent_lifetime; bind_parent_lifetime(); import os,time,pathlib; pathlib.Path(" + repr(str(pidfile)) + ").write_text(str(os.getpid())); time.sleep(20)"
    parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c'," + repr(child) + "]); time.sleep(20)"
    proc = subprocess.Popen([sys.executable, "-c", parent], env={**os.environ, "PYTHONPATH": str(ROOT / "scripts")})
    child_pid = None
    try:
        deadline = time.monotonic() + 5
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert pidfile.exists()
        child_pid = int(pidfile.read_text())
        proc.kill()
        proc.wait(timeout=3)
        while time.monotonic() < deadline:
            stat = Path(f"/proc/{child_pid}/stat")
            if not stat.exists() or stat.read_text().split()[2] == "Z":
                break
            time.sleep(.02)
        else:
            pytest.fail("generation child outlived its killed parent")
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=3)
        if child_pid:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
