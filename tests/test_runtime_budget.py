from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pytest

from xtbflow.runtime import (
    BudgetExceeded,
    ExecutionResult,
    LedgerEvent,
    RunLedger,
    RunManager,
    RunSpec,
    StageBudget,
    SystemInput,
)


def budget(**overrides):
    values = dict(phase="P1", max_system_evaluations=10, max_calculator_calls=20, max_cpu_core_hours=1, max_gpu_hours=1, max_retries_per_job=1)
    values.update(overrides)
    return StageBudget(**values)


def event(system_id: str, *, attempt: int = 1, status: str = "success", calculator_calls: int = 1, cache_hit: bool = False, operation: str = "energy_forces"):
    return LedgerEvent(run_id="run", phase="P1", calculator="mock", protocol_id="proto-v1", calculator_build_hash="build", system_id=system_id, input_hash=f"hash-{system_id}", operation=operation, status=status, attempt=attempt, calculator_calls=calculator_calls, cache_hit=cache_hit, error=None if status == "success" else "failed")


def test_batch_systems_and_hidden_calculator_calls_are_counted():
    ledger = RunLedger(budget(max_system_evaluations=4, max_calculator_calls=4))
    for name in ("a", "b", "c"):
        ledger.append(event(name))
    ledger.append(event("c", attempt=2, operation="hessian", calculator_calls=1))
    summary = ledger.summary()
    assert summary["totals"]["system_evaluations"] == 4
    assert summary["totals"]["calculator_calls"] == 4
    assert summary["by_operation"]["hessian"] == 1


def test_hard_budget_rejects_projected_event_without_mutation():
    ledger = RunLedger(budget(max_system_evaluations=1))
    ledger.append(event("a"))
    with pytest.raises(BudgetExceeded):
        ledger.append(event("b"))
    assert len(ledger.events) == 1


def test_retry_limit_and_failure_are_recorded():
    ledger = RunLedger(budget(max_retries_per_job=1))
    ledger.append(event("a", status="failure"))
    ledger.append(event("a", attempt=2))
    with pytest.raises(BudgetExceeded):
        ledger.append(event("a", attempt=3))
    assert ledger.summary()["totals"]["failures"] == 1
    assert ledger.summary()["totals"]["retries"] == 1


def test_cache_hit_records_a_system_but_no_calculator_call():
    ledger = RunLedger(budget(max_system_evaluations=1, max_calculator_calls=1))
    ledger.append(event("a", calculator_calls=0, cache_hit=True))
    assert ledger.summary()["totals"]["system_evaluations"] == 1
    assert ledger.summary()["totals"]["calculator_calls"] == 0
    assert ledger.summary()["totals"]["cache_hits"] == 1


def test_persistence_and_resume_are_idempotent(tmp_path: Path):
    ledger = RunLedger(budget(max_system_evaluations=4), ledger_id="ledger")
    manager = RunManager(ledger, tmp_path / "runs")
    spec = RunSpec("run", "P1", "mock-task", "mock", "proto-v1", "build", "commit", "env", "ledger")
    systems = [SystemInput("a", "ha", {"x": 1}), SystemInput("b", "hb", {"x": 2})]
    calls = []

    def executor(system):
        calls.append(system.system_id)
        return ExecutionResult()

    manager.run(spec, systems, executor)
    manager.run(spec, systems, executor, resume=True)
    assert calls == ["a", "b"]
    restored = RunLedger.load(tmp_path / "runs" / "ledger.json")
    assert restored.summary()["totals"]["system_evaluations"] == 2
    assert json.loads((tmp_path / "runs" / "run.json").read_text())["status"] == "completed"


def test_concurrent_appends_are_serialized():
    ledger = RunLedger(budget(max_system_evaluations=10))

    def append(name):
        ledger.append(event(name))

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(append, ["a", "b", "c", "d"]))
    assert ledger.summary()["totals"]["system_evaluations"] == 4


def test_runner_retries_a_recorded_failure_on_resume(tmp_path: Path):
    ledger = RunLedger(budget(max_system_evaluations=2, max_retries_per_job=1), ledger_id="ledger")
    manager = RunManager(ledger, tmp_path / "runs")
    spec = RunSpec("run", "P1", "mock-task", "mock", "proto-v1", "build", "commit", "env", "ledger")
    system = [SystemInput("a", "ha")]

    manager.run(spec, system, lambda _: (_ for _ in ()).throw(RuntimeError("transient")))
    manager.run(spec, system, lambda _: ExecutionResult(), resume=True)
    attempts = [event for event in ledger.events if event.system_id == "a"]
    assert [event.status for event in attempts] == ["failure", "success"]
    assert [event.attempt for event in attempts] == [1, 2]
