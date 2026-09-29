from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "run_energy_learning_curve_for_test",
    ROOT / "scripts" / "run_energy_learning_curve.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)


def test_child_receives_remaining_stage_timeout(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(_RUNNER.subprocess, "run", fake_run)
    result = _RUNNER._run_child_with_remaining_budget(
        ["fake-child"],
        cwd=tmp_path,
        started=_RUNNER.time.perf_counter(),
        max_seconds=10.0,
    )
    assert result.returncode == 0
    assert calls[0][1]["timeout"] > 0
    assert calls[0][1]["timeout"] <= 10.0


def test_child_budget_fails_before_launch_when_exhausted(tmp_path):
    with pytest.raises(TimeoutError, match="no execution budget"):
        _RUNNER._run_child_with_remaining_budget(
            ["fake-child"], cwd=tmp_path, started=0.0, max_seconds=0.0
        )


def test_short_lived_fake_child_is_hard_stopped_by_remaining_budget(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        _RUNNER._run_child_with_remaining_budget(
            [sys.executable, "-c", "import time; time.sleep(0.2)"],
            cwd=tmp_path,
            started=_RUNNER.time.perf_counter(),
            max_seconds=0.02,
        )


def test_invalid_child_report_is_settled_as_failed(tmp_path):
    ledger_path = tmp_path / "stage-ledger.json"
    ledger = {"status": "running", "runs": [], "started_monotonic": 0.0}
    run = {"status": "running"}

    _RUNNER._mark_run_failure(ledger_path, ledger, run, "missing report")

    assert run["status"] == "failed"
    assert run["error"] == "missing report"
    assert ledger["status"] == "failed"
    assert "missing report" in ledger_path.read_text(encoding="utf-8")
