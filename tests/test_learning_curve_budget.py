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


def test_stage_failure_is_persisted(tmp_path):
    ledger_path = tmp_path / "stage-ledger.json"
    ledger = {"status": "running", "runs": [], "started_monotonic": 0.0}

    _RUNNER._mark_stage_failure(ledger_path, ledger, "final summary failed")

    persisted = ledger_path.read_text(encoding="utf-8")
    assert ledger["status"] == "failed"
    assert ledger["error"] == "final summary failed"
    assert '"status": "failed"' in persisted
    assert "final summary failed" in persisted


def test_output_directory_rejects_prior_evidence_without_modifying_it(tmp_path):
    output_dir = tmp_path / "old-run"
    output_dir.mkdir()
    report = output_dir / "size12_seed1.json"
    report.write_text("historical report\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="refusing to reuse"):
        _RUNNER._prepare_output_dir(output_dir)

    assert report.read_text(encoding="utf-8") == "historical report\n"


def test_output_directory_is_created_only_when_empty(tmp_path):
    output_dir = tmp_path / "new-run" / "nested"

    _RUNNER._prepare_output_dir(output_dir)

    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []


def test_existing_artifact_path_is_rejected(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text("historical summary\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="already exists"):
        _RUNNER._require_new_artifact(summary, "summary path")

    assert summary.read_text(encoding="utf-8") == "historical summary\n"
