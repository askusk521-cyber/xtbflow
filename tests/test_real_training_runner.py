from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


pytest.importorskip("numpy")
pytest.importorskip("torch")


def _load_runner():
    path = Path(__file__).parents[1] / "scripts" / "run_dft_da_joint_training.py"
    spec = importlib.util.spec_from_file_location("xtbflow_real_training_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load training runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_training_output_is_single_use_and_failures_are_durable(tmp_path: Path):
    runner = _load_runner()
    output = tmp_path / "run"
    run_id, state_path, state = runner._start_run(output, "unit-run")
    assert run_id == "unit-run"
    assert json.loads(state_path.read_text())["status"] == "prepared"
    with pytest.raises(FileExistsError):
        runner._start_run(output, "second-run")

    runner._update_stage(state_path, state, "input_audit", "running", started_at="now")
    runner._fail_run(state_path, state, "input_audit", ValueError("bad source"))
    persisted = json.loads(state_path.read_text())
    assert persisted["status"] == "failed"
    assert persisted["failure"]["status"] == "input_audit_failed"
    assert persisted["stages"]["input_audit"]["error_type"] == "ValueError"


def test_budget_failure_is_distinguished_from_generic_training_failure(tmp_path: Path):
    runner = _load_runner()
    _, state_path, state = runner._start_run(tmp_path / "budget-run", "budget-run")
    runner._update_stage(state_path, state, "training", "running", started_at="now")
    runner._fail_run(state_path, state, "training", RuntimeError("run-level wall budget exhausted"))
    persisted = json.loads(state_path.read_text())
    assert persisted["failure"]["status"] == "budget_exhausted"
