from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "update_run_status.py"


def _module():
    spec = importlib.util.spec_from_file_location("xtbflow_update_run_status", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(**overrides):
    config = {
        "ssh_host": "n2",
        "slurm_user": "lhshen",
        "registry": "/tmp/run_status.d",
        "ssh_timeout_s": 2,
        "ssh_retry_attempts": 2,
        "ssh_retry_delay_s": 0,
    }
    config.update(overrides)
    return config


def test_banner_timeout_is_retryable_but_auth_failure_is_not():
    module = _module()
    assert module.retryable_ssh_error(RuntimeError("banner exchange: Connection timed out"))
    assert module.retryable_ssh_error(subprocess.TimeoutExpired(["ssh"], 2))
    assert not module.retryable_ssh_error(RuntimeError("Permission denied (publickey)"))


def test_collect_retries_transient_ssh_failure(monkeypatch):
    module = _module()
    calls = []
    payload = {"collected_at": "2026-11-08T12:00:00+08:00", "host": "node2", "queue": [], "runs": []}

    def fake_run(command, *, input_text=None, timeout=60):
        calls.append((command, timeout, input_text))
        if len(calls) == 1:
            raise RuntimeError("ssh exited 255: banner exchange: Connection timed out")
        return json.dumps(payload)

    monkeypatch.setattr(module, "run", fake_run)
    assert module.collect(_config()) == payload
    assert len(calls) == 2
    command = calls[0][0]
    assert "ServerAliveInterval=15" in command
    assert "ServerAliveCountMax=2" in command


def test_collect_does_not_retry_auth_failure(monkeypatch):
    module = _module()
    calls = []

    def fake_run(command, *, input_text=None, timeout=60):
        calls.append(command)
        raise RuntimeError("ssh exited 255: Permission denied (publickey)")

    monkeypatch.setattr(module, "run", fake_run)
    with pytest.raises(RuntimeError, match="after 1 attempt"):
        module.collect(_config())
    assert len(calls) == 1
