"""The automated body must retain current task routing and its JSON contract."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_status_routing", ROOT / "scripts/update_run_status.py")
status = importlib.util.module_from_spec(spec)
spec.loader.exec_module(status)


def test_generated_status_retains_route_and_machine_payload():
    cfg = json.loads((ROOT / "configs/run_status.json").read_text())
    payload = dict(collected_at="2026-11-07T10:00:00+08:00", host="node2", queue=[], runs=[], recent=[])
    body = status.render(cfg, payload)
    assert "/issues/129" in body
    assert "English overview" in body and "中文说明" in body
    assert status.extract(body) == payload


def test_configured_status_endpoint_does_not_select_oldest_label_match(monkeypatch):
    calls = []
    def gh(*args):
        calls.append(args)
        return json.dumps(dict(number=130, body="", url="fixture", state="OPEN", labels=[dict(name="run-status")]))
    monkeypatch.setattr(status, "gh", gh)
    assert status.find_issue(dict(repo="owner/repo", issue_number=130, label="run-status"))["number"] == 130
    assert calls[0][:3] == ("issue", "view", "130")


def test_closed_configured_endpoint_cannot_silently_create_new_issue(monkeypatch):
    monkeypatch.setattr(status, "gh", lambda *args: json.dumps(dict(state="CLOSED", labels=[])))
    with pytest.raises(ValueError, match="must be open"):
        status.ensure_issue(dict(repo="owner/repo", issue_number=130, label="run-status"))
