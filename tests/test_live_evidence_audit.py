from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import pytest


spec = importlib.util.spec_from_file_location(
    "live_evidence_audit", Path(__file__).parents[1] / "scripts/audit_live_certification_evidence.py"
)
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def test_audit_rejects_legacy_success_and_preserves_costs_and_inputs(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text('{"budget": 800}\n')
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    source = {"commit": "a" * 40, "dirty": False}
    manifest = {
        "schema": "xtbflow-real-feedback-loop/v1",
        "source": source,
        "config_sha256": config_hash,
        "arms": [{
            "arm": "physical_feedback",
            "certification": {
                "status": "success", "reason_code": "CERTIFIED",
                "cost": {"wall_seconds": 2.1, "cpu_hours": 0.0005, "gpu_hours": 0.0},
                "backend_result": {"calculator_calls": 306, "evidence": {
                    "path_connected": True,
                    "irc_method": "bounded_cartesian_gradient_following_v1",
                    "endpoints": [{"minimum_converged": True, "path": [{}, {}, {}]}] * 2,
                }},
            },
        }],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    published_path = tmp_path / "evidence.json"
    published_path.write_text(json.dumps({"source": source, "config_sha256": "b" * 64}))
    inputs = [config_path, manifest_path, published_path]
    before = [p.read_bytes() for p in inputs]
    result = audit_module.audit(manifest_path, config_path, published_path)
    assert result["status"] == "invalidated"
    assert {f["reason_code"] for f in result["findings"]} == {"PUBLIC_CONFIG_HASH_MISMATCH", "UNSUPPORTED_STRICT_CERTIFICATION"}
    assert result["asset_hashes"]["configuration"] == config_hash
    assert result["arms"][0]["calculator_calls"] == 306
    assert result["arms"][0]["cpu_hours"] == 0.0005
    assert result["counts"] == {"attempts_in_denominator": 1, "strict_certified_channels_supported": 0}
    assert result["audit_calculator_calls"] == 0
    assert [p.read_bytes() for p in inputs] == before


def test_audit_missing_inputs_emit_structured_failure(tmp_path, monkeypatch):
    out = tmp_path / "audit.json"
    monkeypatch.setattr("sys.argv", ["audit", "--manifest", str(tmp_path / "missing"), "--config", "missing", "--published-evidence", "missing", "--output", str(out)])
    assert audit_module.main() == 2
    result = json.loads(out.read_text())
    assert result["status"] == "execution_failed"
    assert result["reason_code"] == "AUDIT_INPUT_FAILURE"
    assert result["attempt"] == 1
    assert all(key in result for key in ("wall_seconds", "cpu_hours", "gpu_hours"))


def test_audit_rerun_rejects_unsupported_verdict_even_in_replay(tmp_path):
    summary = {"outcome": "attempted_unresolved"}
    summary_hash = hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    cert = {
        "status": "attempted_unresolved", "reason_code": "INCOMPLETE_IRC_CERTIFICATION",
        "backend_result": {"claim_level": "proxy_metric", "calculator_calls": 5, "evidence": {"path_connected": False}},
        "ledger_summary": {"totals": {"calculator_calls": 5}},
    }
    run = {"source": {"dirty": False}, "config_sha256": "c" * 64,
           "status": "blocked_certification", "unique_strict_certified_channels": 0,
           "replay": {"deterministic_summary": summary, "deterministic_summary_sha256": summary_hash},
           "arms": [{"certification": cert}]}
    run_path, replay_path = tmp_path / "run.json", tmp_path / "replay.json"
    run_path.write_text(json.dumps(run))
    cert["backend_result"]["evidence"]["path_connected"] = True
    replay_path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="strict connectivity"):
        audit_module.audit_rerun(run_path, replay_path, "c" * 64)
