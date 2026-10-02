from __future__ import annotations

import hashlib
import json
from pathlib import Path

from xtbflow.governance.handoff import ARTIFACT_NAMES, validate_manifest


def _write_artifacts(root: Path) -> dict[str, dict[str, object]]:
    artifacts: dict[str, dict[str, object]] = {}
    for index, name in enumerate(ARTIFACT_NAMES):
        path = root / f"{name}.fixture"
        path.write_text(f"{name}-{index}\n", encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifacts[name] = {
            "kind": "file",
            "locator": path.name,
            "sha256": digest,
            "status": "verified",
            "source_status": "registered",
        }
    return artifacts


def _manifest(root: Path, *, status: str = "ready", remote: str = "verified") -> dict[str, object]:
    evidence_status = "pass" if status in {"ready", "completed"} else "blocked"
    return {
        "schema": "xtbflow-handoff-freeze/v1",
        "manifest_id": "test-freeze-v1",
        "status": status,
        "work_package": {
            "id": "W00",
            "title": "test handoff",
            "owner": "test-agent",
            "write_scope": ["tests/test_handoff_freeze.py"],
        },
        "github": {
            "repository": "askusk521-cyber/xtbflow",
            "base_sha": "a0c959358b988f220f66a3641c53bb9d997069da",
            "head_sha": "b0c959358b988f220f66a3641c53bb9d997069db",
            "base_ref": "origin/main",
            "head_ref": "codex/test",
            "remote_verification": remote,
        },
        "artifacts": _write_artifacts(root),
        "evidence": {
            "level": "software_test",
            "status": evidence_status,
            "failure_reasons": [] if evidence_status == "pass" else ["fixture is intentionally blocked"],
        },
        "execution": {
            "command": "pytest -q tests/test_handoff_freeze.py",
            "ledger_id": "ledger-test-v1",
            "output_index": "test-output",
        },
    }


def test_ready_manifest_verifies_local_hashes_and_identity(tmp_path: Path):
    audit = validate_manifest(_manifest(tmp_path), root=tmp_path)

    assert audit["contract_valid"] is True
    assert audit["ready"] is True
    assert audit["blocking_reasons"] == []
    assert audit["identity"]["artifacts"]["checkpoint"]["sha256"]


def test_offline_mode_cannot_claim_remote_verification(tmp_path: Path):
    audit = validate_manifest(_manifest(tmp_path), root=tmp_path, offline=True)

    assert audit["contract_valid"] is False
    assert audit["ready"] is False
    assert any("offline validation" in reason for reason in audit["blocking_reasons"])


def test_unknown_source_and_missing_hash_block_even_when_shape_is_valid(tmp_path: Path):
    manifest = _manifest(tmp_path, status="blocked", remote="offline")
    manifest["artifacts"]["data"]["sha256"] = None
    manifest["artifacts"]["data"]["status"] = "unknown"
    manifest["artifacts"]["data"]["source_status"] = "quarantine"

    audit = validate_manifest(manifest, root=tmp_path, offline=True)

    assert audit["contract_valid"] is True
    assert audit["ready"] is False
    assert any("data: source_status='quarantine'" in reason for reason in audit["blocking_reasons"])
    assert any("data: complete verified SHA-256" in reason for reason in audit["blocking_reasons"])


def test_declared_hash_mismatch_is_a_blocker(tmp_path: Path):
    manifest = _manifest(tmp_path, status="blocked", remote="offline")
    manifest["artifacts"]["config"]["sha256"] = "0" * 64

    audit = validate_manifest(manifest, root=tmp_path, offline=True)

    assert audit["contract_valid"] is True
    assert any("config: declared SHA-256 does not match" in reason for reason in audit["blocking_reasons"])


def test_failed_evidence_requires_failure_reason(tmp_path: Path):
    manifest = _manifest(tmp_path, status="failed", remote="offline")
    manifest["evidence"]["failure_reasons"] = []

    audit = validate_manifest(manifest, root=tmp_path, offline=True)

    assert audit["contract_valid"] is False
    assert any("blocked/fail evidence" in error for error in audit["errors"])


def test_schema_is_machine_readable():
    schema_path = Path(__file__).parents[1] / "schemas" / "handoff_freeze.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["properties"]["schema"]["const"] == "xtbflow-handoff-freeze/v1"
    assert set(schema["properties"]["artifacts"]["required"]) == set(ARTIFACT_NAMES)
