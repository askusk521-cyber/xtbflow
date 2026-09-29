from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path("scripts/audit_track_b_sources.py")
SPEC = importlib.util.spec_from_file_location("audit_track_b_sources", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def candidate_config(expected_sha256: str) -> dict[str, object]:
    return {
        "schema_version": "xtbflow-track-b-source-candidates/v1",
        "purpose": "test",
        "sources": [
            {
                "source_id": "demo",
                "source_locator": "demo-source",
                "source_revision": "r1",
                "expected_asset_sha256": expected_sha256,
                "expected_asset_size_bytes": 10,
                "declared_record_count": 1,
                "available": [],
                "missing_for_track_b": ["event_label"],
                "decision": "quarantine_diagnostic_only",
                "permitted_use": [],
            }
        ],
    }


def test_source_audit_verifies_local_asset_hash_and_row_count(tmp_path):
    asset = tmp_path / "records.jsonl"
    asset.write_text('{"id": 1}\n', encoding="utf-8")
    expected = hashlib.sha256(asset.read_bytes()).hexdigest()
    config = tmp_path / "sources.json"
    config.write_text(
        json.dumps(candidate_config(expected)),
        encoding="utf-8",
    )
    report = MODULE.build_report(
        config,
        supplied_assets={"demo": asset},
        audit_date="2026-09-29",
        repository_base_commit="abc123",
    )
    assert report["asset_failures"] == []
    assert report["admitted_source_count"] == 0
    observed = report["sources"][0]["observed_asset"]
    assert observed["matches_expected_size"] is True
    assert observed["matches_expected_sha256"] is True
    assert observed["nonempty_jsonl_rows"] == 1
    assert report["scientific_claim_allowed"] is False


def test_source_audit_reports_hash_mismatch_and_unknown_asset(tmp_path):
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"real")
    config = tmp_path / "sources.json"
    payload = candidate_config("0" * 64)
    payload["sources"][0]["expected_asset_size_bytes"] = 4
    config.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    report = MODULE.build_report(
        config,
        supplied_assets={"demo": asset},
        audit_date="2026-09-29",
        repository_base_commit="abc123",
    )
    assert report["asset_failures"] == ["asset hash mismatch: demo"]
    with pytest.raises(ValueError, match="unknown source IDs"):
        MODULE.build_report(
            config,
            supplied_assets={"other": asset},
            audit_date="2026-09-29",
            repository_base_commit="abc123",
        )


def test_source_audit_reports_size_mismatch_before_hash(tmp_path):
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"real")
    expected = hashlib.sha256(asset.read_bytes()).hexdigest()
    payload = candidate_config(expected)
    payload["sources"][0]["expected_asset_size_bytes"] = 5
    config = tmp_path / "sources.json"
    config.write_text(json.dumps(payload), encoding="utf-8")
    report = MODULE.build_report(
        config,
        supplied_assets={"demo": asset},
        audit_date="2026-09-29",
        repository_base_commit="abc123",
    )
    assert report["asset_failures"] == ["asset size mismatch: demo"]
    assert report["sources"][0]["observed_asset"]["matches_expected_size"] is False
