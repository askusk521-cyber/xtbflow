from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[1]
VALIDATOR_PATH = ROOT / "scripts" / "validate_claim_evidence.py"
RENDERER_PATH = ROOT / "scripts" / "render_claim_evidence.py"
EXAMPLE = ROOT / "configs/evidence/claim_evidence_manifest_v1.example.json"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_example_manifest_is_valid_and_result_blind():
    validator = _load(VALIDATOR_PATH, "claim_evidence_validator")
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    summary = validator.validate_manifest(payload)
    assert summary["claim_count"] == 1
    assert summary["status_counts"]["planned"] == 1
    assert payload["panel_manifest_sha256"] is None


def test_scientific_observed_claim_requires_refs_and_complete_accounting():
    validator = _load(VALIDATOR_PATH, "claim_evidence_validator_observed")
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    claim = payload["claims"][0]
    claim["status"] = "observed"
    with pytest.raises(ValueError, match="evidence_refs"):
        validator.validate_manifest(payload)
    claim["evidence_refs"] = [{"kind": "report", "path": "docs/evidence/report.json", "sha256": None}]
    with pytest.raises(ValueError, match="complete budget"):
        validator.validate_manifest(payload)


def test_duplicate_claim_and_comparison_ids_are_rejected():
    validator = _load(VALIDATOR_PATH, "claim_evidence_validator_duplicates")
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    duplicate = json.loads(json.dumps(payload["claims"][0]))
    payload["claims"].append(duplicate)
    with pytest.raises(ValueError, match="claim_id"):
        validator.validate_manifest(payload)


def test_svg_and_json_summary_are_deterministic():
    validator = _load(VALIDATOR_PATH, "validate_claim_evidence")
    renderer = _load(RENDERER_PATH, "claim_evidence_renderer")
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    validation = validator.validate_manifest(payload)
    first = renderer._summary(payload, validation)
    second = renderer._summary(payload, validation)
    assert first == second
    svg = renderer._svg(first)
    assert "w08-joint-over-serial-example" in svg
    assert "summary only" in svg
