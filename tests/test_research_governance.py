from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_research_governance_validator_passes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/validate_research_governance.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "No confirmation run" in result.stdout


def test_origin_seed_registry_is_quarantined() -> None:
    payload = load_json("configs/science/origin_candidates_v0.1.yaml")
    rows = payload["candidates"]
    assert len(rows) == 6
    assert payload["runnable_input_count"] == 0
    assert {row["admission"] for row in rows} == {"quarantine"}


def test_confirmation_draft_cannot_execute() -> None:
    payload = load_json("configs/validation/confirmation_v0.1.yaml")
    assert payload["status"] == "draft_not_frozen"
    assert payload["execution_authorized"] is False
    assert payload["compute_allocation_created"] is False
    assert payload["freeze_manifest"]["checkpoint_hash"] is None


def test_confirmation_public_index_is_empty_and_label_free() -> None:
    payload = load_json("data/manifests/confirmation_public_index.json")
    assert payload["labels_exposed"] is False
    assert payload["batches"] == []
    assert payload["split_hash"] is None
