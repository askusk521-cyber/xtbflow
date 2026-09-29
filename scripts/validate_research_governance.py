#!/usr/bin/env python3
"""Validate workflow-D research/confirmation governance artifacts without network access."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "configs/science/origin_candidates_v0.1.yaml"
CONFIRMATION = ROOT / "configs/validation/confirmation_v0.1.yaml"
PUBLIC_INDEX = ROOT / "data/manifests/confirmation_public_index.json"
ATTEMPT_TEMPLATE = ROOT / "reports/origins_screening/attempt_template.json"
CLAIMS = ROOT / "docs/research/claim_evidence_matrix.md"
SELECTION = ROOT / "docs/science/application_selection_record.md"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {path.relative_to(ROOT)}")
    return payload


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_candidates() -> None:
    payload = load_json(CANDIDATES)
    require(payload.get("schema") == "xtbflow-origin-candidates/v0.1", "Bad candidate schema")
    rows = payload.get("candidates")
    require(isinstance(rows, list) and len(rows) == 6, "Candidate registry must contain six seed rows")
    require(payload.get("candidate_count") == len(rows), "candidate_count mismatch")
    ids = [row.get("candidate_id") for row in rows]
    require(all(isinstance(value, str) and value for value in ids), "Missing candidate_id")
    require(len(set(ids)) == len(ids), "Duplicate candidate_id")
    counts = Counter(row.get("mechanism_family_id") for row in rows)
    require(dict(counts) == payload.get("family_counts"), "family_counts mismatch")

    admitted = 0
    for row in rows:
        require(row.get("admission") in {"quarantine", "admitted"}, f"Bad admission for {row.get('candidate_id')}")
        source = row.get("source")
        require(isinstance(source, dict) and source.get("doi"), f"Missing DOI for {row.get('candidate_id')}")
        require(row.get("event_intent"), f"Missing event intent for {row.get('candidate_id')}")
        require(row.get("competing_events"), f"Missing competition set for {row.get('candidate_id')}")
        if row.get("admission") == "admitted":
            admitted += 1
            require(row.get("structure_status") == "frozen_hashed", "Admitted row lacks frozen structure")
            require(row.get("charge_spin", {}).get("status") == "confirmed", "Admitted row lacks charge/spin evidence")
            require(not row.get("missing"), "Admitted row still has missing fields")

    require(payload.get("runnable_input_count") == admitted, "runnable_input_count mismatch")
    require(admitted == 0, "v0.1 literature seed unexpectedly admits runnable inputs")
    require(payload.get("decision_use") == "exploratory-development", "Candidate use must remain exploratory")


def validate_confirmation() -> None:
    payload = load_json(CONFIRMATION)
    require(payload.get("schema") == "xtbflow-confirmation/v0.1", "Bad confirmation schema")
    require(payload.get("status") == "draft_not_frozen", "Confirmation must remain draft")
    require(payload.get("execution_authorized") is False, "Confirmation execution must be disabled")
    require(payload.get("compute_allocation_created") is False, "Draft must not allocate compute")
    forbidden = set(payload.get("forbidden_confirmatory_inputs", []))
    required_forbidden = {
        "correct_product",
        "reference_ts",
        "reference_mode_or_projector",
        "target_derived_active_water",
        "barrier_or_validation_outcome",
    }
    require(required_forbidden <= forbidden, "Confirmation firewall is incomplete")
    freeze = payload.get("freeze_manifest", {})
    require(isinstance(freeze, dict) and freeze, "Missing freeze manifest")
    require(freeze.get("code_commit") is None, "Draft unexpectedly has a frozen commit")
    require(freeze.get("checkpoint_hash") is None, "Draft unexpectedly has a frozen checkpoint")

    public = load_json(PUBLIC_INDEX)
    require(public.get("schema") == "xtbflow-confirmation-public-index/v0.1", "Bad public-index schema")
    require(public.get("labels_exposed") is False, "Public index exposes sealed labels")
    require(public.get("batches") == [], "Pre-freeze public index must be empty")
    require(public.get("split_hash") is None, "Pre-freeze public index has a split hash")


def validate_documents() -> None:
    claims = CLAIMS.read_text(encoding="utf-8")
    for claim_id in ("C-METHOD-1", "C-METHOD-2", "C-METHOD-3", "C-DISCOVERY-1", "C-CHEM-1"):
        require(claim_id in claims, f"Missing claim {claim_id}")
    selection = SELECTION.read_text(encoding="utf-8").lower()
    require("no application selected" in selection, "Selection record must not preselect an application")
    attempt = load_json(ATTEMPT_TEMPLATE)
    require(attempt.get("outcome", {}).get("status") == "not_run", "Blank attempt template looks executed")
    require(attempt.get("decision_use") == "exploratory-development", "Attempt template use mismatch")


def main() -> int:
    try:
        validate_candidates()
        validate_confirmation()
        validate_documents()
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("PASS: workflow-D governance artifacts are internally consistent.")
    print("No confirmation run or chemistry calculation is authorized by this validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
