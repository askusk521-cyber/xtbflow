from __future__ import annotations

from copy import deepcopy

import pytest

from xtbflow.data.origin_readiness import (
    OriginReadinessError,
    audit_origin_candidate_registry,
)


def seed_record(**overrides):
    values = {
        "schema_version": "xtbflow-independent-reactant-seed/v1",
        "seed_id": "seed-1",
        "parent_system_id": "parent-1",
        "family_id": "family-a",
        "mapped_explicit_h_smiles": "[H:1][O:2][H:3]",
        "geometry_path": "data/reactants/seed-1.xyz",
        "geometry_sha256": "1" * 64,
        "coordinate_unit": "angstrom",
        "charge": 0,
        "multiplicity": 1,
        "charge_spin_evidence": "source/SI plus explicit microstate review",
        "microstate_description": "neutral water fixture",
        "stereochemistry_description": "achiral",
        "solvent_snapshot_id": "none",
        "solvent_selection_origin": "none",
        "reactant_generation_protocol": "fixture-protocol-v1",
        "reactant_generation_protocol_locator": "configs/fixture.json",
        "reactant_generation_protocol_sha256": "2" * 64,
        "provenance": "synthetic unit-test fixture",
        "approved_split_group": "group-1",
        "approved_split_role": "development_train",
        "frozen_at": "2026-09-29T00:00:00+00:00",
        "license_record": "test-only",
        "input_origin": "independent_reactant",
        "allowed_intent": "fixture event",
        "buffer_species_explicit": [],
        "temperature_k": 298.15,
        "ph": 7.0,
    }
    values.update(overrides)
    return values


def candidate(**overrides):
    values = {
        "candidate_id": "candidate-1",
        "parent_reaction_id": "parent-1",
        "mechanism_family_id": "family-a",
        "title": "Fixture candidate",
        "source": {
            "doi": "10.0000/example",
            "locator": "Figure 1",
            "verified_on": "2026-09-29",
        },
        "reactant_identities": ["reactant A", "reactant B"],
        "event_intent": "fixture event",
        "competing_events": ["fixture alternative"],
        "source_conditions": {"phase": "aqueous"},
        "charge_spin": {
            "status": "provisional",
            "charge": 0,
            "multiplicity": 1,
            "reason": "not confirmed",
        },
        "structure_status": "not_prepared",
        "admission": "quarantine",
        "missing": ["exact structures", "input hashes"],
    }
    values.update(overrides)
    return values


def registry(rows, *, runnable_input_count=0):
    family_counts = {}
    for row in rows:
        family = row["mechanism_family_id"]
        family_counts[family] = family_counts.get(family, 0) + 1
    return {
        "schema": "xtbflow-origin-candidates/v0.1",
        "candidate_count": len(rows),
        "runnable_input_count": runnable_input_count,
        "family_counts": family_counts,
        "candidates": rows,
    }


def audit(payload):
    return audit_origin_candidate_registry(
        payload,
        registry_locator="github:test/repo@abc:registry.json",
        registry_source_commit="abc",
        registry_sha256="3" * 64,
        repository_base_commit="def",
    )


def test_quarantined_literature_anchor_does_not_become_seed():
    report = audit(registry([candidate()]))
    row = report["candidate_reports"][0]

    assert report["literature_anchor_ready_count"] == 1
    assert report["identity_state_ready_count"] == 0
    assert report["independent_seed_ready_count"] == 0
    assert report["seed_manifest_population_allowed"] is False
    assert report["search_authorized"] is False
    assert row["provisional_charge_spin_values_ignored"] is True
    assert set(row["blocking_codes"]) >= {
        "candidate_not_admitted",
        "structure_not_frozen_hashed",
        "charge_spin_not_confirmed",
        "declared_requirements_missing",
        "license_record_absent",
        "independent_seed_records_absent",
    }


def test_explicit_contract_valid_seed_can_pass_readiness():
    row = candidate(
        charge_spin={"status": "confirmed", "charge": 0, "multiplicity": 1},
        structure_status="frozen_hashed",
        admission="admitted",
        missing=[],
        license_record="CC0 test fixture",
        independent_seed_records=[seed_record()],
    )
    report = audit(registry([row], runnable_input_count=1))
    result = report["candidate_reports"][0]

    assert result["identity_state_ready"] is True
    assert result["independent_seed_ready"] is True
    assert result["validated_seed_record_count"] == 1
    assert report["seed_manifest_population_allowed"] is True
    assert report["search_authorized"] is False


def test_invalid_seed_parent_is_reported_without_promotion():
    row = candidate(
        charge_spin={"status": "confirmed", "charge": 0, "multiplicity": 1},
        structure_status="frozen_hashed",
        admission="admitted",
        missing=[],
        license_record="test fixture",
        independent_seed_records=[seed_record(parent_system_id="other-parent")],
    )
    report = audit(registry([row]))
    result = report["candidate_reports"][0]

    assert result["independent_seed_ready"] is False
    assert "seed_parent_mismatch[0]" in result["blocking_codes"]
    assert result["seed_validation_errors"] == ["seed_parent_mismatch[0]"]



def test_seed_electronic_state_must_match_confirmed_candidate_state():
    row = candidate(
        charge_spin={"status": "confirmed", "charge": 0, "multiplicity": 1},
        structure_status="frozen_hashed",
        admission="admitted",
        missing=[],
        license_record="test fixture",
        independent_seed_records=[seed_record(charge=1)],
    )
    report = audit(registry([row]))
    result = report["candidate_reports"][0]

    assert result["independent_seed_ready"] is False
    assert "seed_charge_mismatch[0]" in result["blocking_codes"]


def test_duplicate_candidate_ids_are_rejected():
    first = candidate()
    second = deepcopy(first)
    with pytest.raises(OriginReadinessError, match="candidate_id values must be unique"):
        audit(registry([first, second]))


def test_declared_runnable_count_must_match_validated_candidates():
    with pytest.raises(OriginReadinessError, match="runnable_input_count"):
        audit(registry([candidate()], runnable_input_count=1))
