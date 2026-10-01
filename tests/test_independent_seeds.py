from __future__ import annotations

import json
from pathlib import Path

import pytest

from xtbflow.data.attempts import AttemptRecord, ProposalRecord, ValidationRecord
from xtbflow.data.independent_seeds import IndependentSeed, SeedManifest, build_seed_manifest


def seed(**overrides):
    payload = {
        "seed_id": "p1:seed-000",
        "parent_id": "p1",
        "parent_group": "family-1",
        "source_uri": "https://example.org/source",
        "source_revision": "v1",
        "source_hash": "sha256:source",
        "license_ref": "license:fixture",
        "symbols": ("C", "O", "H"),
        "coordinates": ((0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (0.0, 0.0, 1.0)),
        "charge": 0,
        "multiplicity": 1,
        "charge_spin_evidence": "source:state-table",
        "microstate_id": "neutral-0",
        "microstate_evidence": "source:microstate",
        "conformer_id": "conf-0",
        "approach_id": "approach-0",
        "environment_id": "none",
        "environment_provenance": "reactant-side:none",
        "declared_intent": "bounded-pilot",
    }
    payload.update(overrides)
    return IndependentSeed(**payload)


def test_seed_rejects_target_fields_and_out_of_scope_elements():
    with pytest.raises(ValueError, match="target-derived"):
        seed(metadata={"reference_ts": "secret"})
    with pytest.raises(ValueError, match="CHNOS"):
        seed(symbols=("C", "P", "H"))
    with pytest.raises(ValueError, match="explicit hydrogen"):
        seed(symbols=("C", "O"), coordinates=((0, 0, 0), (1, 0, 0)))


def test_seed_requires_explicit_singlet_and_preserves_charge():
    with pytest.raises(ValueError, match="multiplicity=1"):
        seed(multiplicity=2)
    with pytest.raises(ValueError, match="integer"):
        seed(charge=False)
    assert seed(charge=1).reactant_input()["charge"] == 1


def test_atom_permutation_keeps_inventory_but_changes_input_fingerprint():
    first = seed()
    permuted = seed(
        seed_id="p1:seed-001",
        symbols=("O", "C", "H"),
        coordinates=((1.2, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )
    assert first.atom_inventory_hash == permuted.atom_inventory_hash
    assert first.input_fingerprint != permuted.input_fingerprint


def test_manifest_freeze_is_deterministic_and_caps_each_parent():
    config = {
        "schema": "xtbflow-independent-pilot/v1",
        "generation_version": "independent-seeds-v1",
        "scope": {"allowed_elements": ["C", "H", "N", "O", "S"], "allowed_multiplicities": [1], "max_seeds_per_parent": 2},
        "qualification": {"status": "qualified"},
        "parents": [
            {"parent_id": "p1", "seeds": [seed().to_dict()]},
        ],
    }
    # Config rows are allowed to carry their serialised seed fields, but the
    # builder supplies parent identity and strips derived hashes.
    manifest_a = build_seed_manifest(config, freeze_id="freeze-fixed")
    manifest_b = build_seed_manifest(config, freeze_id="freeze-fixed")
    assert manifest_a.status == "ready"
    assert manifest_a.manifest_hash == manifest_b.manifest_hash
    assert manifest_a.parent_count == 1
    with pytest.raises(ValueError, match="exceeds"):
        SeedManifest(
            generation_version="v1",
            config_hash="hash",
            freeze_id="freeze",
            status="ready",
            seeds=(seed(), seed(seed_id="p1:seed-001"), seed(seed_id="p1:seed-002")),
            max_seeds_per_parent=2,
        )


def test_blocked_empty_config_is_explicit_not_a_scientific_pass():
    config = {
        "schema": "xtbflow-independent-pilot/v1",
        "scope": {"allowed_elements": ["C", "H", "N", "O", "S"], "allowed_multiplicities": [1]},
        "qualification": {"status": "blocked"},
        "parents": [],
    }
    manifest = build_seed_manifest(config, freeze_id="freeze")
    assert manifest.status == "blocked"
    assert "no_source_qualified_reactant_seeds" in manifest.blockers


def test_proposal_attempt_validation_are_separate_and_metered():
    proposal = ProposalRecord(
        "proposal-1",
        "seed-1",
        "strong_rule",
        0,
        {"bond_edits": []},
        {"coordinates": []},
        status="unsupported",
        provenance={"source": "reactant-only"},
    )
    with pytest.raises(ValueError, match="failure_reason"):
        AttemptRecord("attempt-1", "seed-1", proposal.proposal_id, "proposal", "unsupported", "ledger", "proto", "none", 0)
    attempt = AttemptRecord("attempt-1", "seed-1", proposal.proposal_id, "proposal", "unsupported", "ledger", "proto", "none", 0, failure_reason="driver_not_qualified")
    validation = ValidationRecord("validation-1", attempt.attempt_id, None, "unknown", "proto", "not_validated")
    assert proposal.to_dict()["schema"].endswith("proposal/v1")
    assert attempt.to_dict()["calculator_calls"] == 0
    assert validation.to_dict()["observed_event"] is None

