from __future__ import annotations

from copy import deepcopy
import json

import pytest

from xtbflow.data.campaign import (
    CampaignManifest,
    CampaignManifestError,
    canonical_hash,
    load_campaign_manifest,
    write_campaign_manifest,
)


def _payload(*, status: str = "development_screening") -> dict:
    source = {
        "locator": "tests/fixtures/w11",
        "revision": "fixture-v1",
        "sha256": "1" * 64,
        "license_record": "test-only",
    }
    atom_pool = [
        {
            "component_id": "substrate-1",
            "role": "substrate",
            "species_label": "synthetic substrate",
            "explicit_hydrogens": True,
            "composition": {"C": 1, "N": 1, "H": 3},
            "charge": 0,
            "multiplicity": 1,
            "input_locator": "tests/fixtures/w11/substrate.xyz",
            "input_sha256": "2" * 64,
            "license_record": "test-only",
        },
        {
            "component_id": "water-1",
            "role": "water",
            "species_label": "synthetic water A",
            "explicit_hydrogens": True,
            "composition": {"H": 2, "O": 1},
            "charge": 0,
            "multiplicity": 1,
            "input_locator": "tests/fixtures/w11/water-a.xyz",
            "input_sha256": "3" * 64,
            "license_record": "test-only",
        },
        {
            "component_id": "water-2",
            "role": "water",
            "species_label": "synthetic water B",
            "explicit_hydrogens": True,
            "composition": {"H": 2, "O": 1},
            "charge": 0,
            "multiplicity": 1,
            "input_locator": "tests/fixtures/w11/water-b.xyz",
            "input_sha256": "4" * 64,
            "license_record": "test-only",
        },
    ]
    candidates = [
        {
            "candidate_id": "candidate-1",
            "seed_id": "seed-1",
            "parent_system_id": "parent-1",
            "family_id": "family-1",
            "geometry_locator": "tests/fixtures/w11/candidate.xyz",
            "geometry_sha256": "5" * 64,
            "charge": 0,
            "multiplicity": 1,
            "component_ids": ["substrate-1", "water-1", "water-2"],
            "water_roles": {"water-1": "participating", "water-2": "spectator"},
            "input_origin": "independent_reactant",
            "provenance": "synthetic fixture",
            "license_record": "test-only",
            "status": "development",
        }
    ]
    input_fingerprint = canonical_hash(
        {"source": source, "atom_pool": atom_pool, "candidates": candidates}
    )
    return {
        "schema_version": "xtbflow-w11-campaign/v1",
        "campaign_id": "campaign-1",
        "status": status,
        "description": "synthetic W11 fixture",
        "source": source,
        "atom_pool": atom_pool,
        "candidates": candidates,
        "reference_protocol": {
            "protocol_id": "fixture-protocol-v1",
            "protocol_sha256": "6" * 64,
            "method": "synthetic",
            "software": "fixture",
            "version": "1",
            "units": {"coordinates": "angstrom", "energy": "hartree"},
        },
        "budget": {
            "calculator_calls": 10,
            "wall_seconds": 60,
            "cpu_core_hours": 0.1,
            "gpu_hours": 0,
        },
        "deduplication": {
            "keys": ["parent_system_id", "family_id", "geometry_sha256"],
            "within_parent_system": True,
            "cross_strategy": True,
        },
        "comparisons": [
            {
                "comparison_id": "rule-fixed",
                "proposal_mode": "rule",
                "physics_mode": "fixed",
                "candidate_ids": ["candidate-1"],
                "budget_id": "fixture",
                "input_fingerprint": input_fingerprint,
            },
            {
                "comparison_id": "joint-fixed",
                "proposal_mode": "joint",
                "physics_mode": "fixed",
                "candidate_ids": ["candidate-1"],
                "budget_id": "fixture",
                "input_fingerprint": input_fingerprint,
            },
        ],
        "claim_limit": "Synthetic contract fixture only.",
        "frozen_at": None,
    }


def test_development_manifest_round_trips_with_shared_input_fingerprint(tmp_path):
    manifest = CampaignManifest.from_dict(_payload())
    assert manifest.input_fingerprint() == manifest.comparisons[0]["input_fingerprint"]
    assert manifest.comparisons[0]["candidate_ids"] == ["candidate-1"]

    path = tmp_path / "campaign.json"
    write_campaign_manifest(path, manifest)
    loaded = load_campaign_manifest(path)
    assert loaded.to_dict() == manifest.to_dict()


def test_frozen_manifest_requires_resolved_water_roles_and_timestamp():
    payload = _payload(status="frozen")
    payload["frozen_at"] = "2026-11-08T00:00:00+00:00"
    with pytest.raises(CampaignManifestError, match="unresolved"):
        payload["candidates"][0]["water_roles"]["water-2"] = "unresolved"
        CampaignManifest.from_dict(payload)

    payload = _payload(status="frozen")
    with pytest.raises(CampaignManifestError, match="frozen_at"):
        CampaignManifest.from_dict(payload)


def test_quarantine_keeps_unknown_electronic_state_explicit():
    payload = _payload()
    payload["status"] = "quarantine"
    payload["candidates"][0].update(
        {
            "status": "quarantine",
            "geometry_sha256": "unknown",
            "charge": "unknown",
            "multiplicity": "unresolved",
            "water_roles": {"water-1": "unresolved", "water-2": "spectator"},
            "missing": ["geometry hash", "electronic state"],
        }
    )
    fingerprint = canonical_hash(
        {
            "source": payload["source"],
            "atom_pool": payload["atom_pool"],
            "candidates": payload["candidates"],
        }
    )
    for comparison in payload["comparisons"]:
        comparison["input_fingerprint"] = fingerprint
    manifest = CampaignManifest.from_dict(payload)
    candidate = manifest.candidates[0]
    assert candidate["geometry_sha256"] == "unknown"
    assert candidate["charge"] == "unknown"
    assert candidate["multiplicity"] == "unresolved"


def test_target_derived_fields_and_mismatched_comparison_inputs_are_rejected():
    payload = _payload()
    target = deepcopy(payload)
    target["candidates"][0]["product"] = {"atoms": []}
    with pytest.raises(CampaignManifestError, match="target-derived"):
        CampaignManifest.from_dict(target)

    mismatch = deepcopy(payload)
    mismatch["comparisons"][1]["candidate_ids"] = ["other-candidate"]
    with pytest.raises(CampaignManifestError, match="common input pool"):
        CampaignManifest.from_dict(mismatch)

    missing_water = deepcopy(payload)
    missing_water["candidates"][0]["component_ids"] = ["substrate-1", "water-1"]
    missing_water["candidates"][0]["water_roles"] = {"water-1": "participating"}
    fingerprint = canonical_hash(
        {
            "source": missing_water["source"],
            "atom_pool": missing_water["atom_pool"],
            "candidates": missing_water["candidates"],
        }
    )
    for comparison in missing_water["comparisons"]:
        comparison["input_fingerprint"] = fingerprint
    with pytest.raises(CampaignManifestError, match="fixed atom pool"):
        CampaignManifest.from_dict(missing_water)


def test_nested_case_and_separator_variants_are_rejected():
    for key in ("Product", "referenceTS", "sealed-test-label", "productCoordinates"):
        payload = _payload()
        payload["candidates"][0]["metadata"] = {"nested": [{key: "hidden"}]}
        with pytest.raises(CampaignManifestError, match="target-derived"):
            CampaignManifest.from_dict(payload)


def test_fixture_is_valid_json_and_has_no_target_fields():
    with open("configs/campaigns/w11_campaign_fixture.json", encoding="utf-8") as handle:
        payload = json.load(handle)
    manifest = CampaignManifest.from_dict(payload)
    assert manifest.status == "development_screening"
    assert not any("product" in row for row in manifest.candidates)
