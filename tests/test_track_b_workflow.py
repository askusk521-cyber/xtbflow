from __future__ import annotations

from dataclasses import replace
import json
import math

import pytest

from xtbflow.data.attempts import (
    SearchAttempt,
    append_attempt_jsonl,
    load_attempt_jsonl,
    latest_attempt_versions,
)
from scripts.audit_search_attempts import build_report as build_attempt_audit_report
from xtbflow.data.independent_seeds import (
    IndependentReactantSeed,
    SeedLeakageError,
    audit_seed_leakage,
    load_seed_jsonl,
    write_seed_jsonl,
)
from xtbflow.data.track_b import (
    TrackBLeakageError,
    TrackBRecord,
    audit_track_b_leakage,
    filter_track_b_records,
    load_track_b_jsonl,
    write_track_b_jsonl,
)


def track_record(**overrides) -> TrackBRecord:
    values = {
        "record_id": "record-1",
        "source_dataset": "demo",
        "source_revision": "revision-1",
        "source_record_id": "source-1",
        "source_asset_sha256": "a" * 64,
        "source_record_sha256": "6" * 64,
        "license_record": "reviewed",
        "parent_reaction_id": "parent-1",
        "family_id": "family-1",
        "split_group": "group-1",
        "atoms": {"atomic_numbers": [1, 1], "map_ids": [0, 1]},
        "reactant": {
            "graph_locator": "graphs/demo.json#1",
            "graph_sha256": "b" * 64,
            "coordinates_locator": "coords/demo.xyz#1",
            "coordinates_sha256": "c" * 64,
            "coordinate_unit": "angstrom",
            "input_origin": "independent_reactant",
            "solvent_selection_origin": "none",
            "charge": 0,
            "multiplicity": 1,
            "charge_spin_evidence": "explicit_source",
            "microstate_id": "microstate-1",
            "geometry_generation_protocol": "reactant-v1",
            "geometry_generation_protocol_locator": "protocols/reactant-v1.json",
            "geometry_generation_protocol_sha256": "5" * 64,
            "declared_intent": None,
        },
        "event_label": {
            "locator": "labels/events.json#1",
            "sha256": "d" * 64,
            "representation": "mapped_bond_edits_v1",
            "evidence": "observed",
        },
        "product_label": {
            "graph_locator": "labels/product.json#1",
            "graph_sha256": "7" * 64,
            "graph_evidence": "observed",
            "coordinates_locator": "labels/product.xyz#1",
            "coordinates_sha256": "8" * 64,
            "coordinate_unit": "angstrom",
            "coordinate_evidence": "observed",
            "atom_order_matches_input": True,
        },
        "ts_geometry": {
            "locator": "labels/ts.xyz#1",
            "sha256": "e" * 64,
            "coordinate_unit": "angstrom",
            "evidence": "observed",
            "atom_order_matches_input": True,
        },
        "reference_protocol": {
            "protocol_id": "reference-v1",
            "protocol_locator": "protocols/reference-v1.json",
            "protocol_sha256": "f" * 64,
            "method": "documented-reference-method",
            "software": "documented-reference-software",
            "version": "pinned-test",
            "environment": "gas_phase",
        },
        "pretraining_overlap_audit": "pass",
        "admission": "development_train",
        "claim_limit": "development evidence only",
    }
    values.update(overrides)
    return TrackBRecord(**values)


def seed_record(**overrides) -> IndependentReactantSeed:
    values = {
        "seed_id": "seed-1",
        "parent_system_id": "parent-1",
        "family_id": "family-1",
        "mapped_explicit_h_smiles": "[H:1][H:2]",
        "geometry_path": "inputs/h2.xyz",
        "geometry_sha256": "1" * 64,
        "charge": 0,
        "multiplicity": 1,
        "charge_spin_evidence": "explicit_source",
        "microstate_description": "neutral H2",
        "stereochemistry_description": "not applicable",
        "solvent_snapshot_id": "none",
        "solvent_selection_origin": "none",
        "reactant_generation_protocol": "seed-protocol-v1",
        "reactant_generation_protocol_locator": "protocols/seed-v1.json",
        "reactant_generation_protocol_sha256": "3" * 64,
        "provenance": "locally generated before search",
        "approved_split_group": "group-1",
        "approved_split_role": "development_train",
        "frozen_at": "2026-09-29T00:00:00Z",
        "license_record": "project-generated",
    }
    values.update(overrides)
    return IndependentReactantSeed(**values)


def calculator_protocol() -> dict[str, object]:
    return {
        "protocol_id": "gfn2-v1",
        "protocol_sha256": "9" * 64,
        "method": "GFN2-xTB",
        "software": "xtb",
        "version": "pinned-test",
        "units": {
            "coordinates": "angstrom",
            "energy": "hartree",
            "forces": "hartree/angstrom",
        },
    }


def attempt_record(**overrides) -> SearchAttempt:
    values = {
        "attempt_id": "attempt-1",
        "attempt_version": 1,
        "seed_id": "seed-1",
        "parent_system_id": "parent-1",
        "family_id": "family-1",
        "proposal_method": "rule-baseline",
        "proposal_revision": "revision-1",
        "random_seed": 7,
        "proposed_event": {"bond_edits": [[0, 1, "form"]]},
        "initial_geometry_locator": "inputs/h2.xyz",
        "initial_geometry_sha256": "1" * 64,
        "evidence_status": "unattempted",
        "created_at": "2026-09-29T00:00:00Z",
    }
    values.update(overrides)
    return SearchAttempt(**values)


def test_track_b_reactant_view_excludes_labels_and_round_trips(tmp_path):
    first = track_record()
    changed = track_record(
        event_label={
            "locator": "labels/other.json#1",
            "sha256": "2" * 64,
            "representation": "other",
            "evidence": "observed",
        },
        product_label={
            "graph_locator": "labels/other-product.json#1",
            "graph_sha256": "0" * 64,
            "graph_evidence": "derived_under_contract",
            "coordinates_locator": None,
            "coordinates_sha256": None,
            "coordinate_unit": None,
            "coordinate_evidence": "unavailable",
            "atom_order_matches_input": None,
        },
        ts_geometry={
            "locator": "labels/other.xyz#1",
            "sha256": "3" * 64,
            "coordinate_unit": "angstrom",
            "evidence": "observed",
            "atom_order_matches_input": True,
        },
    )
    assert first.reactant_view() == changed.reactant_view()
    assert first.input_fingerprint() == changed.input_fingerprint()
    path = tmp_path / "track-b.jsonl"
    write_track_b_jsonl(str(path), [first])
    assert load_track_b_jsonl(str(path)) == [first]


def test_product_supervision_is_explicitly_masked_when_geometry_is_missing():
    unavailable_geometry = {
        "graph_locator": "labels/product.json#1",
        "graph_sha256": "7" * 64,
        "graph_evidence": "observed",
        "coordinates_locator": None,
        "coordinates_sha256": None,
        "coordinate_unit": None,
        "coordinate_evidence": "unavailable",
        "atom_order_matches_input": None,
    }
    assert track_record(product_label=unavailable_geometry).policy_reasons() == ()

    inconsistent = dict(unavailable_geometry)
    inconsistent["coordinates_locator"] = "labels/product.xyz#1"
    with pytest.raises(ValueError, match="violates contract"):
        track_record(product_label=inconsistent)

    missing_graph = dict(unavailable_geometry)
    missing_graph["graph_sha256"] = "unknown"
    with pytest.raises(ValueError, match="violates contract"):
        track_record(product_label=missing_graph)


def test_reference_derived_input_is_quarantined_and_cannot_be_admitted():
    unsafe_reactant = dict(track_record().reactant)
    unsafe_reactant["input_origin"] = "reference_derived"
    with pytest.raises(ValueError, match="violates contract"):
        track_record(reactant=unsafe_reactant)

    quarantined = track_record(
        admission="quarantine",
        reactant=unsafe_reactant,
        quarantine_reasons=("source_requires_review",),
    )
    result = filter_track_b_records([quarantined])
    assert not result.accepted
    assert "reactant_input_target_derived_or_unresolved" in (
        result.quarantined[0].quarantine_reasons
    )


def test_track_b_leakage_audit_blocks_parent_and_family_overlap():
    training = track_record()
    held_out = track_record(
        record_id="record-2",
        source_record_id="source-2",
        source_record_sha256="7" * 64,
        parent_reaction_id="parent-2",
        split_group="group-2",
        admission="development_test",
        reactant={
            **track_record().reactant,
            "coordinates_locator": "coords/demo.xyz#2",
            "coordinates_sha256": "4" * 64,
        },
    )
    with pytest.raises(TrackBLeakageError, match="family_id"):
        audit_track_b_leakage([training, held_out])

    allowed = audit_track_b_leakage(
        [training, held_out],
        strict_family_holdout=False,
    )
    assert allowed["split_counts"] == {"test": 1, "train": 1}

    same_parent = replace(
        held_out,
        parent_reaction_id=training.parent_reaction_id,
        family_id="family-2",
    )
    with pytest.raises(TrackBLeakageError, match="parent_reaction_id"):
        audit_track_b_leakage(
            [training, same_parent],
            strict_family_holdout=False,
        )


def test_independent_seed_rejects_nonindependent_input_and_round_trips(tmp_path):
    with pytest.raises(ValueError, match="independently generated"):
        seed_record(input_origin="reference_derived")
    path = tmp_path / "seeds.jsonl"
    first = seed_record(seed_id="z")
    second = seed_record(seed_id="a", geometry_sha256="2" * 64)
    write_seed_jsonl(str(path), [first, second])
    assert [row.seed_id for row in load_seed_jsonl(str(path))] == ["a", "z"]


def test_search_attempts_are_append_only_and_keep_proposal_separate(tmp_path):
    path = tmp_path / "attempts.jsonl"
    first = attempt_record()
    append_attempt_jsonl(path, first)
    second = attempt_record(
        attempt_version=2,
        evidence_status="valid_alternative",
        calculator_protocols=(calculator_protocol(),),
        calculator_calls={"gfn2-v1": 12},
        observed_event={"bond_edits": [[0, 1, "break"]]},
        raw_log_locator="logs/attempt-1-v2.json",
        raw_log_sha256="8" * 64,
        wall_seconds=4.5,
    )
    append_attempt_jsonl(path, second)
    rows = load_attempt_jsonl(path)
    assert [row.attempt_version for row in rows] == [1, 2]
    assert rows[1].proposed_event != rows[1].observed_event

    with pytest.raises(ValueError, match="must append version 3"):
        append_attempt_jsonl(path, replace(second, attempt_version=4))


def test_unattempted_record_cannot_report_calculator_work():
    with pytest.raises(ValueError, match="cannot report calculator work"):
        attempt_record(
            calculator_protocols=(calculator_protocol(),),
            calculator_calls={"gfn2-v1": 1},
        )


def test_empty_scientific_manifests_are_explicitly_zero_admission():
    assert load_track_b_jsonl(
        "data/manifests/track_b_reaction_ts.v0.1.jsonl"
    ) == []
    assert load_seed_jsonl("data/manifests/independent_seeds.v1.jsonl") == []
    assert load_attempt_jsonl("data/manifests/search_attempts.v1.jsonl") == []
    source_audit = json.loads(
        open(
            "docs/evidence/track_b_source_audit_20260929.json",
            encoding="utf-8",
        ).read()
    )
    manifest_audit = json.loads(
        open(
            "docs/evidence/track_b_manifest_audit_20260929.json",
            encoding="utf-8",
        ).read()
    )
    seed_audit = json.loads(
        open(
            "docs/evidence/independent_seed_manifest_audit_20260929.json",
            encoding="utf-8",
        ).read()
    )
    attempt_audit = json.loads(
        open(
            "docs/evidence/search_attempt_manifest_audit_20260929.json",
            encoding="utf-8",
        ).read()
    )
    assert source_audit["admitted_source_count"] == 0
    assert manifest_audit["admitted_record_count"] == 0
    assert seed_audit["record_count"] == 0
    assert seed_audit["search_may_start"] is False
    assert attempt_audit["attempt_count"] == 0
    assert attempt_audit["total_calculator_calls"] == 0


def test_exact_input_leakage_ignores_record_identity():
    training = track_record()
    held_out = track_record(
        record_id="record-other",
        source_record_id="source-other",
        source_record_sha256="7" * 64,
        parent_reaction_id="parent-other",
        family_id="family-other",
        split_group="group-other",
        admission="development_test",
    )
    assert training.reactant_view() == held_out.reactant_view()
    assert training.input_fingerprint() == held_out.input_fingerprint()
    with pytest.raises(TrackBLeakageError, match="input_fingerprint"):
        audit_track_b_leakage(
            [training, held_out],
            strict_family_holdout=False,
        )


def test_exact_input_fingerprint_excludes_declared_intent_provenance():
    first_intent = {
        "permission_mode": "declared_center",
        "reactive_map_ids": [0],
        "net_bond_edits": [],
        "provenance": "proposal-source-a",
        "uses_reference_labels": False,
    }
    second_intent = {**first_intent, "provenance": "proposal-source-b"}
    first = track_record(
        reactant={**track_record().reactant, "declared_intent": first_intent}
    )
    second = track_record(
        record_id="record-provenance",
        source_record_id="source-provenance",
        parent_reaction_id="parent-provenance",
        family_id="family-provenance",
        split_group="group-provenance",
        admission="development_test",
        reactant={**track_record().reactant, "declared_intent": second_intent},
    )
    assert first.input_fingerprint() == second.input_fingerprint()
    with pytest.raises(TrackBLeakageError, match="input_fingerprint"):
        audit_track_b_leakage([first, second], strict_family_holdout=False)


def test_source_record_payload_cannot_cross_splits():
    training = track_record()
    held_out = track_record(
        record_id="record-other",
        source_record_id="source-other",
        parent_reaction_id="parent-other",
        family_id="family-other",
        split_group="group-other",
        admission="development_test",
        reactant={
            **track_record().reactant,
            "coordinates_locator": "coords/other.xyz",
            "coordinates_sha256": "7" * 64,
        },
    )
    with pytest.raises(TrackBLeakageError, match="source_record_bytes"):
        audit_track_b_leakage(
            [training, held_out],
            strict_family_holdout=False,
        )


def test_declared_intent_cannot_use_reference_labels():
    unsafe = dict(track_record().reactant)
    unsafe["declared_intent"] = {
        "permission_mode": "declared_center",
        "reactive_map_ids": [0],
        "net_bond_edits": [],
        "provenance": "derived from reference TS",
        "uses_reference_labels": True,
    }
    with pytest.raises(ValueError, match="violates contract"):
        track_record(reactant=unsafe)

    safe = dict(track_record().reactant)
    safe["declared_intent"] = {
        "permission_mode": "declared_center",
        "reactive_map_ids": [0],
        "net_bond_edits": [],
        "provenance": "declared before label access",
        "uses_reference_labels": False,
    }
    assert track_record(reactant=safe).policy_reasons() == ()


def test_confirmatory_record_requires_independent_reactant_input():
    reactant = dict(track_record().reactant)
    reactant["input_origin"] = "declared_reactant_endpoint"
    with pytest.raises(ValueError, match="violates contract"):
        track_record(admission="confirmatory", reactant=reactant)


def test_track_b_load_rejects_duplicate_record_ids(tmp_path):
    path = tmp_path / "duplicate-track-b.jsonl"
    payload = json.dumps(track_record().to_dict(), sort_keys=True)
    path.write_text(payload + "\n" + payload + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate Track-B record_id"):
        load_track_b_jsonl(str(path))


def test_seed_requires_atom_maps_timezone_and_protocol_hash():
    with pytest.raises(ValueError, match="atom-map indices"):
        seed_record(mapped_explicit_h_smiles="[H][H]")
    with pytest.raises(ValueError, match="explicit timezone"):
        seed_record(frozen_at="2026-09-29T00:00:00")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        seed_record(reactant_generation_protocol_sha256="unknown")


def test_seed_fingerprint_ignores_bookkeeping_identity():
    first = seed_record()
    second = seed_record(
        seed_id="seed-other",
        parent_system_id="parent-other",
        family_id="family-other",
        approved_split_group="group-other",
        frozen_at="2026-09-30T00:00:00Z",
    )
    assert first.input_fingerprint() == second.input_fingerprint()


def test_attempt_requires_complete_protocol_and_hashed_log():
    with pytest.raises(ValueError, match="missing required fields"):
        attempt_record(
            evidence_status="attempted_unresolved",
            calculator_protocols=({"protocol_id": "gfn2-v1"},),
        )
    with pytest.raises(ValueError, match="hashed raw log"):
        attempt_record(
            evidence_status="attempted_unresolved",
            calculator_protocols=(calculator_protocol(),),
            calculator_calls={"gfn2-v1": 1},
        )
    with pytest.raises(ValueError, match="at least one calculator call"):
        attempt_record(
            evidence_status="se_validated",
            calculator_protocols=(calculator_protocol(),),
            calculator_calls={"gfn2-v1": 0},
        )


def test_attempt_append_rejects_rewritten_proposal(tmp_path):
    path = tmp_path / "immutable-attempts.jsonl"
    append_attempt_jsonl(path, attempt_record())
    rewritten = attempt_record(
        attempt_version=2,
        proposed_event={"bond_edits": [[0, 1, "break"]]},
        evidence_status="attempted_unresolved",
        calculator_protocols=(calculator_protocol(),),
        calculator_calls={"gfn2-v1": 1},
        raw_log_locator="logs/attempt-1-v2.json",
        raw_log_sha256="8" * 64,
    )
    with pytest.raises(ValueError, match="rewrites immutable fields"):
        append_attempt_jsonl(path, rewritten)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("calculator_calls", {"gfn2-v1": 2}, "calculator calls decrease"),
        ("wall_seconds", 2.0, "wall_seconds decrease"),
        ("retry_count", 1, "retry_count decrease"),
    ],
)
def test_attempt_append_rejects_decreasing_cumulative_cost_before_write(
    tmp_path, field, value, message
):
    path = tmp_path / f"decreasing-{field}.jsonl"
    first = attempt_record(
        evidence_status="attempted_unresolved",
        calculator_protocols=(calculator_protocol(),),
        calculator_calls={"gfn2-v1": 3},
        raw_log_locator="logs/attempt-1-v1.json",
        raw_log_sha256="7" * 64,
        wall_seconds=3.0,
        retry_count=2,
    )
    append_attempt_jsonl(path, first)
    original = path.read_text(encoding="utf-8")
    update = attempt_record(
        attempt_version=2,
        evidence_status="attempted_unresolved",
        calculator_protocols=(calculator_protocol(),),
        calculator_calls={"gfn2-v1": 3},
        raw_log_locator="logs/attempt-1-v2.json",
        raw_log_sha256="8" * 64,
        wall_seconds=3.0,
        retry_count=2,
    )
    update = replace(update, **{field: value})

    with pytest.raises(ValueError, match=message):
        append_attempt_jsonl(path, update)
    assert path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("wall_seconds", [math.nan, math.inf, -math.inf])
def test_attempt_rejects_nonfinite_wall_seconds(wall_seconds):
    with pytest.raises(ValueError, match="wall_seconds must be finite"):
        attempt_record(wall_seconds=wall_seconds)


def test_attempt_load_rejects_version_gaps(tmp_path):
    path = tmp_path / "gap-attempts.jsonl"
    rows = [attempt_record(), attempt_record(attempt_version=3)]
    path.write_text(
        "".join(
            json.dumps(row.to_dict(), sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="version gap"):
        load_attempt_jsonl(path)


def test_attempt_load_rejects_decreasing_cumulative_cost(tmp_path):
    path = tmp_path / "decreasing-attempts.jsonl"
    rows = [
        attempt_record(
            evidence_status="attempted_unresolved",
            calculator_protocols=(calculator_protocol(),),
            calculator_calls={"gfn2-v1": 3},
            raw_log_locator="logs/attempt-1-v1.json",
            raw_log_sha256="7" * 64,
            wall_seconds=3.0,
        ),
        attempt_record(
            attempt_version=2,
            evidence_status="attempted_unresolved",
            calculator_protocols=(calculator_protocol(),),
            calculator_calls={"gfn2-v1": 2},
            raw_log_locator="logs/attempt-1-v2.json",
            raw_log_sha256="8" * 64,
            wall_seconds=2.0,
        ),
    ]
    path.write_text(
        "".join(json.dumps(row.to_dict(), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cumulative calculator calls decrease"):
        load_attempt_jsonl(path)


def test_attempt_audit_uses_latest_cumulative_snapshot(tmp_path):
    path = tmp_path / "versioned-attempts.jsonl"
    first = attempt_record(
        evidence_status="attempted_unresolved",
        calculator_protocols=(calculator_protocol(),),
        calculator_calls={"gfn2-v1": 3},
        raw_log_locator="logs/attempt-1-v1.json",
        raw_log_sha256="7" * 64,
        wall_seconds=1.25,
    )
    second = attempt_record(
        attempt_version=2,
        evidence_status="valid_alternative",
        calculator_protocols=(calculator_protocol(),),
        calculator_calls={"gfn2-v1": 12},
        observed_event={"bond_edits": [[0, 1, "break"]]},
        raw_log_locator="logs/attempt-1-v2.json",
        raw_log_sha256="8" * 64,
        wall_seconds=4.5,
    )
    append_attempt_jsonl(path, first)
    append_attempt_jsonl(path, second)

    rows = load_attempt_jsonl(path)
    assert [row.attempt_version for row in latest_attempt_versions(rows)] == [2]
    report = build_attempt_audit_report(path, repository_base_commit="test")
    assert report["version_record_count"] == 2
    assert report["cost_semantics"] == "latest_cumulative_snapshot_per_attempt"
    assert report["status_counts"] == {"valid_alternative": 1}
    assert report["total_calculator_calls"] == 12
    assert report["total_wall_seconds"] == pytest.approx(4.5)


def test_json_schemas_match_serialized_record_fields():
    cases = [
        (
            "schemas/track_b_reaction_ts.schema.json",
            track_record().to_dict(),
        ),
        (
            "schemas/independent_reactant_seed_record.schema.json",
            seed_record().to_dict(),
        ),
        (
            "schemas/search_attempt.schema.json",
            attempt_record().to_dict(),
        ),
    ]
    for schema_path, payload in cases:
        schema = json.loads(open(schema_path, encoding="utf-8").read())
        assert set(payload) == set(schema["required"])
        assert set(payload) == set(schema["properties"])


def test_seed_split_firewall_blocks_family_parent_and_exact_input_overlap():
    training = seed_record()
    held_out = seed_record(
        seed_id="seed-other",
        parent_system_id="parent-other",
        family_id="family-other",
        approved_split_group="group-other",
        approved_split_role="development_test",
    )
    with pytest.raises(SeedLeakageError, match="input_fingerprint"):
        audit_seed_leakage(
            [training, held_out],
            strict_family_holdout=False,
        )

    distinct_input = seed_record(
        seed_id="seed-distinct",
        parent_system_id="parent-distinct",
        family_id="family-1",
        approved_split_group="group-distinct",
        approved_split_role="development_test",
        geometry_sha256="4" * 64,
    )
    with pytest.raises(SeedLeakageError, match="family_id"):
        audit_seed_leakage([training, distinct_input])

    report = audit_seed_leakage(
        [training, replace(distinct_input, family_id="family-distinct")]
    )
    assert report["split_counts"] == {
        "development_test": 1,
        "development_train": 1,
    }
