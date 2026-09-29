from __future__ import annotations

from pathlib import Path
import json

import pytest

from xtbflow.data.records import CHNOS_ELEMENTS, PublicRecord, RecordPolicy, filter_records, load_jsonl, write_jsonl
from xtbflow.data.splits import SplitError, assign_group_splits, audit_no_group_leakage, admitted_records


def record(**overrides):
    values = dict(
        source_dataset="demo", source_revision="r1", record_id="r1",
        parent_reaction_id="parent-1", split_group="family-1",
        geometry_locator="file.xyz#1", geometry_hash="abc", geometry_origin="independent_initial_guess",
        reference_protocol_id="proto", units={"coordinates": "angstrom", "energy": "hartree", "forces": "hartree/angstrom"},
        available_label_mask={"energy": True, "forces": True}, charge_spin_evidence="explicit_source",
        license_record="reviewed", pretraining_overlap_audit="pass", record_status="observed",
        admission="train", input_data={"atomic_numbers": [8, 1, 1], "reactant_coordinates": [[0, 0, 0], [1, 0, 0], [-1, 0, 0]], "charge": 0, "multiplicity": 1},
        label_data={"energy": -1.0, "ts_geometry": [[0, 0, 0]]}, claim_limit="diagnostic only",
    )
    values.update(overrides)
    return PublicRecord(**values)


def test_product_and_ts_labels_do_not_change_reactant_view():
    first = record()
    changed = record(label_data={"energy": -9.0, "ts_geometry": [[3, 2, 1]], "product": "different"})
    assert first.reactant_input() == changed.reactant_input()
    assert first.input_fingerprint() == changed.input_fingerprint()


def test_forbidden_target_feature_in_reactant_input_is_rejected():
    with pytest.raises(ValueError, match="target-derived"):
        record(input_data={"atomic_numbers": [6], "product_coordinates": [[0, 0, 0]]})


def test_unknown_state_cannot_be_admitted():
    with pytest.raises(ValueError, match="unresolved"):
        record(charge_spin_evidence="unknown")


def test_group_split_keeps_groups_together_and_quarantine_out():
    records = [record(record_id="a", split_group="g1"), record(record_id="b", split_group="g1"), record(record_id="c", split_group="unknown", parent_reaction_id="unknown", admission="quarantine")]
    assignments = assign_group_splits(records, seed="test")
    assert assignments["a"] == assignments["b"]
    assert assignments["c"] == "quarantine"
    audit_no_group_leakage(records, assignments)
    assert len(admitted_records(records, assignments)) <= 2


def test_split_audit_rejects_manual_leakage():
    records = [record(record_id="a", split_group="same"), record(record_id="b", split_group="same")]
    with pytest.raises(SplitError):
        audit_no_group_leakage(records, {"a": "train", "b": "test"})


def test_public_manifest_preserves_quarantine_and_reconstructs_spice_pilot_splits():
    rows = load_jsonl(Path("data/manifests/public_records.v1.jsonl"))
    historical = [row for row in rows if row.source_dataset != "spice2_openff_v1.1_pilot"]
    spice = [row for row in rows if row.source_dataset == "spice2_openff_v1.1_pilot"]
    assert len(rows) == 262
    assert len(historical) == 6 and all(row.admission == "quarantine" for row in historical)
    assert len(spice) == 256
    assignments = assign_group_splits(
        rows,
        seed="spice2-openff-pilot-v1-7",
        ratios={"train": 0.75, "validation": 0.125, "test": 0.125},
    )
    audit_no_group_leakage(rows, assignments)
    assert all(assignments[row.record_id] == row.admission for row in spice)
    assert len(admitted_records(rows, assignments)) == 256
    groups = {}
    for row in spice:
        groups.setdefault(row.split_group, []).append(row)
    assert len(groups) == 64
    assert {len(group) for group in groups.values()} == {4}
    parent_splits = {"train": 0, "validation": 0, "test": 0}
    for group in groups.values():
        split_values = {row.admission for row in group}
        assert len(split_values) == 1
        parent_splits[split_values.pop()] += 1
    assert parent_splits == {"train": 48, "validation": 8, "test": 8}


def test_record_policy_quarantines_out_of_scope_and_parity_rows_deterministically():
    valid = record(record_id="valid")
    out_of_scope = record(
        record_id="out-of-scope",
        input_data={"atomic_numbers": [8, 1, 9], "reactant_coordinates": [[0, 0, 0], [1, 0, 0], [-1, 0, 0]], "charge": 0, "multiplicity": 1},
    )
    parity_mismatch = record(
        record_id="parity-mismatch",
        input_data={"atomic_numbers": [1], "reactant_coordinates": [[0, 0, 0]], "charge": 0, "multiplicity": 1},
    )
    policy = RecordPolicy(allowed_elements=CHNOS_ELEMENTS)
    result = filter_records([parity_mismatch, valid, out_of_scope], policy)

    assert [row.record_id for row in result.accepted] == ["valid"]
    assert [row.record_id for row in result.quarantined] == ["out-of-scope", "parity-mismatch"]
    assert result.policy_digest == policy.digest
    assert result.quarantined[0].admission == "quarantine"
    assert result.quarantined[0].quarantine_reasons == ("elements_out_of_scope",)
    assert result.quarantined[1].quarantine_reasons == ("electron_spin_parity_mismatch",)


def test_public_record_rejects_boolean_charge_and_multiplicity():
    base = record(admission="quarantine")
    for key in ("charge", "multiplicity"):
        payload = base.to_dict()
        payload["input_data"] = dict(base.input_data)
        payload["input_data"][key] = True
        with pytest.raises(ValueError, match="strict integer"):
            PublicRecord.from_dict(payload)


def test_jsonl_load_rejects_duplicate_ids_and_write_is_sorted(tmp_path):
    first = record(record_id="z")
    second = record(record_id="a")
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [first, second])
    assert [row.record_id for row in load_jsonl(path)] == ["a", "z"]

    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(json.dumps(first.to_dict()) + "\n" + json.dumps(first.to_dict()) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate public record_id"):
        load_jsonl(duplicate)
