from __future__ import annotations

from pathlib import Path

import pytest

from xtbflow.data.records import PublicRecord, load_jsonl
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


def test_source_audit_manifest_is_explicitly_quarantined():
    rows = load_jsonl(Path("data/manifests/public_records.v1.jsonl"))
    assert len(rows) == 6
    assert all(row.admission == "quarantine" for row in rows)
    assert not admitted_records(rows, assign_group_splits(rows))
