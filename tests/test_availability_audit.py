from __future__ import annotations

import json

from xtbflow.data import audit_task_availability
from xtbflow.data.records import PublicRecord


def record(**overrides):
    values = dict(
        source_dataset="demo", source_revision="r1", record_id="r1",
        parent_reaction_id="parent-1", split_group="family-1",
        geometry_locator="file.xyz#1", geometry_hash="abc", geometry_origin="independent_initial_guess",
        reference_protocol_id="proto", units={"coordinates": "angstrom", "energy": "hartree", "forces": "hartree/angstrom"},
        available_label_mask={
            "event": True, "geometry": True, "energy": True, "forces": True,
            "same_geometry_e_f": True, "paired_identity": True,
        },
        charge_spin_evidence="explicit_source", license_record="reviewed",
        pretraining_overlap_audit="pass", record_status="observed", admission="train",
        input_data={"atomic_numbers": [8], "reactant_coordinates": [[0, 0, 0]], "charge": 0, "multiplicity": 1},
        label_data={"event": {"edits": []}, "ts_geometry": [[0, 0, 0]], "energy": -1.0, "forces": [[0, 0, 0]]},
        claim_limit="diagnostic only",
    )
    values.update(overrides)
    return PublicRecord(**values)


def test_audit_separates_declared_labels_from_formal_admission():
    rows = [
        record(record_id="a", parent_reaction_id="parent-1"),
        record(
            record_id="b", parent_reaction_id="parent-1", admission="quarantine",
            quarantine_reasons=("charge_unverified",),
        ),
        record(
            record_id="c", parent_reaction_id="unknown", admission="quarantine",
            available_label_mask={"energy": True, "forces": True},
            quarantine_reasons=("mapping_failed",),
        ),
    ]
    audit = audit_task_availability(rows)

    assert audit.record_count == 3
    assert audit.unique_record_count == 3
    assert audit.parent_reaction_count == 2
    assert audit.identified_parent_reaction_count == 1
    assert audit.admitted_record_count == 1
    assert audit.quarantined_record_count == 2
    assert audit.task_stats["paired_joint"].declared_records == 2
    assert audit.task_stats["paired_joint"].admitted_records == 1
    assert audit.task_stats["paired_joint"].declared_parent_reactions == 1
    assert audit.task_stats["energy_force"].declared_records == 2
    assert audit.task_stats["energy_force"].admitted_records == 1
    assert audit.quarantine_reason_counts == {"charge_unverified": 1, "mapping_failed": 1}


def test_audit_does_not_infer_pairing_from_energy_and_force_presence():
    row = record(available_label_mask={"energy": True, "forces": True})
    audit = audit_task_availability([row])
    assert audit.task_stats["energy_force"].declared_records == 0


def test_audit_fingerprint_and_json_are_deterministic(tmp_path):
    audit = audit_task_availability([record()])
    output = tmp_path / "audit.json"
    audit.write_json(output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "xtbflow-task-availability-audit/v1"
    assert payload["audit_fingerprint"] == audit.fingerprint
    assert payload["task_stats"]["event_only"]["admitted_records"] == 1
