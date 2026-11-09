"""Synthetic handoff tests; no quantum-chemistry calculations are performed."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest
pytest.importorskip("rdkit")

from xtbflow.data.records import canonical_hash
from xtbflow.runtime import LedgerEvent, RunLedger, StageBudget
from xtbflow.governance.live_reference import build_reactant_context, load_live_feedback
from xtbflow.validation.certification import file_hash


def context_fixture(tmp_path):
    xyz = tmp_path / "reactant.xyz"
    xyz.write_text("3\nsynthetic fixture\nO 0 0 0\nH .9572 0 0\nH -.239987 .927297 0\n")
    payload = {"symbols": ["O", "H", "H"], "coordinates": [[0.0, 0.0, 0.0], [.9572, 0.0, 0.0], [-.239987, .927297, 0.0]],
               "charge": 0, "multiplicity": 1, "environment": {"periodic": False, "solvent": "none"}}
    smiles = "[O:1]([H:2])[H:3]"
    record = {
        "admission": "development_train", "license_record": "synthetic test only", "source_asset_sha256": "b" * 64,
        "atoms": {"map_ids": [1, 2, 3], "atomic_numbers": [8, 1, 1]},
        "reactant": {"graph_sha256": canonical_hash({"mapped_smiles": smiles}), "coordinates_sha256": file_hash(xyz),
                     "charge": 0, "multiplicity": 1, "charge_spin_evidence": "explicit synthetic state", "input_origin": "synthetic"},
    }
    return smiles, record, xyz, payload


def write(path, data):
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")


def live_fixture(tmp_path):
    smiles, record, xyz, payload = context_fixture(tmp_path)
    context = build_reactant_context(smiles, record, xyz, payload)
    ctx_path = tmp_path / "context.json"
    write(ctx_path, context)
    cfg = tmp_path / "config.json"
    write(cfg, {"seed": 7, "reactant": payload})
    initial = {"parent_id": "fixture-parent", "attempt_id": "fixture-attempt", "random_seed": 7,
               "generation_step": 0, "geometry": payload["coordinates"], "event": [0.0] * 4,
               "event_mask": [True] * 4, "metadata": {"input_hash": canonical_hash(payload)},
               "provenance": {}, "event_frozen": False}
    initial["snapshot_id"] = canonical_hash(initial)
    arms = []
    for name in ("no_feedback", "physical_feedback"):
        after = deepcopy(initial)
        after.pop("snapshot_id")
        after["generation_step"] = 1
        after["event_frozen"] = True
        if name == "physical_feedback":
            after["geometry"][0][0] += 0.001
            after["event"][0] = 0.1
        after["snapshot_id"] = canonical_hash(after)
        ledger = RunLedger(StageBudget(phase=name, max_calculator_calls=1))
        ledger.append(LedgerEvent(run_id=name, phase=name, calculator="fixture", protocol_id="fixture",
                                 calculator_build_hash="fixture", system_id="fixture-parent", input_hash=canonical_hash(payload),
                                 operation="energy_forces", status="success"), persist_path=tmp_path / f"ledger_{name}.json")
        arms.append({"arm": name, "snapshot_before": initial, "snapshot_after": after,
                     "runtime_ledger": ledger.summary(), "geometry_changed": name == "physical_feedback",
                     "event_changed": name == "physical_feedback", "certification": {"backend_result": {"calculator_calls": 10, "reference_ts": "must-not-cross"}}})
    manifest = {"schema": "xtbflow-real-feedback-loop/v1", "source": {"commit": "a" * 40, "dirty": False},
                "input": payload, "input_hash": canonical_hash(payload), "parent_id": "fixture-parent", "seed": 7,
                "config_sha256": file_hash(cfg), "arms": arms}
    write(tmp_path / "manifest.json", manifest)
    expected = {"manifest_sha256": file_hash(tmp_path / "manifest.json"), "context_sha256": file_hash(ctx_path),
                "config_sha256": file_hash(cfg), "parent_id": "fixture-parent", "seed": 7, "source_commit": "a" * 40}
    return cfg, ctx_path, expected, manifest


def test_context_reuses_mapped_graph_audit_and_ignores_hidden_labels(tmp_path):
    smiles, record, xyz, payload = context_fixture(tmp_path)
    first = build_reactant_context(smiles, record, xyz, payload)
    record.update(product_label={"smiles": "arbitrary answer"}, ts_geometry={"coordinates": "sealed"}, event_label="hidden")
    assert build_reactant_context(smiles, record, xyz, payload) == first
    assert first["atom_ids"] == ["source-map:1", "source-map:2", "source-map:3"]
    assert len(first["reactant_automorphisms"]) == 2
    with pytest.raises(ValueError, match="reactant SMILES only"):
        build_reactant_context(smiles + ">>answer", record, xyz, payload)


def test_context_does_not_default_missing_mapping_or_electronic_state(tmp_path):
    smiles, record, xyz, payload = context_fixture(tmp_path)
    record["atoms"]["map_ids"] = [2, 1, 3]
    with pytest.raises(ValueError, match="mapped source row order"):
        build_reactant_context(smiles, record, xyz, payload)
    record["atoms"]["map_ids"] = [1, 2, 3]
    record["reactant"]["charge_spin_evidence"] = None
    with pytest.raises(ValueError, match="electronic state"):
        build_reactant_context(smiles, record, xyz, payload)


def test_handoff_uses_only_final_snapshot_and_retains_null_proposal(tmp_path):
    cfg, context, expected, manifest = live_fixture(tmp_path)
    report, rows = load_live_feedback(tmp_path, cfg, context, expected)
    assert report["parent_denominator"] == 1 and report["candidate_denominator"] == 2
    for (item, candidate), arm in zip(rows, manifest["arms"]):
        assert item["method"] == arm["arm"]
        assert candidate.coordinates_angstrom == arm["snapshot_after"]["geometry"]
        assert candidate.planned_product_be is None
        assert "must-not-cross" not in json.dumps(candidate.content())
        assert candidate.provenance["snapshot_id"] == arm["snapshot_after"]["snapshot_id"]


def test_handoff_rejects_snapshot_tamper_and_unsettled_calls(tmp_path):
    cfg, context, expected, manifest = live_fixture(tmp_path)
    path = tmp_path / "manifest.json"
    original = deepcopy(manifest)
    manifest["arms"][1]["snapshot_after"]["geometry"][0][0] = 10.0
    write(path, manifest)
    expected["manifest_sha256"] = file_hash(path)
    with pytest.raises(ValueError, match="immutable snapshot hash"):
        load_live_feedback(tmp_path, cfg, context, expected)
    write(path, original)
    expected["manifest_sha256"] = file_hash(path)
    ledger_path = tmp_path / "ledger_no_feedback.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["budget"]["max_calculator_calls"] = 2
    ledger["reservations"] = {"pending": {"calculator_calls": 1}}
    write(ledger_path, ledger)
    with pytest.raises(ValueError, match="unsettled"):
        load_live_feedback(tmp_path, cfg, context, expected)
