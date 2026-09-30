from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from xtbflow.data.reaction_qm import (
    Bond,
    MappedGraph,
    PublicSourceConfig,
    REACTION_QM_SOURCE,
    ReactionQMLoader,
    ReactionQMRecord,
    SourceFile,
    derive_event_label,
    reaction_qm_record_hash,
)
from xtbflow.data.records import PublicRecord
from xtbflow.data.splits import SplitError, audit_no_group_leakage, audit_no_input_fingerprint_leakage


def graphs():
    reactant = MappedGraph({1: "C", 2: "O", 3: "H"}, (Bond(1, 2, 1), Bond(2, 3, 1)))
    product = MappedGraph({1: "C", 2: "O", 3: "H"}, (Bond(1, 2, 2),))
    return reactant, product


def record(**overrides):
    reactant, product = graphs()
    values = dict(
        record_id="RXN_1",
        parent_reaction_id="mother-1",
        reaction_family_id="family-1",
        independent_reactant_system_id="reactants-1",
        repeated_ts_group="ts-1",
        reactant_graph=reactant,
        product_graph=product,
        reactant_coordinates=((0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (1.8, 0.0, 0.0)),
        ts_coordinates=((0.0, 0.0, 0.0), (1.1, 0.1, 0.0), (1.9, 0.0, 0.0)),
        charge=-1,
        multiplicity=1,
        reference_protocol={"method": "B3LYP-D3", "basis": "TZVP", "coordinates": "angstrom", "energy": "hartree"},
        event_label=derive_event_label(reactant, product, source_record_id="RXN_1"),
        admission="development_train",
        coordinate_map_evidence={"source": "explicit_source_map", "map_ids": [1, 2, 3]},
    )
    values.update(overrides)
    values["source_record_hash"] = reaction_qm_record_hash(
        record_id=values["record_id"],
        parent_reaction_id=values["parent_reaction_id"],
        reaction_family_id=values["reaction_family_id"],
        independent_reactant_system_id=values["independent_reactant_system_id"],
        repeated_ts_group=values["repeated_ts_group"],
        reactant_graph=values["reactant_graph"],
        product_graph=values["product_graph"],
        reactant_coordinates=values["reactant_coordinates"],
        ts_coordinates=values["ts_coordinates"],
        charge=values["charge"],
        multiplicity=values["multiplicity"],
        reference_protocol=values["reference_protocol"],
        coordinate_map_evidence=values["coordinate_map_evidence"],
    )
    return ReactionQMRecord(**values)


def test_valid_chnos_neutral_closed_shell_record_is_explicit():
    row = record()
    assert row.charge == -1
    assert row.multiplicity == 1
    assert row.event_label["evidence"] == "derived_under_contract"
    assert row.event_label["source_record_id"] == row.record_id


def test_missing_charge_and_multiplicity_are_rejected():
    with pytest.raises(ValueError, match="strict integers"):
        record(charge="unknown")
    with pytest.raises(ValueError, match="strict integers"):
        record(multiplicity="unknown")


def test_electronic_state_and_grouping_are_required_for_admission():
    with pytest.raises(ValueError, match="inconsistent with the electron count"):
        record(charge=0)
    with pytest.raises(ValueError, match="unresolved grouping"):
        record(parent_reaction_id="unknown")


def test_source_hash_binds_coordinates_and_electronic_state():
    row = record()
    with pytest.raises(ValueError, match="does not bind"):
        replace(row, reactant_coordinates=((0.1, 0.0, 0.0), *row.reactant_coordinates[1:]))


def test_duplicate_mapping_and_atom_count_mismatch_are_rejected():
    with pytest.raises(ValueError, match="duplicate atom mapping"):
        MappedGraph.from_dict({"atoms": {"1": "C", "01": "O"}, "bonds": []})
    with pytest.raises(ValueError, match="atom mappings must match"):
        record(product_graph=MappedGraph({1: "C", 2: "O"}, (Bond(1, 2, 2),)))


def test_ambiguous_bond_change_is_quarantined_by_derivation():
    reactant = MappedGraph({1: "C", 2: "C"}, (Bond(1, 2, 1, aromatic=True),))
    product = MappedGraph({1: "C", 2: "C"}, (Bond(1, 2, 2),))
    with pytest.raises(ValueError, match="ambiguous_aromatic_bond"):
        derive_event_label(reactant, product, source_record_id="RXN_ambiguous")


def test_mapped_smiles_preserves_explicit_hydrogen_maps():
    pytest.importorskip("rdkit")
    graph = MappedGraph.from_smiles("[C:1]([H:2])([H:3])[O:4][H:5]")
    assert graph.atom_maps == frozenset({1, 2, 3, 4, 5})


def test_product_ts_and_event_do_not_enter_reactant_input():
    row = record()
    visible = row.reactant_input()
    assert set(visible) == {"graph", "coordinates", "charge", "multiplicity"}
    assert "product" not in visible and "ts_coordinates" not in visible and "event_label" not in visible


def test_source_config_pins_observed_small_asset_hashes():
    files = {item.name: item for item in REACTION_QM_SOURCE.files}
    assert files["B3LYPD3_TZVP_reaction_info.csv"].expected_sha256 == "2facf37090a4cba872394ec6ab0c360b011d41acff9658268ff2eb61e6fd5ae1"
    assert files["B3LYPD3_TZVP.h5"].expected_sha256 == "3d0fc655819a9a2747f554a9025cd36cbdffd1175c4a1d40934b6fe5530af82a"


def test_event_label_must_match_endpoint_graphs():
    row = record()
    forged = dict(row.event_label)
    forged["bond_edits"] = [{"atom_i": 1, "atom_j": 3, "delta": 1.0}]
    with pytest.raises(ValueError, match="does not match"):
        record(event_label=forged)


def test_source_audit_omits_cache_absolute_paths(tmp_path):
    from xtbflow.data.reaction_qm import SourceFile, audit_source_files

    (tmp_path / "asset.dat").write_bytes(b"asset")
    config = REACTION_QM_SOURCE.__class__(
        dataset="test",
        revision="v1",
        landing_url="https://example.invalid/test",
        license_status="check",
        field_mapping={"asset": "asset.dat"},
        files=(SourceFile("asset.dat", "https://example.invalid/asset", "test"),),
        claim_limit="test only",
    )
    report = audit_source_files(config, tmp_path)
    assert report[0]["relative_path"] == "asset.dat"
    assert str(tmp_path) not in str(report)


def test_loader_reads_only_hash_verified_reaction_info(tmp_path):
    name = "B3LYPD3_TZVP_reaction_info.csv"
    payload = b"reaction_id,reaction_smiles\nRXN_1,[C:1]>>[C:1]\n"
    (tmp_path / name).write_bytes(payload)
    config = PublicSourceConfig(
        dataset="test",
        revision="v1",
        landing_url="https://example.invalid/test",
        license_status="check",
        field_mapping={"reaction_info": {"reaction_id": "reaction_id"}},
        files=(SourceFile(name, "https://example.invalid/asset", "reaction_info", expected_sha256=hashlib.sha256(payload).hexdigest()),),
        claim_limit="test only",
    )
    loader = ReactionQMLoader(cache_dir=tmp_path, config=config)
    assert list(loader.reaction_info())[0]["reaction_id"] == "RXN_1"
    (tmp_path / name).write_bytes(payload + b"tampered")
    with pytest.raises(ValueError, match="hash-verified"):
        next(loader.reaction_info())


def public_record(record_id: str, parent: str, group: str, symbols=("C", "O")):
    return PublicRecord(
        source_dataset="reaction-qm-dev",
        source_revision="v2",
        record_id=record_id,
        parent_reaction_id=parent,
        split_group=group,
        geometry_locator="cache#0",
        geometry_hash="hash-" + record_id,
        geometry_origin="source_reactant_endpoint",
        reference_protocol_id="B3LYP-D3-TZVP",
        units={"coordinates": "angstrom", "energy": "hartree", "forces": "hartree/angstrom"},
        available_label_mask={"event": True},
        charge_spin_evidence="explicit_source",
        license_record="pending_review",
        pretraining_overlap_audit="not_applicable",
        record_status="observed",
        admission="train",
        input_data={"symbols": list(symbols), "reactant_coordinates": [[0, 0, 0], [1, 0, 0]], "charge": 0, "multiplicity": 1},
        label_data={"event": {"evidence": "derived_under_contract"}},
        claim_limit="development",
    )


def test_parent_reaction_cannot_cross_split():
    rows = [public_record("a", "mother", "mother"), public_record("b", "mother", "mother")]
    with pytest.raises(SplitError):
        audit_no_group_leakage(rows, {"a": "train", "b": "test"})


def test_same_input_fingerprint_is_detectable_across_splits():
    first = public_record("a", "mother-a", "group-a")
    second = public_record("b", "mother-b", "group-b")
    assert first.input_fingerprint() == second.input_fingerprint()
    with pytest.raises(SplitError, match="input-fingerprint"):
        audit_no_input_fingerprint_leakage([first, second], {"a": "train", "b": "test"})
