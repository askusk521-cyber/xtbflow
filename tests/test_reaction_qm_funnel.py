"""Regression tests for the Reaction-QM task admission funnel.

Fixtures are tiny hand-written mapped reactions.  They exercise the *logic* of
the funnel (graph correspondence, symmetry handling, scope and identity gates);
they say nothing about the chemistry of the real source records.
"""
from __future__ import annotations

import pytest

pytest.importorskip("rdkit")

from rdkit import Chem  # noqa: E402

from xtbflow.data.reaction_qm_funnel import (  # noqa: E402
    MAP_AMBIGUOUS_CENTER,
    MAP_AMBIGUOUS_CENTER_H,
    MAP_NO_MATCH,
    MAP_REPRESENTATION_MISMATCH,
    MAP_SYMMETRIC_NONCENTER,
    MAP_UNIQUE,
    ReactionInput,
    SpeciesInput,
    _assign_components,
    analyse_reaction,
    assign_parent_groups,
    build_report,
    event_edits,
    map_endpoint,
    parse_mapped,
    record_row,
    split_side,
    bond_table,
)


def _z_by_map(smiles: str) -> tuple[int, ...]:
    mols = split_side(smiles)
    table = {a.GetAtomMapNum(): a.GetAtomicNum() for m in mols for a in m.GetAtoms()}
    return tuple(table[i] for i in range(1, len(table) + 1))


def _species(name: str, smiles: str, *, charge: int = 0, mult: int = 1, z=None, energies=(-1.0, -0.9, -0.95)):
    reactant_side = smiles.split(">>")[0]
    z = z if z is not None else _z_by_map(reactant_side)
    return SpeciesInput(
        name=name,
        smiles=smiles,
        atomic_numbers=tuple(z),
        charge=charge,
        multiplicity=mult,
        energies=tuple(energies),
        coordinate_shape=(len(z), 3),
        coordinates_finite=True,
    )


def _record(record_id: str, ts: str, reactants: list[str], products: list[str], **overrides) -> ReactionInput:
    species = {"TS": overrides.get("ts_species") or _species("TS", ts)}
    for i, s in enumerate(reactants):
        species[f"R{i}"] = _species(f"R{i}", s)
    for i, s in enumerate(products):
        species[f"P{i}"] = _species(f"P{i}", s)
    return ReactionInput(record_id, species)


# HCN -> HNC.  Endpoint numbering is deliberately NOT the global order so that a
# monotone "k-th smallest" shortcut would be wrong, and every atom is distinct
# so the graph correspondence is unique.
HCN_TS = "[H:1][C:2]#[N:3]>>[N:3]#[C:2].[H:1]"  # placeholder, replaced below
HCN_TS = "[H:1][C:2]#[N:3]>>[C:2]#[N:3][H:1]"
HCN_R0 = "[H:1][C:2]#[N:3]"
HCN_P0 = "[C:1]#[N:2][H:3]"


def test_unique_correspondence_with_non_monotone_numbering():
    rec = _record("RXN_hcn", HCN_TS, [HCN_R0], [HCN_P0])
    out = analyse_reaction(rec)
    assert out.readable and out.in_scope and out.identity_verified
    assert out.mapping["P0"] == MAP_UNIQUE
    assert out.tasks == {
        "event_only": True,
        "geometry_only": True,
        "energy_only": True,
        "energy_force": False,
        "paired_joint": True,
    }
    # H1 leaves C2 and joins N3.
    assert sorted(out.edits) == [(1, 2, -1), (1, 3, 1)]
    assert "forces_absent_in_source" in out.reasons


def test_map_endpoint_reports_non_monotone_numbering():
    species = parse_mapped(HCN_P0)
    component = split_side("[C:2]#[N:3][H:1]")[0]
    result = map_endpoint(species, component, frozenset({1, 2, 3}))
    assert result.status == MAP_UNIQUE
    assert result.monotone_consistent is False


def test_symmetric_species_with_centre_on_singleton_classes_is_certified():
    # Methanol O-H adds across H-C#N; the methyl hydrogens are interchangeable
    # but never touch the reaction centre.
    ts = (
        "[C:1]([O:2][H:3])([H:4])([H:5])[H:6].[H:7][C:8]#[N:9]"
        ">>[C:1]([O:2][C:8]([H:7])=[N:9][H:3])([H:4])([H:5])[H:6]"
    )
    rec = _record(
        "RXN_meoh",
        ts,
        ["[C:1]([O:2][H:3])([H:4])([H:5])[H:6]", "[H:1][C:2]#[N:3]"],
        ["[C:1]([O:2][C:8]([H:7])=[N:9][H:3])([H:4])([H:5])[H:6]"],
    )
    out = analyse_reaction(rec)
    assert out.mapping["R0"] == MAP_SYMMETRIC_NONCENTER
    assert out.mapping["R1"] == MAP_UNIQUE
    assert out.tasks["paired_joint"] is True


def test_symmetric_species_with_centre_in_equivalence_class_is_ambiguous():
    # Ethylene's two carbons are interchangeable and both sit in the centre, so
    # coordinate rows cannot be tied to the global ids from the graphs alone.
    ts = (
        "[C:1](=[C:2]([H:5])[H:6])([H:3])[H:4].[N:7]([H:8])([H:9])[H:10]"
        ">>[C:1]([C:2]([H:5])([H:6])[N:7]([H:9])[H:10])([H:3])([H:4])[H:8]"
    )
    rec = _record(
        "RXN_ethylene",
        ts,
        ["[C:1](=[C:2]([H:5])[H:6])([H:3])[H:4]", "[N:1]([H:2])([H:3])[H:4]"],
        ["[C:1]([C:2]([H:5])([H:6])[N:7]([H:9])[H:10])([H:3])([H:4])[H:8]"],
    )
    out = analyse_reaction(rec)
    assert out.identity_verified
    assert out.mapping["R0"] == MAP_AMBIGUOUS_CENTER
    assert out.mapping["R1"] == MAP_AMBIGUOUS_CENTER_H  # the transferred NH3 hydrogen
    assert out.tasks["event_only"] is True  # the event label never needs endpoint mapping
    assert out.tasks["paired_joint"] is False
    assert "endpoint_mapping_not_certified" in out.reasons


def test_species_that_is_not_a_component_has_no_match():
    species = parse_mapped("[C:1]([H:2])([H:3])([H:4])[H:5]")
    component = split_side("[N:1]([H:2])([H:3])[H:4]")[0]
    assert map_endpoint(species, component, frozenset()).status == MAP_NO_MATCH


def test_extra_bond_in_target_is_not_accepted_as_isomorphism():
    # A path of three atoms embeds in a triangle, so a bare substructure match
    # succeeds; the target carries one more bond, so the round-trip check must
    # reject it under both the full-graph and the skeleton test.
    species = parse_mapped("[C:1]([C:2])[C:3]")
    component = split_side("[C:1]1[C:2][C:3]1")[0]
    assert map_endpoint(species, component, frozenset()).status == MAP_NO_MATCH


def test_bond_order_disagreement_is_a_representation_mismatch_not_a_missing_molecule():
    species = parse_mapped("[C:1]([C:2]([H:5])[H:6])([H:3])[H:4]")
    component = split_side("[C:1](=[C:2]([H:5])[H:6])([H:3])[H:4]")[0]
    assert map_endpoint(species, component, frozenset()).status == MAP_REPRESENTATION_MISMATCH


def test_endpoint_that_disagrees_with_the_ts_side_withholds_event_and_pairing():
    # The TS string writes H-C#N; the endpoint species writes H-C=N.  Same
    # skeleton, different bond orders: the event label would be representation
    # dependent, so event_only must be withheld while geometry stays usable.
    out = analyse_reaction(_record("RXN_repr", HCN_TS, ["[H:1][C:2]=[N:3]"], [HCN_P0]))
    assert out.identity_verified
    assert out.mapping["R0"] == MAP_REPRESENTATION_MISMATCH
    assert "endpoint_representation_mismatch" in out.reasons
    assert out.tasks["event_only"] is False and out.tasks["paired_joint"] is False
    assert out.tasks["geometry_only"] is True


def test_endpoint_with_different_connectivity_is_flagged():
    out = analyse_reaction(_record("RXN_conn", HCN_TS, ["[C:1]([H:2])=[N:3]"], [HCN_P0]))
    # Local numbering differs too, but what matters is that H-C-N is not H-N-C.
    assert "endpoint_connectivity_mismatch" in out.reasons or "endpoint_representation_mismatch" in out.reasons
    assert out.tasks["event_only"] is False


def test_hydrogen_only_centre_ambiguity_is_reported_separately():
    # Ammonia hydrogens are equivalent; one of them is in the reaction centre.
    species = parse_mapped("[N:1]([H:2])([H:3])[H:4]")
    component = split_side("[N:1]([H:2])([H:3])[H:4]")[0]
    assert map_endpoint(species, component, frozenset({2, 1})).status == MAP_AMBIGUOUS_CENTER_H
    assert map_endpoint(species, component, frozenset()).status == MAP_SYMMETRIC_NONCENTER

def test_identical_reactants_use_the_declared_order_rule_and_are_flagged():
    assigned = _assign_components(
        [("R0", "O"), ("R1", "O")], component_keys=["O", "O"], component_min_id=[7, 1]
    )
    # R0 follows the component holding the smallest global map id.
    assert assigned == {"R0": (1, "rule_min_id_order"), "R1": (0, "rule_min_id_order")}


def test_inventory_mismatch_in_assignment_returns_none():
    assert _assign_components([("R0", "O")], component_keys=["N"], component_min_id=[1]) is None


def test_aromatic_spectator_blocks_event_label_but_not_geometry():
    benzene = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    for i, atom in enumerate(benzene.GetAtoms(), start=1):
        atom.SetAtomMapNum(i)
    benzene_smiles = Chem.MolToSmiles(benzene)
    ts = f"{benzene_smiles}.[H:13][C:14]#[N:15]>>{benzene_smiles}.[C:14]#[N:15][H:13]"
    rec = _record(
        "RXN_aromatic",
        ts,
        [benzene_smiles, "[H:1][C:2]#[N:3]"],
        [benzene_smiles, "[C:1]#[N:2][H:3]"],
    )
    out = analyse_reaction(rec)
    assert out.identity_verified
    assert out.aromatic and out.aromatic_integral_edits
    assert out.tasks["event_only"] is False
    assert out.tasks["paired_joint"] is False
    assert out.tasks["geometry_only"] is True
    assert "aromatic_representation_pending" in out.reasons


def test_out_of_scope_element_is_excluded_before_identity_checks():
    ts = "[Cl:1][H:2]>>[H:2][Cl:1]"
    rec = _record("RXN_cl", ts, ["[Cl:1][H:2]"], ["[H:1][Cl:2]"])
    out = analyse_reaction(rec)
    assert out.readable and not out.in_scope
    assert out.first_block == "out_of_scope_element"
    assert not any(out.tasks.values())


def test_open_shell_and_parity_inconsistency_are_quarantined():
    rec = _record("RXN_hcn", HCN_TS, [HCN_R0], [HCN_P0])
    bad = dict(rec.species)
    bad["R0"] = _species("R0", HCN_R0, mult=2)
    out = analyse_reaction(ReactionInput("RXN_open_shell", bad))
    assert "non_singlet_state" in out.reasons and "electronic_state_inconsistent" in out.reasons
    assert not out.in_scope


def test_charge_must_be_conserved():
    rec = _record("RXN_hcn", HCN_TS, [HCN_R0], [HCN_P0])
    bad = dict(rec.species)
    bad["P0"] = _species("P0", HCN_P0, charge=1)
    out = analyse_reaction(ReactionInput("RXN_charge", bad))
    assert "charge_not_conserved" in out.reasons


def test_ts_coordinate_order_mismatch_is_not_identity_verified():
    rec = _record("RXN_hcn", HCN_TS, [HCN_R0], [HCN_P0])
    bad = dict(rec.species)
    bad["TS"] = _species("TS", HCN_TS, z=(6, 1, 7))  # rows not in global-map order
    out = analyse_reaction(ReactionInput("RXN_order", bad))
    assert out.in_scope and not out.identity_verified
    assert out.first_block == "ts_coordinate_order_unverified"
    assert not any(out.tasks.values())


def test_endpoint_row_order_must_follow_its_own_map_order():
    rec = _record("RXN_hcn", HCN_TS, [HCN_R0], [HCN_P0])
    bad = dict(rec.species)
    bad["P0"] = _species("P0", HCN_P0, z=(1, 6, 7))  # global order, not P0's local order
    out = analyse_reaction(ReactionInput("RXN_endpoint_order", bad))
    assert "species_coordinate_order_unverified" in out.reasons
    assert not out.identity_verified


def test_duplicate_map_ids_make_the_record_unreadable():
    ts = "[H:1][C:1]#[N:3]>>[C:1]#[N:3][H:1]"
    out = analyse_reaction(
        _record("RXN_dup", ts, [HCN_R0], [HCN_P0], ts_species=_species("TS", ts, z=(1, 6, 7)))
    )
    assert not out.readable
    assert out.first_block == "map_ids_invalid"


def test_identical_graphs_have_no_event():
    ts = "[H:1][C:2]#[N:3]>>[H:1][C:2]#[N:3]"
    out = analyse_reaction(_record("RXN_noop", ts, [HCN_R0], ["[H:1][C:2]#[N:3]"]))
    assert out.identity_verified
    assert out.tasks["event_only"] is False
    assert "no_bond_change" in out.reasons
    assert out.tasks["geometry_only"] is True


def test_non_finite_energy_only_disables_the_energy_task():
    rec = _record("RXN_hcn", HCN_TS, [HCN_R0], [HCN_P0])
    bad = dict(rec.species)
    bad["TS"] = _species("TS", HCN_TS, energies=(float("nan"), 0.0, 0.0))
    out = analyse_reaction(ReactionInput("RXN_nan", bad))
    assert out.tasks["energy_only"] is False
    assert out.tasks["event_only"] is True and out.tasks["geometry_only"] is True


def test_event_edits_sign_and_integrality():
    reactant = bond_table(split_side("[H:1][C:2]#[N:3]"))
    product = bond_table(split_side("[C:2]#[N:3][H:1]"))
    edits, integral = event_edits(reactant, product)
    assert integral and edits == ((1, 2, -1), (1, 3, 1))
    aromatic = bond_table(split_side("[c:1]1[c:2][c:3][c:4][c:5][c:6]1"))
    single = bond_table(split_side("[C:1]1[C:2][C:3][C:4][C:5][C:6]1"))
    _, integral = event_edits(aromatic, single)
    assert integral is False


def test_group_ids_link_forward_and_reverse_and_ignore_record_order():
    rows = [
        {"record_id": "a", "reactant_key": "A.B", "product_key": "C"},
        {"record_id": "b", "reactant_key": "C", "product_key": "A.B"},  # reverse of a
        {"record_id": "c", "reactant_key": "X", "product_key": "Y"},
    ]
    groups = assign_parent_groups(rows)
    assert groups["a"] == groups["b"] != groups["c"]
    assert assign_parent_groups(list(reversed(rows))) == groups


def test_report_counts_records_and_parent_groups_per_stage():
    ok = analyse_reaction(_record("RXN_hcn", HCN_TS, [HCN_R0], [HCN_P0]))
    cl = analyse_reaction(_record("RXN_cl", "[Cl:1][H:2]>>[H:2][Cl:1]", ["[Cl:1][H:2]"], ["[H:1][Cl:2]"]))
    rows = [record_row(ok), record_row(cl)]
    groups = assign_parent_groups(rows)
    report = build_report(rows, groups, source_records=2)
    stages = {s["stage"]: s for s in report["funnel"]}
    assert stages["source_records"]["records"] == 2
    assert stages["readable"]["records"] == 2
    assert stages["in_scope"]["records"] == 1
    assert stages["identity_verifiable"]["records"] == 1
    assert report["tasks"]["paired_joint"]["records"] == 1
    assert report["tasks"]["energy_force"]["records"] == 0
    assert report["tasks"]["energy_force"]["unavailable_reason"] == "forces_absent_in_source"
    assert report["exclusion_reasons"]["out_of_scope_element"]["first_blocking_records"] == 1
    # Every exclusion reason must carry a next action.
    assert all(v["next_action"] != "unclassified; inspect" for v in report["exclusion_reasons"].values())
    assert "no record is promoted" in report["claim_limits"][0]


def test_ts_event_distance_is_label_free_evidence_not_a_gate():
    import numpy as np

    rec = _record("RXN_hcn", HCN_TS, [HCN_R0], [HCN_P0])
    # H1 moves C2 -> N3: edited pairs are (1,2) and (1,3); ids are 1-based rows.
    coords = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.4, 0.0, 0.0]])
    ts = SpeciesInput(**{**rec.species["TS"].__dict__, "coordinates": coords})
    out = analyse_reaction(ReactionInput("RXN_hcn_xyz", {**rec.species, "TS": ts}))
    assert out.ts_max_edit_distance == pytest.approx(2.4)
    assert out.tasks["paired_joint"] is True  # distance never changes admission
    far = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [50.0, 0.0, 0.0]])
    ts_far = SpeciesInput(**{**rec.species["TS"].__dict__, "coordinates": far})
    out_far = analyse_reaction(ReactionInput("RXN_hcn_far", {**rec.species, "TS": ts_far}))
    assert out_far.tasks["paired_joint"] is True
    assert out_far.ts_max_edit_distance == pytest.approx(50.0)


def test_report_quantifies_what_resolving_each_ambiguity_would_recover():
    ethylene_ts = (
        "[C:1](=[C:2]([H:5])[H:6])([H:3])[H:4].[N:7]([H:8])([H:9])[H:10]"
        ">>[C:1]([C:2]([H:5])([H:6])[N:7]([H:9])[H:10])([H:3])([H:4])[H:8]"
    )
    amb = analyse_reaction(
        _record(
            "RXN_amb",
            ethylene_ts,
            ["[C:1](=[C:2]([H:5])[H:6])([H:3])[H:4]", "[N:1]([H:2])([H:3])[H:4]"],
            ["[C:1]([C:2]([H:5])([H:6])[N:7]([H:9])[H:10])([H:3])([H:4])[H:8]"],
        )
    )
    ok = analyse_reaction(_record("RXN_hcn", HCN_TS, [HCN_R0], [HCN_P0]))
    rows = [record_row(amb), record_row(ok)]
    report = build_report(rows, assign_parent_groups(rows), source_records=2)
    recover = report["paired_joint_recoverable"]
    assert recover["currently_certified"]["records"] == 1
    # The ethylene case has a heavy-atom (C1/C2) ambiguity, so only the "all" tier recovers it.
    assert recover["if_hydrogen_centre_ambiguity_resolved"]["records"] == 1
    assert recover["if_all_centre_ambiguity_resolved"]["records"] == 2