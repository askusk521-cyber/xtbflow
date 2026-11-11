"""Read-only handoff from immutable live snapshots to the existing #137 service.

Only generated coordinates and audited reactant identity cross this boundary.
Proxy verdicts, energies, reference coordinates and endpoint labels are never
candidate inputs. Four-dimensional guidance scores are not chemical bond edits.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from xtbflow.data.records import canonical_hash
from xtbflow.runtime import RunLedger
from xtbflow.validation.certification import CertificationCandidate, file_hash


def build_reactant_context(
    mapped_reactant_smiles: str,
    source_record: dict[str, Any],
    xyz_path: Path,
    input_payload: dict[str, Any],
) -> dict[str, Any]:
    """Reuse the source adapter's mapped graph/XYZ audit, without target labels."""
    import numpy as np
    from rdkit import Chem
    from xtbflow.data import dft_da as source_adapter
    from xtbflow.data.dft_da import _mapped_mol, _bond_matrix, _graph_matches, _read_xyz

    if ">" in mapped_reactant_smiles:
        raise ValueError("supply reactant SMILES only")
    if source_record.get("admission") != "development_train":
        raise ValueError("this fixed development handoff cannot grant new data admission")
    reactant = source_record["reactant"]
    atoms = source_record["atoms"]
    if canonical_hash({"mapped_smiles": mapped_reactant_smiles}) != reactant["graph_sha256"]:
        raise ValueError("mapped reactant graph hash mismatch")
    if file_hash(xyz_path) != reactant["coordinates_sha256"]:
        raise ValueError("reactant XYZ hash mismatch")
    symbols, coordinates, _ = _read_xyz(xyz_path)
    if list(symbols) != input_payload["symbols"] or coordinates.tolist() != input_payload["coordinates"]:
        raise ValueError("live input is not the audited reactant XYZ")
    if (input_payload["charge"], input_payload["multiplicity"]) != (reactant["charge"], reactant["multiplicity"]):
        raise ValueError("live electronic state differs from source evidence")
    if input_payload["environment"] != {"periodic": False, "solvent": "none"}:
        raise ValueError("live input must explicitly declare nonperiodic gas-phase modeling")
    if not reactant.get("charge_spin_evidence") or not source_record.get("license_record"):
        raise ValueError("electronic state and license evidence are required")
    molecule = _mapped_mol(mapped_reactant_smiles)
    map_ids = tuple(atom.GetAtomMapNum() for atom in molecule.GetAtoms())
    numbers = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
    if list(map_ids) != atoms["map_ids"] or numbers != atoms["atomic_numbers"]:
        raise ValueError("mapped source row order differs from the audited atom inventory")
    if not _graph_matches(molecule, map_ids, symbols, coordinates):
        raise ValueError("coordinate connectivity does not match the mapped reactant graph")
    Chem.Kekulize(molecule, clearAromaticFlags=True)
    bonds = _bond_matrix(molecule, map_ids)
    if not np.equal(bonds, bonds.astype(int)).all():
        raise ValueError("reactant bond orders are ambiguous")
    bonds = bonds.astype(int)
    valence = {1: 1, 6: 4, 7: 5, 8: 6}
    for index, atom in enumerate(molecule.GetAtoms()):
        if atom.GetNumRadicalElectrons():
            raise ValueError("open-shell reactant is outside the reference scope")
        bonds[index, index] = valence[atom.GetAtomicNum()] - atom.GetFormalCharge() - int(bonds[index].sum())
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    permutations = sorted(molecule.GetSubstructMatches(molecule, uniquify=False, useChirality=True, maxMatches=2049))
    if not permutations or len(permutations) > 2048:
        raise ValueError("reactant automorphism enumeration exceeds the bounded contract")
    return {
        "schema": "xtbflow-live-reactant-context/v1",
        "input_hash": canonical_hash(input_payload),
        "atomic_numbers": numbers,
        "atom_ids": [f"source-map:{value}" for value in map_ids],
        "reactant_be": bonds.tolist(),
        "reactant_automorphisms": [list(row) for row in permutations],
        "charge": input_payload["charge"], "multiplicity": input_payload["multiplicity"],
        "environment": {"kind": "gas", "periodic": False},
        "provenance": {
            "input_sha256": canonical_hash(input_payload),
            "atom_mapping_evidence": "source mapped reactant order and coordinate-connectivity audit; generated snapshots preserve row indices",
            "electronic_state_evidence": reactant["charge_spin_evidence"],
            "reactant_xyz_sha256": reactant["coordinates_sha256"],
            "reactant_graph_sha256": reactant["graph_sha256"],
            "source_asset_sha256": source_record["source_asset_sha256"],
            "context_builder_sha256": file_hash(Path(__file__)),
            "source_adapter_sha256": file_hash(Path(source_adapter.__file__)),
            "graph_environment": {"rdkit": Chem.rdBase.rdkitVersion, "numpy": np.__version__},
            "license_record": source_record["license_record"],
            "input_origin": reactant["input_origin"],
            "scope": "development fixture using a declared source endpoint; not independent reactant preparation or held-out confirmation",
        },
    }


def _checked_snapshot(snapshot: dict[str, Any], parent: str, seed: int) -> dict[str, Any]:
    payload = {key: value for key, value in snapshot.items() if key != "snapshot_id"}
    if canonical_hash(payload) != snapshot["snapshot_id"]:
        raise ValueError("immutable snapshot hash mismatch")
    if snapshot["parent_id"] != parent or snapshot["random_seed"] != seed:
        raise ValueError("snapshot parent/seed mismatch")
    return snapshot


def load_live_feedback(
    folder: Path,
    generation_config: Path,
    context_path: Path,
    expected: dict[str, Any],
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], CertificationCandidate]]]:
    """Whitelist two final snapshots; never read a proxy-refined TS as a seed."""
    folder = folder.resolve()
    manifest_path = folder / "manifest.json"
    if file_hash(manifest_path) != expected["manifest_sha256"]:
        raise ValueError("live manifest hash mismatch")
    if file_hash(context_path) != expected["context_sha256"]:
        raise ValueError("reactant context hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    context = json.loads(context_path.read_text())
    if manifest["schema"] != "xtbflow-real-feedback-loop/v1" or context["schema"] != "xtbflow-live-reactant-context/v1":
        raise ValueError("unsupported live/context schema")
    if manifest["source"] != {"commit": expected["source_commit"], "dirty": False}:
        raise ValueError("clean exact generation source required")
    if file_hash(generation_config) != expected["config_sha256"] or manifest["config_sha256"] != expected["config_sha256"]:
        raise ValueError("live configuration hash mismatch")
    if canonical_hash(manifest["input"]) != manifest["input_hash"] or context["input_hash"] != manifest["input_hash"]:
        raise ValueError("reactant input identity mismatch")
    if manifest["parent_id"] != expected["parent_id"] or manifest["seed"] != expected["seed"]:
        raise ValueError("frozen parent/seed mismatch")
    if (context["charge"], context["multiplicity"]) != (manifest["input"]["charge"], manifest["input"]["multiplicity"]):
        raise ValueError("context electronic state mismatch")
    symbols = {1: "H", 6: "C", 7: "N", 8: "O"}
    if [symbols.get(z) for z in context["atomic_numbers"]] != manifest["input"]["symbols"]:
        raise ValueError("context atom inventory differs from live input")
    if context["environment"] != {"kind": "gas", "periodic": False} or manifest["input"]["environment"] != {"periodic": False, "solvent": "none"}:
        raise ValueError("explicit gas-phase model declaration required")
    if [arm["arm"] for arm in manifest["arms"]] != ["no_feedback", "physical_feedback"]:
        raise ValueError("exactly the two frozen arms are required, in generation order")
    candidates = []
    upstream = []
    initial_id = None
    for arm in manifest["arms"]:
        before = _checked_snapshot(arm["snapshot_before"], manifest["parent_id"], manifest["seed"])
        after = _checked_snapshot(arm["snapshot_after"], manifest["parent_id"], manifest["seed"])
        if initial_id is not None and before["snapshot_id"] != initial_id:
            raise ValueError("matched arms did not start from the same candidate")
        initial_id = before["snapshot_id"]
        if after["metadata"] != before["metadata"] or before["metadata"]["input_hash"] != context["input_hash"]:
            raise ValueError("snapshot changed the reactant context")
        ledger_path = folder / f"ledger_{arm['arm']}.json"
        ledger = RunLedger.load(ledger_path)
        if ledger.reservations or ledger.summary()["totals"] != arm["runtime_ledger"]["totals"]:
            raise ValueError("unsettled or mismatched upstream calculator accounting")
        provenance = dict(context["provenance"])
        provenance.update({
            "live_manifest_sha256": expected["manifest_sha256"],
            "live_source_commit": expected["source_commit"],
            "reactant_context_sha256": expected["context_sha256"],
            "snapshot_id": after["snapshot_id"],
            "event_representation": "guidance scores only; no mapped chemical proposal",
        })
        candidate = CertificationCandidate(
            candidate_id=f"{manifest['parent_id']}:{arm['arm']}", parent_id=manifest["parent_id"],
            atomic_numbers=context["atomic_numbers"], coordinates_angstrom=after["geometry"],
            atom_ids=context["atom_ids"], reactant_be=context["reactant_be"],
            planned_product_be=None, reactant_automorphisms=context["reactant_automorphisms"],
            charge=context["charge"], multiplicity=context["multiplicity"],
            environment=context["environment"], provenance=provenance,
        )
        candidates.append(({"method": arm["arm"]}, candidate))
        upstream.append({
            "arm": arm["arm"], "snapshot_id": after["snapshot_id"],
            "feedback_ledger_sha256": file_hash(ledger_path), "feedback_totals": ledger.summary()["totals"],
            "prior_diagnostic_cost": arm["certification"].get("cost"),
            "prior_diagnostic_calls": (arm["certification"].get("backend_result") or {}).get("calculator_calls"),
            "geometry_changed": arm["geometry_changed"], "event_changed": arm["event_changed"],
        })
    return {
        "kind": "live_feedback_handoff", "source_commit": expected["source_commit"],
        "source_dirty": False, "report_sha256": expected["manifest_sha256"],
        "config_sha256": expected["config_sha256"], "context_sha256": expected["context_sha256"],
        "parent_id": manifest["parent_id"], "seed": manifest["seed"],
        "parent_denominator": 1, "candidate_denominator": 2, "arms": upstream,
    }, candidates
