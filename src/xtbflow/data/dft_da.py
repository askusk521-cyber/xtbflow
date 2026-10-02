"""Audited adapter for the public Diels--Alder reaction-space archive.

The Figshare archive stores mapped reaction SMILES beside unlabeled XYZ files.
This module admits only rows for which the archive's atom order can be checked
against the mapped reactant/product graphs and an explicit electronic-state
line is present in the distributed TS log.  Element order alone is never used
as atom-map evidence.

Raw archive bytes remain on the execution host.  The returned Track-B rows
contain logical archive locators and cryptographic hashes only.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from .records import canonical_hash
from .track_b import TrackBRecord, filter_track_b_records


SOURCE_URL = "https://figshare.com/articles/dataset/Diels-Alder_reaction_space_for_self-healing_polymer/29118509"
SOURCE_REVISION = "figshare:29118509:v5"
SOURCE_LICENSE = "CC-BY-4.0"
SOURCE_ARCHIVE_NAME = "DATASET_DA_F.tar.gz"
SOURCE_PROTOCOL_ID = "dft-da-m06-2x-def2-tzvp-ts-tools-v5"
SOURCE_PROTOCOL = {
    "protocol_id": SOURCE_PROTOCOL_ID,
    "protocol_locator": SOURCE_URL,
    "method": "M06-2X/def2-TZVP",
    "software": "Gaussian16 with TS-tools; GFN2-xTB external state audit",
    "version": "Figshare article 29118509 version 5",
    "environment": "published Diels-Alder reaction-space archive",
}
SOURCE_PROTOCOL_SHA256 = canonical_hash(SOURCE_PROTOCOL)
FAMILY_IDENTITY_RULE = "canonical_reactant_components_without_atom_maps_v1"
STATE_RE = re.compile(r"Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)")
XTB_STATE_RE = re.compile(r"xtb\s+\S+\s+--chrg\s+(-?\d+)\s+--uhf\s+(\d+)")
SUPPORTED_ATOMIC_NUMBERS = frozenset({1, 6, 7, 8})
VALENCE = {1: 1.0, 6: 4.0, 7: 5.0, 8: 6.0}


@dataclass(frozen=True)
class DftDaSample:
    """One admitted source row plus arrays needed by the training runner."""

    record: TrackBRecord
    reactant_smiles: str
    product_smiles: str
    reactant_coordinates: np.ndarray
    product_coordinates: np.ndarray
    ts_coordinates: np.ndarray
    reactant_bonds: np.ndarray
    product_bonds: np.ndarray
    atomic_numbers: tuple[int, ...]
    map_ids: tuple[int, ...]


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_asset_hash(cache_root: Path) -> str:
    archive = cache_root / SOURCE_ARCHIVE_NAME
    if not archive.is_file():
        raise FileNotFoundError(f"missing public archive: {archive}")
    return hash_file(archive)


def _mapped_mol(smiles: str) -> Chem.Mol:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    molecule = Chem.MolFromSmiles(smiles, params)
    if molecule is None:
        raise ValueError("mapped SMILES did not parse")
    maps = [atom.GetAtomMapNum() for atom in molecule.GetAtoms()]
    if not maps or any(type(value) is not int or value < 1 for value in maps):
        raise ValueError("mapped SMILES has unresolved atom maps")
    if len(set(maps)) != len(maps):
        raise ValueError("mapped SMILES has duplicate atom maps")
    return molecule


def _bond_matrix(molecule: Chem.Mol, map_order: tuple[int, ...]) -> np.ndarray:
    index = {atom.GetAtomMapNum(): idx for idx, atom in enumerate(molecule.GetAtoms())}
    matrix = np.zeros((len(map_order), len(map_order)), dtype=np.float64)
    for bond in molecule.GetBonds():
        left = map_order.index(bond.GetBeginAtom().GetAtomMapNum())
        right = map_order.index(bond.GetEndAtom().GetAtomMapNum())
        order = float(bond.GetBondTypeAsDouble())
        matrix[left, right] = order
        matrix[right, left] = order
    if set(index) != set(map_order):
        raise ValueError("mapped atom set changed while constructing bond matrix")
    return matrix


def _canonical_component_smiles(smiles: str) -> str:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    components: list[str] = []
    for component in smiles.split("."):
        molecule = Chem.MolFromSmiles(component, params)
        if molecule is None:
            raise ValueError("component SMILES did not parse")
        # Family identity must not depend on arbitrary source atom-map labels.
        # Keep the original mapped SMILES for calculation identity and event
        # labels; this normalized key is only for chemical grouping.
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        components.append(Chem.MolToSmiles(molecule, canonical=True))
    return ".".join(sorted(components))


def _read_xyz(path: Path) -> tuple[tuple[str, ...], np.ndarray, bytes]:
    raw = path.read_bytes()
    lines = raw.decode("utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"malformed XYZ: {path.name}")
    count = int(lines[0].strip())
    rows = lines[2 : 2 + count]
    if len(rows) != count:
        raise ValueError(f"XYZ atom count mismatch: {path.name}")
    symbols: list[str] = []
    coordinates: list[list[float]] = []
    for row in rows:
        fields = row.split()
        if len(fields) < 4:
            raise ValueError(f"malformed XYZ coordinate row: {path.name}")
        symbols.append(fields[0])
        coordinates.append([float(fields[1]), float(fields[2]), float(fields[3])])
    array = np.asarray(coordinates, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"non-finite XYZ coordinates: {path.name}")
    return tuple(symbols), array, raw


def _coordinate_graph(symbols: tuple[str, ...], coordinates: np.ndarray) -> np.ndarray:
    molecule = Chem.RWMol()
    for symbol in symbols:
        molecule.AddAtom(Chem.Atom(symbol))
    block = f"{len(symbols)}\n\n" + "\n".join(
        f"{symbol} {x:.12f} {y:.12f} {z:.12f}"
        for symbol, (x, y, z) in zip(symbols, coordinates.tolist())
    ) + "\n"
    geometry = Chem.MolFromXYZBlock(block)
    if geometry is None:
        raise ValueError("RDKit could not parse geometry for connectivity audit")
    rdDetermineBonds.DetermineConnectivity(geometry, useVdw=True, covFactor=1.25)
    return Chem.GetAdjacencyMatrix(geometry)


def _graph_matches(molecule: Chem.Mol, map_order: tuple[int, ...], symbols: tuple[str, ...], coordinates: np.ndarray) -> bool:
    atoms_by_map = {atom.GetAtomMapNum(): atom for atom in molecule.GetAtoms()}
    expected_symbols = tuple(atoms_by_map[map_id].GetSymbol() for map_id in map_order)
    if expected_symbols != symbols:
        return False
    expected = Chem.GetAdjacencyMatrix(molecule)
    expected_indices = [atoms_by_map[map_id].GetIdx() for map_id in map_order]
    expected = expected[np.ix_(expected_indices, expected_indices)]
    actual = _coordinate_graph(symbols, coordinates)
    return bool(np.array_equal(expected, actual))


def _event_edits(reactant: np.ndarray, product: np.ndarray, map_ids: tuple[int, ...]) -> tuple[tuple[int, int, int], ...]:
    edits: list[tuple[int, int, int]] = []
    for left in range(len(map_ids)):
        for right in range(left + 1, len(map_ids)):
            delta = float(product[left, right] - reactant[left, right])
            if abs(delta) < 1e-9:
                continue
            rounded = int(round(delta))
            if abs(delta - rounded) > 1e-9 or rounded == 0:
                raise ValueError("non-integral bond edit in mapped source")
            edits.append((map_ids[left], map_ids[right], rounded))
    return tuple(edits)


def _state_evidence(log_paths: Iterable[Path]) -> str:
    paths = tuple(log_paths)
    if len(paths) != 2:
        raise ValueError("both monomer state logs are required")
    evidence = []
    for log_path in paths:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        states = set(STATE_RE.findall(text))
        xtb_states = set(XTB_STATE_RE.findall(text))
        if states != {("0", "1")} or xtb_states != {("0", "0")}:
            raise ValueError("monomer log does not provide a unique neutral-singlet state")
        if "Normal termination of Gaussian" not in text or "Error termination" in text:
            raise ValueError("monomer log did not terminate normally")
        evidence.append(f"{log_path.name}: Charge=0 Multiplicity=1; xTB --chrg 0 --uhf 0")
    return "; ".join(evidence)


def _row_sample(cache_root: Path, row: Mapping[str, str], source_asset_sha256: str) -> DftDaSample:
    reaction_smiles = row.get("smiles") or row.get("reaction")
    if not reaction_smiles:
        raise ValueError("source row has no mapped reaction SMILES")
    try:
        reactant_smiles, product_smiles = reaction_smiles.split(">>")
    except ValueError as exc:
        raise ValueError("reaction SMILES must contain one >>") from exc
    reactant = _mapped_mol(reactant_smiles)
    product = _mapped_mol(product_smiles)
    map_ids = tuple(atom.GetAtomMapNum() for atom in reactant.GetAtoms())
    product_map_ids = tuple(atom.GetAtomMapNum() for atom in product.GetAtoms())
    if set(map_ids) != set(product_map_ids):
        raise ValueError("reactant and product mapped atom sets differ")
    if any(atom.GetAtomicNum() not in SUPPORTED_ATOMIC_NUMBERS for atom in reactant.GetAtoms()):
        raise ValueError("source contains an element outside the frozen CHNO scope")

    reaction_dir = cache_root / (row.get("R_dir") or f"reaction_{row['R']}")
    rid = row["R"]
    reactant_symbols, reactant_coordinates, reactant_bytes = _read_xyz(reaction_dir / f"{rid}_reactant.xyz")
    product_symbols, product_coordinates, product_bytes = _read_xyz(reaction_dir / f"{rid}_product.xyz")
    ts_symbols, ts_coordinates, ts_bytes = _read_xyz(reaction_dir / f"{rid}_ts.xyz")
    expected_reactant_symbols = tuple(atom.GetSymbol() for atom in reactant.GetAtoms())
    product_by_map = {atom.GetAtomMapNum(): atom for atom in product.GetAtoms()}
    expected_product_symbols = tuple(product_by_map[map_id].GetSymbol() for map_id in map_ids)
    if reactant_symbols != expected_reactant_symbols or product_symbols != expected_product_symbols or ts_symbols != expected_reactant_symbols:
        raise ValueError("XYZ atom order is not the mapped source order")
    if not _graph_matches(reactant, map_ids, reactant_symbols, reactant_coordinates):
        raise ValueError("reactant coordinate graph does not match mapped graph")
    product_order = tuple(product_by_map[map_id].GetAtomMapNum() for map_id in map_ids)
    if not _graph_matches(product, product_order, product_symbols, product_coordinates):
        raise ValueError("product coordinate graph does not match mapped graph")

    state_evidence = _state_evidence((reaction_dir / "monomer_1.log", reaction_dir / "monomer_2.log"))
    reactant_bonds = _bond_matrix(reactant, map_ids)
    product_bonds = _bond_matrix(product, map_ids)
    edits = _event_edits(reactant_bonds, product_bonds, map_ids)
    if not edits:
        raise ValueError("source row has no event edits")

    record_sha256 = canonical_hash(dict(row))
    graph_sha256 = canonical_hash({"mapped_smiles": reactant_smiles})
    product_graph_sha256 = canonical_hash({"mapped_smiles": product_smiles})
    family_key = _canonical_component_smiles(reactant_smiles)
    family_id = f"dft-da-family:{canonical_hash(family_key)[:24]}"
    archive_prefix = "archive:DATASET_DA_F"
    locator_base = f"{archive_prefix}/{reaction_dir.name}"
    protocol_locator = f"{SOURCE_URL}#TS-tools"
    declared_intent = {
        "permission_mode": "none",
        "reactive_map_ids": [],
        "net_bond_edits": [],
        "provenance": "source reactant endpoint; no reference labels used in input",
        "uses_reference_labels": False,
    }
    event_payload = {
        "edits": [list(edit) for edit in edits],
        "map_ids": list(map_ids),
        "source_record": rid,
    }
    record = TrackBRecord(
        record_id=f"dft-da:{rid}",
        source_dataset="diels-alder-reaction-space",
        source_revision=f"{SOURCE_REVISION};archive_sha256={source_asset_sha256}",
        source_record_id=rid,
        source_asset_sha256=source_asset_sha256,
        source_record_sha256=record_sha256,
        license_record=f"{SOURCE_LICENSE};{SOURCE_URL}",
        parent_reaction_id=row["ID"],
        family_id=family_id,
        split_group=f"dft-da-parent:{row['ID']}",
        atoms={
            "atomic_numbers": [atom.GetAtomicNum() for atom in reactant.GetAtoms()],
            "map_ids": list(map_ids),
        },
        reactant={
            "graph_locator": f"{locator_base}/dataset_xtb_final.csv#{rid}:reactant",
            "graph_sha256": graph_sha256,
            "coordinates_locator": f"{locator_base}/{rid}_reactant.xyz",
            "coordinates_sha256": hashlib.sha256(reactant_bytes).hexdigest(),
            "coordinate_unit": "angstrom",
            "input_origin": "declared_reactant_endpoint",
            "solvent_selection_origin": "none",
            "charge": 0,
            "multiplicity": 1,
            "charge_spin_evidence": state_evidence,
            "microstate_id": f"{family_id}:neutral-singlet",
            "geometry_generation_protocol": "TS-tools mapped-reactant endpoint archive",
            "geometry_generation_protocol_locator": protocol_locator,
            "geometry_generation_protocol_sha256": SOURCE_PROTOCOL_SHA256,
            "declared_intent": declared_intent,
        },
        event_label={
            "locator": f"{locator_base}/dataset_xtb_final.csv#{rid}:mapped-event",
            "sha256": canonical_hash(event_payload),
            "representation": "net_bond_edits_map_ids",
            "evidence": "derived_under_contract",
        },
        product_label={
            "graph_locator": f"{locator_base}/dataset_xtb_final.csv#{rid}:product",
            "graph_sha256": product_graph_sha256,
            "graph_evidence": "observed",
            "coordinates_locator": f"{locator_base}/{rid}_product.xyz",
            "coordinates_sha256": hashlib.sha256(product_bytes).hexdigest(),
            "coordinate_unit": "angstrom",
            "coordinate_evidence": "observed",
            "atom_order_matches_input": True,
        },
        ts_geometry={
            "locator": f"{locator_base}/{rid}_ts.xyz",
            "sha256": hashlib.sha256(ts_bytes).hexdigest(),
            "coordinate_unit": "angstrom",
            "evidence": "observed",
            "atom_order_matches_input": True,
        },
        reference_protocol={**SOURCE_PROTOCOL, "protocol_sha256": SOURCE_PROTOCOL_SHA256},
        pretraining_overlap_audit="not_applicable",
        admission="development_train",
        intended_use="event_geometry_supervision",
        claim_limit="development evidence only; reactant endpoint is declared_reactant_endpoint",
    )
    return DftDaSample(
        record=record,
        reactant_smiles=reactant_smiles,
        product_smiles=product_smiles,
        reactant_coordinates=reactant_coordinates,
        product_coordinates=product_coordinates,
        ts_coordinates=ts_coordinates,
        reactant_bonds=reactant_bonds,
        product_bonds=product_bonds,
        atomic_numbers=tuple(atom.GetAtomicNum() for atom in reactant.GetAtoms()),
        map_ids=map_ids,
    )


def load_dft_da_samples(cache_root: str | Path) -> tuple[list[DftDaSample], dict[str, Any]]:
    """Audit all source rows and return only rows eligible for development."""

    root = Path(cache_root).expanduser()
    source_asset_sha256 = _source_asset_hash(root)
    csv_path = root / "extracted" / "DATASET_DA_F" / "dataset_xtb_final.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing extracted source table: {csv_path}")
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    samples: list[DftDaSample] = []
    rejection_counts: dict[str, int] = {}
    rejection_examples: dict[str, str] = {}

    def rejection_reason(exc: Exception) -> str:
        """Reduce source-parser failures to stable, path-free reason codes."""

        if isinstance(exc, FileNotFoundError):
            return "missing_required_source_file"
        message = str(exc)
        if "outside the frozen CHNO scope" in message:
            return "unsupported_element_scope"
        if "non-integral bond edit" in message:
            return "non_integral_bond_edit"
        if "monomer state logs" in message or "neutral-singlet state" in message:
            return "electronic_state_unverified"
        if "XYZ atom order" in message or "coordinate graph" in message:
            return "coordinate_mapping_or_connectivity_unverified"
        if "mapped SMILES" in message or "mapped atom" in message:
            return "mapped_graph_unverified"
        if "event edits" in message:
            return "event_label_unverified"
        return "source_row_audit_failed"

    for row in rows:
        try:
            samples.append(_row_sample(root / "extracted" / "DATASET_DA_F", row, source_asset_sha256))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            key = rejection_reason(exc)
            rejection_counts[key] = rejection_counts.get(key, 0) + 1
            rejection_examples.setdefault(key, row.get("R", "unknown"))
    samples.sort(key=lambda sample: sample.record.record_id)
    filtered = filter_track_b_records(sample.record for sample in samples)
    accepted_ids = {record.record_id for record in filtered.accepted}
    samples = [sample for sample in samples if sample.record.record_id in accepted_ids]
    audit = {
        "source_url": SOURCE_URL,
        "source_revision": SOURCE_REVISION,
        "source_archive_sha256": source_asset_sha256,
        "source_csv_sha256": hash_file(csv_path),
        "source_row_count": len(rows),
        "state_log_candidates": sum(1 for row in rows if all((root / "extracted" / "DATASET_DA_F" / (row.get("R_dir") or f"reaction_{row['R']}") / name).is_file() for name in ("monomer_1.log", "monomer_2.log"))),
        "graph_and_state_admitted_count": len(samples),
        "admitted_parent_count": len({sample.record.parent_reaction_id for sample in samples}),
        "rejection_counts": dict(sorted(rejection_counts.items(), key=lambda item: (-item[1], item[0]))),
        "rejection_examples": dict(sorted(rejection_examples.items())),
        "filter_quarantine_count": len(filtered.quarantined),
    }
    return samples, audit


def assign_parent_splits(samples: Iterable[DftDaSample], *, seed: str = "dft-da-track-b-v1") -> dict[str, str]:
    """Assign whole family groups (and therefore their parent groups) to splits."""

    assignments: dict[str, str] = {}
    for sample in samples:
        group = sample.record.family_id
        digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") / 2**64
        split = "train" if bucket < 0.70 else "validation" if bucket < 0.85 else "test"
        previous = assignments.setdefault(group, split)
        if previous != split:
            raise ValueError("family split assignment is not deterministic")
    return assignments


def with_splits(samples: Iterable[DftDaSample], *, seed: str = "dft-da-track-b-v1") -> list[DftDaSample]:
    assignments = assign_parent_splits(samples, seed=seed)
    output: list[DftDaSample] = []
    for sample in samples:
        split = assignments[sample.record.family_id]
        output.append(
            DftDaSample(
                record=sample.record.__class__(**{**sample.record.to_dict(), "admission": f"development_{split}"}),
                reactant_smiles=sample.reactant_smiles,
                product_smiles=sample.product_smiles,
                reactant_coordinates=sample.reactant_coordinates,
                product_coordinates=sample.product_coordinates,
                ts_coordinates=sample.ts_coordinates,
                reactant_bonds=sample.reactant_bonds,
                product_bonds=sample.product_bonds,
                atomic_numbers=sample.atomic_numbers,
                map_ids=sample.map_ids,
            )
        )
    return sorted(output, key=lambda sample: sample.record.record_id)


def write_manifest(path: str | Path, samples: Iterable[DftDaSample]) -> None:
    rows = sorted((sample.record.to_dict() for sample in samples), key=lambda row: row["record_id"])
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
