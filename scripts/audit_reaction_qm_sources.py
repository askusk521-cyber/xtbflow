#!/usr/bin/env python3
"""Audit cached Reaction-QM and RGD1 public assets.

The command is intended to run on n2 with the cache paths supplied explicitly.
It never downloads, modifies, or copies source bytes.  Missing HDF5 assets are
reported as unavailable instead of being represented by guessed zero counts.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
from typing import Any, Iterable


CHNOS = frozenset({"C", "H", "N", "O", "S"})
ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "Cl": 17, "Si": 14}


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def audit_file(path: Path, *, url: str, role: str, expected_sha256: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"relative_path": path.name, "name": path.name, "url": url, "role": role}
    if not path.is_file():
        row.update({"status": "unavailable", "size_bytes": None, "sha256": None, "md5": None})
        return row
    sha = digest(path)
    row.update({"size_bytes": path.stat().st_size, "sha256": sha, "md5": digest(path, "md5")})
    if expected_sha256 is None:
        row["status"] = "observed_pending_expected_sha256"
    else:
        row["status"] = "verified" if sha == expected_sha256 else "hash_mismatch"
    return row


def _rdkit():
    try:
        from rdkit import Chem  # type: ignore
    except ImportError as exc:
        raise RuntimeError("RDKit is required for mapped graph auditing") from exc
    return Chem


def _parse_side(smiles: str, chem: Any) -> list[Any] | None:
    # Preserve explicit mapped H atoms; default sanitizing removes them from
    # the graph and would create false mapping mismatches.
    molecules = [chem.MolFromSmiles(part, sanitize=False) for part in smiles.split(".")]
    return None if any(molecule is None for molecule in molecules) else molecules


def _graph_stats(reactant: str, product: str, chem: Any) -> dict[str, Any]:
    result = Counter()
    reactants = _parse_side(reactant, chem)
    products = _parse_side(product, chem)
    if reactants is None or products is None:
        result["smiles_parse_fail"] += 1
        return dict(result)
    r_atoms = [atom for molecule in reactants for atom in molecule.GetAtoms()]
    p_atoms = [atom for molecule in products for atom in molecule.GetAtoms()]
    r_maps = [atom.GetAtomMapNum() for atom in r_atoms]
    p_maps = [atom.GetAtomMapNum() for atom in p_atoms]
    if len(r_maps) != len(set(r_maps)) or len(p_maps) != len(set(p_maps)) or 0 in r_maps or 0 in p_maps:
        result["mapping_invalid"] += 1
        return dict(result)
    if set(r_maps) != set(p_maps):
        result["mapping_set_mismatch"] += 1
        return dict(result)
    result["mapping_complete"] += 1
    symbols = {atom.GetSymbol() for atom in r_atoms + p_atoms}
    if symbols.issubset(CHNOS):
        result["chnos"] += 1
    else:
        result["out_of_chnos"] += 1
    result["neutral_formal_charge_proxy"] += int(
        sum(atom.GetFormalCharge() for atom in r_atoms) == 0
        and sum(atom.GetFormalCharge() for atom in p_atoms) == 0
    )

    def bonds(molecules: Iterable[Any]) -> dict[tuple[int, int], tuple[float, bool]]:
        output: dict[tuple[int, int], tuple[float, bool]] = {}
        for molecule in molecules:
            for bond in molecule.GetBonds():
                i = bond.GetBeginAtom().GetAtomMapNum()
                j = bond.GetEndAtom().GetAtomMapNum()
                output[tuple(sorted((i, j)))] = (float(bond.GetBondTypeAsDouble()), bool(bond.GetIsAromatic()))
        return output

    r_bonds = bonds(reactants)
    p_bonds = bonds(products)
    result["aromatic_bond_record"] += int(any(value[1] for value in (*r_bonds.values(), *p_bonds.values())))
    edits = sum(
        (p_bonds.get(key, (0.0, False))[0] - r_bonds.get(key, (0.0, False))[0]) != 0
        for key in set(r_bonds) | set(p_bonds)
    )
    result["event_change"] += int(edits > 0)
    result["no_bond_change"] += int(edits == 0)
    return dict(result)


def audit_mapped_csv(path: Path, *, id_field: str, reactant_field: str, product_field: str, extra_fields: tuple[str, ...] = ()) -> dict[str, Any]:
    chem = _rdkit()
    counts = Counter()
    ids: list[str] = []
    missing_fields = Counter()
    extra_values = {field: Counter() for field in extra_fields}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        required = {id_field, reactant_field, product_field, *extra_fields}
        missing_schema = sorted(required - set(fields))
        if missing_schema:
            raise ValueError(f"{path} is missing fields: {missing_schema}")
        for row in reader:
            counts["rows"] += 1
            ids.append(row[id_field])
            for field in fields:
                if row.get(field, "").strip() == "":
                    missing_fields[field] += 1
            for field in extra_fields:
                extra_values[field][row[field]] += 1
            for key, value in _graph_stats(row[reactant_field], row[product_field], chem).items():
                counts[key] += value
    counts["unique_ids"] = len(set(ids))
    counts["duplicate_ids"] = len(ids) - counts["unique_ids"]
    counts["charge_field_missing"] = counts["rows"]
    counts["multiplicity_field_missing"] = counts["rows"]
    return {
        "fields": list(fields),
        "counts": dict(counts),
        "missing_field_counts": dict(missing_fields),
        "extra_value_counts": {
            key: {"unique": len(counter), "top": counter.most_common(20)}
            for key, counter in extra_values.items()
        },
    }


def audit_reaction_info(path: Path) -> dict[str, Any]:
    counts = Counter()
    missing = Counter()
    ids: list[str] = []
    fields: list[str] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        for row in reader:
            counts["rows"] += 1
            ids.append(row.get("reaction_id", ""))
            for key, value in row.items():
                if value is None or not value.strip():
                    missing[key] += 1
    counts["unique_reaction_ids"] = len(set(ids))
    counts["duplicate_reaction_ids"] = len(ids) - counts["unique_reaction_ids"]
    return {"fields": fields, "counts": dict(counts), "missing_field_counts": dict(missing)}


def audit_hdf5(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "unavailable", "reaction_group_count": None, "root_keys": None}
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        return {"status": "dependency_missing", "error": str(exc), "reaction_group_count": None}
    groups = 0
    species_field_counts: Counter[str] = Counter()

    def visit(group: Any) -> None:
        nonlocal groups
        labels = {str(key).upper() for key in group.keys()}
        if any(label.startswith("R") for label in labels) and any(label.startswith("P") for label in labels) and any(label.startswith("TS") for label in labels):
            groups += 1
            for key in group.keys():
                child = group[key]
                if hasattr(child, "keys"):
                    for field in ("smiles", "EHG", "charge", "multiplicity", "atomic_numbers", "coordinates", "energies", "forces"):
                        if field in child:
                            species_field_counts[field] += 1
            return
        for key in group.keys():
            child = group[key]
            if hasattr(child, "keys"):
                visit(child)

    with h5py.File(path, "r") as handle:
        all_root_keys = sorted(str(key) for key in handle.keys())
        root_keys = {"count": len(all_root_keys), "first": all_root_keys[:5], "last": all_root_keys[-5:]}
        visit(handle)
    return {"status": "observed", "reaction_group_count": groups, "root_keys": root_keys, "species_field_counts": dict(species_field_counts)}


def audit_reaction_hdf5_records(path: Path, reaction_info_ids: set[str]) -> dict[str, Any]:
    """Audit Reaction-QM TS mapping/state/coordinate fields from the HDF5."""

    if not path.is_file():
        return {"status": "unavailable"}
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        return {"status": "dependency_missing", "error": str(exc)}
    counts = Counter()
    reasons = Counter()
    examples: dict[str, str] = {}
    chem = _rdkit()

    def reason(code: str, record_id: str) -> None:
        reasons[code] += 1
        examples.setdefault(code, record_id)

    with h5py.File(path, "r") as handle:
        for block in handle.values():
            for record_id, group in block.items():
                counts["records"] += 1
                if record_id not in reaction_info_ids:
                    reason("reaction_info_missing", record_id)
                if "TS" not in group:
                    reason("ts_group_missing", record_id)
                    continue
                ts = group["TS"]
                fields = set(ts.keys())
                for field in ("smiles", "charge", "multiplicity", "atomic_numbers", "coordinates"):
                    if field not in fields:
                        reason(f"ts_{field}_missing", record_id)
                if not {"smiles", "charge", "multiplicity", "atomic_numbers", "coordinates"}.issubset(fields):
                    continue
                raw_smiles = ts["smiles"][()]
                smiles = raw_smiles.decode() if isinstance(raw_smiles, bytes) else str(raw_smiles)
                sides = [_parse_side(side, chem) for side in smiles.split(">>")]
                if len(sides) != 2 or any(side is None for side in sides):
                    reason("ts_smiles_parse_fail", record_id)
                    continue
                # Calculate counters directly from the parsed TS sides so the
                # reaction arrow is never treated as a SMILES component.
                reactants, products = sides
                r_atoms = [atom for molecule in reactants for atom in molecule.GetAtoms()]
                p_atoms = [atom for molecule in products for atom in molecule.GetAtoms()]
                r_maps = [atom.GetAtomMapNum() for atom in r_atoms]
                p_maps = [atom.GetAtomMapNum() for atom in p_atoms]
                if len(r_maps) != len(set(r_maps)) or len(p_maps) != len(set(p_maps)) or 0 in r_maps or 0 in p_maps:
                    reason("mapping_invalid", record_id)
                    continue
                if set(r_maps) != set(p_maps):
                    reason("mapping_set_mismatch", record_id)
                    continue
                counts["mapping_complete"] += 1
                symbols = {atom.GetSymbol() for atom in r_atoms + p_atoms}
                if symbols.issubset(CHNOS):
                    counts["chnos"] += 1
                else:
                    reason("out_of_chnos", record_id)
                charge = int(ts["charge"][()])
                multiplicity = int(ts["multiplicity"][()])
                if charge == 0 and multiplicity == 1:
                    counts["neutral_closed_shell"] += 1
                else:
                    reason("non_neutral_or_open_shell", record_id)
                expected_atoms = len(r_maps)
                if len(ts["atomic_numbers"]) != expected_atoms or tuple(ts["coordinates"].shape) != (expected_atoms, 3):
                    reason("ts_coordinate_shape_mismatch", record_id)
                expected_numbers = [ATOMIC_NUMBERS.get(atom.GetSymbol(), -1) for atom in r_atoms]
                source_numbers = [int(value) for value in ts["atomic_numbers"][()].tolist()]
                if source_numbers != expected_numbers:
                    if sorted(source_numbers) == sorted(expected_numbers):
                        reason("coordinate_atom_order_unverified", record_id)
                    else:
                        reason("coordinate_atom_inventory_mismatch", record_id)
                if any(bond.GetIsAromatic() for molecule in (*reactants, *products) for bond in molecule.GetBonds()):
                    reason("aromatic_bond_ambiguity", record_id)
                # The endpoint event is computed only as a count here; the
                # adapter performs the full provenance-bearing label.
                def bond_map(molecules: Iterable[Any]) -> dict[tuple[int, int], float]:
                    output: dict[tuple[int, int], float] = {}
                    for molecule in molecules:
                        for bond in molecule.GetBonds():
                            key = tuple(sorted((bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum())))
                            output[key] = float(bond.GetBondTypeAsDouble())
                    return output
                rb, pb = bond_map(reactants), bond_map(products)
                if any(pb.get(key, 0.0) != rb.get(key, 0.0) for key in set(rb) | set(pb)):
                    counts["endpoint_event_change"] += 1
                else:
                    reason("no_bond_change", record_id)
    counts["reaction_info_rows"] = len(reaction_info_ids)
    return {"status": "observed", "counts": dict(counts), "reason_counts": dict(reasons), "reason_examples": examples}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    rq = Path(args.reaction_qm_cache)
    rgd = Path(args.rgd1_cache)
    report: dict[str, Any] = {
        "schema": "xtbflow-public-reaction-source-audit/v1",
        "generated_on": args.generated_on,
        "execution": {"host": platform.node(), "python": platform.python_version(), "pid": os.getpid()},
        "sources": {
            "reaction_qm": {"revision": "zenodo:18551029@v2", "url": "https://doi.org/10.5281/zenodo.18551029", "license_status": "verify_record_license_before_redistribution"},
            "rgd1": {"revision": "figshare:21066901@v6", "url": "https://doi.org/10.6084/m9.figshare.21066901.v6", "license_status": "repository_gpl3_data_terms_require_check"},
        },
        "assets": {
            "reaction_qm": [
                audit_file(rq / "B3LYPD3_TZVP_reaction_info.csv", url="https://zenodo.org/records/18551029/files/B3LYPD3_TZVP_reaction_info.csv?download=1", role="reaction_info", expected_sha256="2facf37090a4cba872394ec6ab0c360b011d41acff9658268ff2eb61e6fd5ae1"),
                audit_file(rq / "B3LYPD3_TZVP.h5", url="https://zenodo.org/records/18551029/files/B3LYPD3_TZVP.h5?download=1", role="geometry_and_ts", expected_sha256="3d0fc655819a9a2747f554a9025cd36cbdffd1175c4a1d40934b6fe5530af82a"),
                audit_file(rq / "B3LYPD3_TZVP_IRC.h5", url="https://zenodo.org/records/18551029/files/B3LYPD3_TZVP_IRC.h5?download=1", role="irc_path"),
            ],
            "rgd1": [
                audit_file(rgd / "RGD1CHNO_AMsmiles.csv", url="https://ndownloader.figshare.com/files/40272727", role="mapped_index", expected_sha256="7b0ecae0a7f7b2439a03bed4bfbf80fbc9b647b0823e5a8e7b1e3be96d0ad05e"),
                audit_file(rgd / "DFT_reaction_info.csv", url="https://ndownloader.figshare.com/files/40273231", role="dft_reaction_info", expected_sha256="afb3bfb9ca5e44fb1a6ea44664da56e0ef2a2ee95c82b22d651a778fad5e0b00"),
                audit_file(rgd / "RandP_smiles.txt", url="https://ndownloader.figshare.com/files/43291989", role="endpoint_mapping", expected_sha256="6ca74bafe9286ac1c6cba48e700e5f5100e0bb3d6a8f05592efd214ffdc624ff"),
                audit_file(rgd / "RGD1_CHNO.h5", url="https://ndownloader.figshare.com/files/38170323", role="geometry_and_ts", expected_sha256="ed125b4cb1eac9af670a7cae8b9d29c88a9f8be24d0206a027f0f2c035f8f268"),
                audit_file(rgd / "RGD1_RPs.h5", url="https://ndownloader.figshare.com/files/43293162", role="endpoint_geometry"),
            ],
        },
    }
    reaction_info = rq / "B3LYPD3_TZVP_reaction_info.csv"
    reaction_info_audit = audit_reaction_info(reaction_info) if reaction_info.is_file() else {"status": "unavailable"}
    reaction_info_ids: set[str] = set()
    if reaction_info.is_file():
        with reaction_info.open(newline="", encoding="utf-8-sig") as handle:
            reaction_info_ids = {row["reaction_id"] for row in csv.DictReader(handle)}
    report["reaction_qm"] = {
        "reaction_info": reaction_info_audit,
        "hdf5": audit_hdf5(rq / "B3LYPD3_TZVP.h5"),
        "record_contract_audit": audit_reaction_hdf5_records(rq / "B3LYPD3_TZVP.h5", reaction_info_ids),
    }
    mapped = rgd / "RGD1CHNO_AMsmiles.csv"
    report["rgd1"] = {
        "mapped_smiles": audit_mapped_csv(mapped, id_field="reaction", reactant_field="reactant", product_field="product") if mapped.is_file() else {"status": "unavailable"},
        "dft_reaction_info": audit_mapped_csv(rgd / "DFT_reaction_info.csv", id_field="channel", reactant_field="reactant", product_field="product", extra_fields=("type", "R_ind")) if (rgd / "DFT_reaction_info.csv").is_file() else {"status": "unavailable"},
        "hdf5": audit_hdf5(rgd / "RGD1_CHNO.h5"),
        "endpoint_hdf5": audit_hdf5(rgd / "RGD1_RPs.h5"),
    }
    report["admission_summary"] = {
        "reaction_qm": {"status": "awaiting_hdf5_and_record_contract", "confirmatory": False, "claim_limit": "Do not call endpoint-derived labels electron-density paths."},
        "rgd1": {"status": "cross_source_validation_only", "confirmatory": False, "reason": "RGD1 remains separate from Reaction-QM training and lacks source-wide explicit charge/multiplicity in the CSV index."},
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reaction-qm-cache", required=True, type=Path)
    parser.add_argument("--rgd1-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--generated-on", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing report: {args.output}")
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
