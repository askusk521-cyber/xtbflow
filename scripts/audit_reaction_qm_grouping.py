#!/usr/bin/env python3
"""Audit official Reaction-QM identity metadata without inventing groups.

This report distinguishes cross-level record identity from parent/family or
coordinate-map evidence.  It reads only an existing n2 cache and never writes
or mutates source assets.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


CHNOS = frozenset({"C", "H", "N", "O", "S"})


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _rdkit():
    try:
        from rdkit import Chem  # type: ignore
    except ImportError as exc:
        raise RuntimeError("RDKit is required for mapped identity auditing") from exc
    return Chem


def mapped_graph_stats(smiles: str, chem: Any) -> dict[str, int]:
    counts = Counter()
    sides = smiles.split(">>")
    if len(sides) != 2:
        counts["reaction_arrow_invalid"] += 1
        return dict(counts)
    parsed = []
    for side in sides:
        molecules = [chem.MolFromSmiles(part, sanitize=False) for part in side.split(".")]
        if any(molecule is None for molecule in molecules):
            counts["smiles_parse_fail"] += 1
            return dict(counts)
        parsed.append(molecules)
    reactants, products = parsed
    r_atoms = [atom for molecule in reactants for atom in molecule.GetAtoms()]
    p_atoms = [atom for molecule in products for atom in molecule.GetAtoms()]
    r_maps = [atom.GetAtomMapNum() for atom in r_atoms]
    p_maps = [atom.GetAtomMapNum() for atom in p_atoms]
    if any(value <= 0 for value in (*r_maps, *p_maps)) or len(r_maps) != len(set(r_maps)) or len(p_maps) != len(set(p_maps)):
        counts["mapping_invalid"] += 1
        return dict(counts)
    if set(r_maps) != set(p_maps):
        counts["mapping_set_mismatch"] += 1
        return dict(counts)
    counts["mapping_complete"] += 1
    counts["chnos"] += int({atom.GetSymbol() for atom in (*r_atoms, *p_atoms)}.issubset(CHNOS))

    def bond_map(molecules: list[Any]) -> dict[tuple[int, int], float]:
        result: dict[tuple[int, int], float] = {}
        for molecule in molecules:
            for bond in molecule.GetBonds():
                key = tuple(sorted((bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum())))
                result[key] = float(bond.GetBondTypeAsDouble())
        return result

    before, after = bond_map(reactants), bond_map(products)
    counts["endpoint_event_change"] += int(any(after.get(key, 0.0) != before.get(key, 0.0) for key in set(before) | set(after)))
    return dict(counts)


def audit_common(path: Path, primary_ids: set[str]) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "unavailable"}
    chem = _rdkit()
    counts = Counter()
    missing = Counter()
    b3_ids: list[str] = []
    gfn_ids: list[str] = []
    smiles: list[str] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        required = {"GFN_ID", "B3LYP_ID", "RXN_SMILES"}
        if not required.issubset(fields):
            raise ValueError(f"common metadata is missing fields: {sorted(required - set(fields))}")
        for row in reader:
            counts["rows"] += 1
            b3_ids.append(row["B3LYP_ID"])
            gfn_ids.append(row["GFN_ID"])
            smiles.append(row["RXN_SMILES"])
            for field in fields:
                if not (row.get(field) or "").strip():
                    missing[field] += 1
            for key, value in mapped_graph_stats(row["RXN_SMILES"], chem).items():
                counts[key] += value
    counts["unique_b3lyp_ids"] = len(set(b3_ids))
    counts["duplicate_b3lyp_ids"] = len(b3_ids) - counts["unique_b3lyp_ids"]
    counts["unique_gfn_ids"] = len(set(gfn_ids))
    counts["duplicate_gfn_ids"] = len(gfn_ids) - counts["unique_gfn_ids"]
    counts["unique_rxn_smiles"] = len(set(smiles))
    counts["duplicate_rxn_smiles"] = len(smiles) - counts["unique_rxn_smiles"]
    counts["b3lyp_ids_missing_from_reaction_info"] = len(set(b3_ids) - primary_ids)
    return {
        "fields": fields,
        "counts": dict(counts),
        "missing_field_counts": dict(missing),
        "grouping_fields_present": [],
        "coordinate_map_fields_present": [],
    }


def audit_combinations(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "unavailable"}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    nonempty = [line for line in lines if line.strip()]
    return {
        "fields": ["reactant_combination_smiles"],
        "counts": {
            "lines": len(lines),
            "nonempty_lines": len(nonempty),
            "unique_nonempty_lines": len(set(nonempty)),
            "blank_lines": len(lines) - len(nonempty),
            "lines_with_reaction_ids": sum("RXN_" in line for line in nonempty),
        },
        "grouping_fields_present": [],
        "coordinate_map_fields_present": [],
    }


def build_report(cache: Path, generated_on: str) -> dict[str, Any]:
    reaction_info = cache / "B3LYPD3_TZVP_reaction_info.csv"
    primary_ids: set[str] = set()
    with reaction_info.open(newline="", encoding="utf-8-sig") as handle:
        primary_ids = {row["reaction_id"] for row in csv.DictReader(handle)}
    common = cache / "common_reaction_info.csv"
    combinations = cache / "reactant_combinations.txt"
    assets = []
    for path, role, url in (
        (common, "cross_level_metadata", "https://zenodo.org/records/18551029/files/common_reaction_info.csv?download=1"),
        (combinations, "reactant_enumeration", "https://zenodo.org/records/18551029/files/reactant_combinations.txt?download=1"),
    ):
        assets.append({
            "name": path.name,
            "url": url,
            "role": role,
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": digest(path) if path.is_file() else None,
            "md5": digest(path, "md5") if path.is_file() else None,
            "status": "verified" if path.is_file() else "unavailable",
        })
    return {
        "schema": "xtbflow-reaction-qm-grouping-audit/v1",
        "generated_on": generated_on,
        "source_revision": "zenodo:18551029@v2",
        "source_url": "https://doi.org/10.5281/zenodo.18551029",
        "assets": assets,
        "common_reaction_info": audit_common(common, primary_ids),
        "reactant_combinations": audit_combinations(combinations),
        "admission_summary": {
            "status": "identity_metadata_only",
            "parent_family_independent_system_and_repeated_ts": "unavailable",
            "coordinate_map": "unavailable",
            "claim_limit": "Cross-level IDs are source identities, not grouping or coordinate-map evidence.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--generated-on", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing report: {args.output}")
    args.output.write_text(json.dumps(build_report(args.cache, args.generated_on), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
