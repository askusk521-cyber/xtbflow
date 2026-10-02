#!/usr/bin/env python3
"""Audit Reaction-QM coordinate rows against mapped-atom order.

The upstream Reaction-QM code maps an atom-map number ``m`` to the zero-based
array index ``m - 1`` when all map numbers are present and unique.  This audit
checks that convention directly: it parses the mapped SMILES, builds
``map_id -> atomic_number``, sorts by map id, and compares the resulting list
with the HDF5 ``atomic_numbers`` array.  It never uses textual SMILES order as
the expected coordinate order.

The HDF5 writer is not included in the upstream repository.  Consequently the
source-code evidence and the empirical HDF5 audit are recorded separately in
the JSON output; an exact match is not silently promoted to evidence of the
other admission gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from rdkit import Chem


UPSTREAM_REPOSITORY = "https://github.com/Kangbeomgyu/Reaction-QM"
UPSTREAM_COMMIT = "3a553bf1562356a4f978ab5daa39778e434a9310"
SOURCE_REVISION = "zenodo:18551029@v2"
EXPECTED_H5_SHA256 = "3d0fc655819a9a2747f554a9025cd36cbdffd1175c4a1d40934b6fe5530af82a"

_RXN_NUMBER = re.compile(r"(?:RXN_)?(\d+)$")


def _reaction_sort_key(name: str) -> tuple[int, str]:
    match = _RXN_NUMBER.fullmatch(name)
    return (int(match.group(1)), name) if match else (2**63 - 1, name)


def _decode_smiles(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _parse_mapped_side(smiles: str, *, reaction_id: str, species: str) -> tuple[dict[int, int], list[str]]:
    """Return map id -> Z and machine-readable parse failures.

    ``species=TS`` is passed a full ``R>>P`` reaction string.  The TS
    coordinate rows use the reactant-side atom inventory, so only that side is
    used for the row-order check.  Product-side consistency is checked as a
    separate duplicate-map validation.
    """

    failures: list[str] = []
    sides = smiles.split(">>")
    if species == "TS":
        if len(sides) != 2:
            return {}, ["malformed_reaction_smiles"]
        target_side = sides[0]
        all_sides = sides
    else:
        if len(sides) != 1:
            return {}, ["malformed_endpoint_smiles"]
        target_side = sides[0]
        all_sides = sides

    map_to_z: dict[int, int] = {}
    target_map_to_z: dict[int, int] = {}
    target_map_ids: list[int] = []
    for side_index, side in enumerate(all_sides):
        for fragment_index, fragment in enumerate(side.split(".")):
            if not fragment:
                failures.append("empty_fragment")
                continue
            mol = Chem.MolFromSmiles(fragment, sanitize=False)
            if mol is None:
                failures.append("smiles_parse_failed")
                continue
            is_target = side_index == 0
            for atom in mol.GetAtoms():
                map_id = int(atom.GetAtomMapNum())
                if map_id <= 0:
                    failures.append("map_id_nonpositive_or_missing")
                    continue
                atomic_number = int(atom.GetAtomicNum())
                if map_id in map_to_z and map_to_z[map_id] != atomic_number:
                    failures.append("duplicate_map_id_conflicting_element")
                elif map_id in map_to_z and is_target:
                    failures.append("duplicate_map_id_in_target_side")
                else:
                    map_to_z.setdefault(map_id, atomic_number)
                if is_target:
                    target_map_ids.append(map_id)
                    target_map_to_z.setdefault(map_id, atomic_number)

    if len(target_map_ids) != len(set(target_map_ids)):
        failures.append("duplicate_map_id_in_target_side")
    unique_target = sorted(set(target_map_ids))
    if unique_target and unique_target != list(range(1, len(unique_target) + 1)):
        failures.append("map_id_not_contiguous_from_one")
    if not unique_target:
        failures.append("no_target_map_ids")
    return target_map_to_z, sorted(set(failures))


def _expected_map_order(smiles: str, *, reaction_id: str, species: str) -> tuple[list[int] | None, list[str]]:
    map_to_z, failures = _parse_mapped_side(smiles, reaction_id=reaction_id, species=species)
    if failures:
        return None, failures
    ids = sorted(map_to_z)
    return [map_to_z[map_id] for map_id in ids], []


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_reaction_ids(handle: h5py.File) -> Iterable[str]:
    names: list[str] = []
    for chunk_name in handle.keys():
        names.extend(str(name) for name in handle[chunk_name].keys())
    yield from sorted(names, key=_reaction_sort_key)


def _audit_species(
    group: h5py.Group,
    reaction_id: str,
    species: str,
    *,
    failures_by_record: dict[str, list[str]],
) -> tuple[bool, str | None]:
    if species not in group:
        reason = "missing_species_group"
        failures_by_record.setdefault(f"{reaction_id}:{species}", []).append(reason)
        return False, reason
    item = group[species]
    required = {"smiles", "atomic_numbers", "coordinates"}
    missing = sorted(required.difference(item.keys()))
    if missing:
        reason = "missing_fields:" + ",".join(missing)
        failures_by_record.setdefault(f"{reaction_id}:{species}", []).append(reason)
        return False, reason
    smiles = _decode_smiles(item["smiles"][()])
    observed = np.asarray(item["atomic_numbers"][()], dtype=np.int64).tolist()
    coordinates = np.asarray(item["coordinates"][()])
    local_failures: list[str] = []
    if coordinates.ndim != 2 or coordinates.shape != (len(observed), 3):
        local_failures.append("coordinate_shape_mismatch")
    expected, parse_failures = _expected_map_order(smiles, reaction_id=reaction_id, species=species)
    local_failures.extend(parse_failures)
    if expected is not None:
        if len(expected) != len(observed):
            local_failures.append("atomic_inventory_mismatch")
        elif expected != observed:
            local_failures.append("map_order_atomic_numbers_mismatch")
    if local_failures:
        failures_by_record.setdefault(f"{reaction_id}:{species}", []).extend(sorted(set(local_failures)))
        return False, local_failures[0]
    return True, None


def audit(path: Path, *, limit: int | None, verify_sha256: bool) -> dict[str, Any]:
    failures_by_record: dict[str, list[str]] = {}
    summary = Counter()
    species_summary: dict[str, Counter[str]] = {name: Counter() for name in ("R", "P", "TS")}
    processed_ids: list[str] = []

    with h5py.File(path, "r") as handle:
        chunk_count = len(handle.keys())
        all_ids = list(_iter_reaction_ids(handle))
        selected_ids = all_ids if limit is None else all_ids[:limit]
        for reaction_id in selected_ids:
            group = next(handle[chunk][reaction_id] for chunk in handle.keys() if reaction_id in handle[chunk])
            processed_ids.append(reaction_id)
            summary["reaction_records_processed"] += 1
            ts_ok, _ = _audit_species(group, reaction_id, "TS", failures_by_record=failures_by_record)
            species_summary["TS"]["exact_map_order_match" if ts_ok else "failed"] += 1
            summary["exact_map_order_match" if ts_ok else "ts_failed"] += 1
            for species in sorted(name for name in group.keys() if name.startswith("R") or name.startswith("P")):
                category = species[0]
                ok, _ = _audit_species(group, reaction_id, species, failures_by_record=failures_by_record)
                species_summary[category]["exact_map_order_match" if ok else "failed"] += 1

    reason_counts = Counter(reason for reasons in failures_by_record.values() for reason in set(reasons))
    map_id_reason_names = {
        "map_id_nonpositive_or_missing",
        "duplicate_map_id_conflicting_element",
        "duplicate_map_id_in_target_side",
        "map_id_not_contiguous_from_one",
        "no_target_map_ids",
    }
    atomic_inventory_reason_names = {"atomic_inventory_mismatch"}
    asset_version_reason_names = {"asset_version_mismatch"}
    summary["map_id_invalid"] = sum(reason_counts[name] for name in map_id_reason_names)
    summary["atomic_inventory_mismatch"] = sum(reason_counts[name] for name in atomic_inventory_reason_names)
    summary["asset_version_mismatch"] = sum(reason_counts[name] for name in asset_version_reason_names)
    summary["other_failure"] = sum(
        count
        for name, count in reason_counts.items()
        if name not in map_id_reason_names
        and name not in atomic_inventory_reason_names
        and name not in asset_version_reason_names
        and name != "map_order_atomic_numbers_mismatch"
    )
    summary["failed_reaction_records"] = sum(
        1 for reaction_id in processed_ids if any(key.startswith(reaction_id + ":") for key in failures_by_record)
    )
    output: dict[str, Any] = {
        "schema": "xtbflow-reaction-qm-coordinate-map-audit/v1",
        "source_dataset": "Reaction-QM",
        "source_revision": SOURCE_REVISION,
        "source_asset": {
            "relative_path": path.name,
            "sha256_expected": EXPECTED_H5_SHA256,
            "sha256_computed": _sha256(path) if verify_sha256 else None,
            "sha256_verified": bool(verify_sha256 and _sha256(path) == EXPECTED_H5_SHA256),
            "size_bytes": path.stat().st_size,
        },
        "upstream_provenance": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "coordinate_mapping_convention": "mapped atom map number m is converted to zero-based index m-1; if all map numbers are unique, get_permuted_molecule reorders atom_list and graph arrays to that index",
            "source_code_evidence": [
                {
                    "file": "chem.py",
                    "lines": "621-632",
                    "behavior": "GetAtomMapNum()-1 builds atom_mapping; complete unique mapping invokes process.get_permuted_molecule",
                },
                {
                    "file": "process.py",
                    "lines": "900-944",
                    "behavior": "permutation is applied to atom_list, atom features, and graph matrices via new_index[permutation[i]]=old_index[i]",
                },
                {
                    "file": "process.py",
                    "lines": "402-413",
                    "behavior": "coordinate row i is assigned to atom_list[i]",
                },
                {
                    "file": "README.md",
                    "lines": "50-59,74-86",
                    "behavior": "HDF5 examples print atomic_numbers followed by coordinate rows as XYZ-like pairs",
                },
            ],
            "hdf5_writer_present_in_upstream_commit": False,
        },
        "audit": {
            "limit": limit,
            "hdf5_chunk_count": chunk_count,
            "reaction_records_selected": len(processed_ids),
            "reaction_ids_first": processed_ids[:5],
            "reaction_ids_last": processed_ids[-5:],
            "counts": dict(summary),
            "species_counts": {key: dict(value) for key, value in species_summary.items()},
            "reason_counts": dict(reason_counts),
            "failure_records": [
                {"record_id": record_id, "reasons": sorted(set(reasons))}
                for record_id, reasons in sorted(failures_by_record.items())
            ],
        },
        "admission_boundary": {
            "coordinate_mapping_evidence": "source_code_verified_plus_asset_map_order_audit",
            "claim_limit": "Atomic-number map-order agreement only establishes coordinate-row to map-index order; charge/spin, endpoint local-to-global correspondence, elementary-step pairing, grouping, licence, and leakage gates remain independent.",
        },
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Number of reaction records, sorted by numeric reaction ID")
    parser.add_argument("--verify-sha256", action="store_true", help="Read the complete HDF5 file and verify its raw SHA-256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    report = audit(args.h5, limit=args.limit, verify_sha256=args.verify_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["audit"]["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
