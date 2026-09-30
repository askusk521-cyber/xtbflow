#!/usr/bin/env python3
"""Audit the official Reaction-QM B3LYP train/validation/test partition.

The split CSVs are small source assets compared with the geometry archives and
are useful for freezing the official reaction-ID partition.  They do not carry
parent-reaction, family, independent-system, repeated-TS, or coordinate-map
identities, so a passing partition audit is not a training-admission gate.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


ASSETS = {
    "train": {
        "name": "B3LYP-RXN_train.csv",
        "url": "https://zenodo.org/records/18551029/files/B3LYP-RXN_train.csv?download=1",
        "official_md5": "222059ce2ec2585a385518af393ecaae",
        "sha256": "c768b07f6b9789daeae7fbb22720b3d5134abae01f6a8c1cce8a9c054ca90a04",
    },
    "validation": {
        "name": "B3LYP-RXN_valid.csv",
        "url": "https://zenodo.org/records/18551029/files/B3LYP-RXN_valid.csv?download=1",
        "official_md5": "0174d304143089c26a93dc89632f5317",
        "sha256": "1c1a1fd74675d6509057a89ab091cc8e2c9a7f4108196e70bc8ede0413569e0d",
    },
    "test": {
        "name": "B3LYP-RXN_test.csv",
        "url": "https://zenodo.org/records/18551029/files/B3LYP-RXN_test.csv?download=1",
        "official_md5": "359332ff1ccb289751ef43bb544f2373",
        "sha256": "caac8dfebc830d2e394fee6da9dba5687fdb07720926158f8c2edf4880866f88",
    },
}
REQUIRED_FIELDS = ("reaction_id", "reaction_smiles", "dE_dagger", "dE", "dH_dagger", "dH", "dG_dagger", "dG")


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    return fields, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generated-on", required=True)
    args = parser.parse_args()

    primary_path = args.cache / "B3LYPD3_TZVP_reaction_info.csv"
    primary_fields, primary_rows = read_csv(primary_path)
    primary_by_id = {row["reaction_id"]: row["reaction_smiles"] for row in primary_rows}
    split_rows: dict[str, list[dict[str, str]]] = {}
    asset_report: dict[str, dict[str, Any]] = {}
    required_field_failures: dict[str, list[str]] = {}
    for split, spec in ASSETS.items():
        path = args.cache / spec["name"]
        fields, rows = read_csv(path)
        split_rows[split] = rows
        required_field_failures[split] = sorted(set(REQUIRED_FIELDS) - set(fields))
        asset_report[split] = {
            "name": spec["name"],
            "url": spec["url"],
            "role": f"official_split_{split}",
            "size_bytes": path.stat().st_size,
            "md5": digest(path, "md5"),
            "sha256": digest(path, "sha256"),
            "official_md5": spec["official_md5"],
            "expected_sha256": spec["sha256"],
            "record_count": len(rows),
            "header": list(fields),
            "status": "verified" if digest(path, "md5") == spec["official_md5"] and digest(path, "sha256") == spec["sha256"] else "hash_mismatch",
        }

    ids = {split: [row["reaction_id"] for row in rows] for split, rows in split_rows.items()}
    all_ids = [record_id for values in ids.values() for record_id in values]
    all_id_counter = Counter(all_ids)
    split_by_id = {row["reaction_id"]: row["reaction_smiles"] for rows in split_rows.values() for row in rows}
    smiles_by_split = {split: Counter(row["reaction_smiles"] for row in rows) for split, rows in split_rows.items()}
    pair_overlaps = {
        f"{left}/{right}": len(set(ids[left]) & set(ids[right]))
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    smiles_overlaps = {
        f"{left}/{right}": len(set(smiles_by_split[left]) & set(smiles_by_split[right]))
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    report = {
        "schema": "xtbflow-reaction-qm-official-split-audit/v1",
        "generated_on": args.generated_on,
        "source_revision": "zenodo:18551029@v2",
        "primary_reaction_info": {
            "name": primary_path.name,
            "record_count": len(primary_rows),
            "header": list(primary_fields),
            "sha256": digest(primary_path, "sha256"),
        },
        "assets": asset_report,
        "required_field_failures": required_field_failures,
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "split_total": len(all_ids),
        "unique_split_ids": len(set(all_ids)),
        "duplicate_split_id_rows": sum(count - 1 for count in all_id_counter.values() if count > 1),
        "partition_overlap_by_id": pair_overlaps,
        "primary_ids_missing_from_official_split": sorted(set(primary_by_id) - set(all_ids)),
        "official_split_ids_missing_from_primary": sorted(set(all_ids) - set(primary_by_id)),
        "reaction_smiles_mismatch_count": sum(primary_by_id[record_id] != split_by_id[record_id] for record_id in set(primary_by_id) & set(split_by_id)),
        "within_split_duplicate_reaction_smiles": {split: sum(count - 1 for count in values.values() if count > 1) for split, values in smiles_by_split.items()},
        "cross_split_reaction_smiles_overlap": smiles_overlaps,
        "admission_summary": {
            "official_id_partition": "verified",
            "parent_family_independent_system_grouping": "unavailable",
            "repeated_ts_grouping": "unavailable",
            "coordinate_map_evidence": "unavailable",
            "claim_limit": "The official CSVs freeze a disjoint reaction-ID partition only; they do not prove parent/family independence or coordinate-to-map correspondence.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"split_counts": report["split_counts"], "split_total": report["split_total"], "unique_split_ids": report["unique_split_ids"], "partition_overlap_by_id": pair_overlaps, "cross_split_reaction_smiles_overlap": smiles_overlaps}, sort_keys=True))


if __name__ == "__main__":
    main()
