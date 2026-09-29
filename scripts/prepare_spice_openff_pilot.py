#!/usr/bin/env python3
"""Audit and freeze a bounded SPICE2 OpenFF E/F pilot manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

HARTREE_KJ_PER_MOL = 2625.499639479163
ANGSTROM_PER_NM = 10.0
ALLOWED_Z = frozenset({1, 6, 7, 8, 16})
SYMBOL_BY_Z = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S"}
REFERENCE_PROTOCOL_ID = "spice2-openff-b3lyp-d3bj-dzvp-v1.1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _unit(dataset: Any) -> str:
    value = dataset.attrs.get("u")
    return _text(value)


SPLIT_SEED = "spice2-openff-pilot-v1-7"
SPLIT_RATIOS = {"train": 0.75, "validation": 0.125, "test": 0.125}


def _split_map(keys: list[str]) -> dict[str, str]:
    """Use the repository's hash-bucket split contract on complete parent groups."""

    train_boundary = SPLIT_RATIOS["train"]
    validation_boundary = train_boundary + SPLIT_RATIOS["validation"]
    output = {}
    for key in keys:
        digest = hashlib.sha256(f"{SPLIT_SEED}:{key}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        output[key] = "train" if value < train_boundary else "validation" if value < validation_boundary else "test"
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--public-records-base", type=Path)
    parser.add_argument("--public-records-output", type=Path)
    parser.add_argument("--count", type=int, default=64)
    args = parser.parse_args()
    if args.count != 64:
        parser.error("the v1 pilot is frozen at exactly 64 independent source records")
    if _sha256(args.source) != args.source_sha256.lower():
        parser.error("source SHA-256 does not match")

    try:
        import h5py
        import numpy as np
        from rdkit import Chem
    except ImportError as exc:
        parser.error(f"SPICE audit requires h5py, numpy, and rdkit: {exc}")

    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    unit_contract = {
        "positions": "nanometer",
        "dft_total_energy": "kilojoule_per_mole",
        "dft_total_force": "kilojoule_per_mole / nanometer",
        "total_charge": "elementary_charge",
    }
    with h5py.File(args.source, "r") as handle:
        total_records = len(handle)
        total_configs = 0
        for record_id in sorted(handle.keys()):
            group = handle[record_id]
            n_configs = int(group["n_configs"][()])
            total_configs += n_configs
            observed_units = {name: _unit(group[name]) for name in unit_contract}
            if observed_units != unit_contract:
                rejected.append({"record_id": record_id, "reason": "unit_contract", "observed": observed_units})
                continue
            numbers = np.asarray(group["atomic_numbers"]).reshape(-1).astype(int)
            charges = np.asarray(group["total_charge"]).reshape(-1).astype(int)
            smiles = _text(group["canonical_isomeric_explicit_hydrogen_mapped_smiles"][()])
            source = _text(group["source"][()])
            mol = Chem.MolFromSmiles(smiles)
            radical_electrons = None if mol is None else sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
            reasons: list[str] = []
            if not set(numbers.tolist()).issubset(ALLOWED_Z):
                reasons.append("outside_CHNOS")
            if len(set(charges.tolist())) != 1:
                reasons.append("charge_varies_across_configs")
                charge = None
            else:
                charge = int(charges[0])
                if charge not in {-1, 0, 1}:
                    reasons.append("charge_outside_pilot_scope")
            if mol is None:
                reasons.append("smiles_parse_failure")
            elif radical_electrons != 0:
                reasons.append("radical_smiles")
            if charge is not None and (int(numbers.sum()) - charge) % 2:
                reasons.append("odd_electron_count")
            if reasons:
                rejected.append({"record_id": record_id, "reason": ",".join(reasons)})
                continue
            candidates.append(
                {
                    "record_id": record_id,
                    "source": source,
                    "charge": charge,
                    "contains_sulfur": 16 in numbers,
                    "atomic_numbers": numbers.tolist(),
                    "symbols": [SYMBOL_BY_Z[int(number)] for number in numbers],
                    "smiles": smiles,
                    "n_configs": n_configs,
                }
            )

        charged = [row for row in candidates if row["charge"] != 0]
        sulfur = [row for row in candidates if row["contains_sulfur"] and row not in charged]
        ordinary = [row for row in candidates if row not in charged and row not in sulfur]
        selected = (sorted(charged, key=lambda row: row["record_id"]) +
                    sorted(sulfur, key=lambda row: row["record_id"]) +
                    sorted(ordinary, key=lambda row: row["record_id"]))[:args.count]
        if len(selected) != args.count:
            raise SystemExit(f"only {len(selected)} eligible independent records; expected {args.count}")
        splits = _split_map([row["record_id"] for row in selected])

        output_rows = []
        for row in selected:
            group = handle[row["record_id"]]
            config_index = 0
            positions_nm = np.asarray(group["positions"][config_index], dtype=float)
            energy_kj_mol = float(np.asarray(group["dft_total_energy"][config_index]).reshape(-1)[0])
            force_kj_mol_nm = np.asarray(group["dft_total_force"][config_index], dtype=float)
            output_rows.append(
                {
                    "schema": "xtbflow-spice2-openff-pilot-record/v1",
                    "source_record_id": f"spice2-openff:{row['record_id']}:config-{config_index}",
                    "parent_record_id": row["record_id"],
                    "config_index": config_index,
                    "split": splits[row["record_id"]],
                    "source_collection": row["source"],
                    "source_file_sha256": args.source_sha256.lower(),
                    "smiles": row["smiles"],
                    "closed_shell_evidence": {
                        "rdkit_radical_electrons": 0,
                        "electron_count_parity": "even",
                        "multiplicity": 1,
                        "basis": "explicit-H mapped SMILES has no radicals; even electron count; SPICE QCSchema workflow uses explicit molecular multiplicity and singlet closed-shell inputs",
                    },
                    "system": {
                        "symbols": row["symbols"],
                        "coordinates": (positions_nm * ANGSTROM_PER_NM).tolist(),
                        "charge": row["charge"],
                        "multiplicity": 1,
                        "environment": {
                            "periodic": False,
                            "source_dataset": "SPICE2 OpenFF",
                            "source_record_id": row["record_id"],
                            "source_config_index": config_index,
                        },
                        "system_id": f"spice2-openff:{row['record_id']}:config-{config_index}",
                    },
                    "reference_protocol_id": REFERENCE_PROTOCOL_ID,
                    "reference": {
                        "energy_hartree": energy_kj_mol / HARTREE_KJ_PER_MOL,
                        "forces_hartree_per_angstrom": (force_kj_mol_nm / (HARTREE_KJ_PER_MOL * ANGSTROM_PER_NM)).tolist(),
                    },
                    "source_units": unit_contract,
                    "canonical_units": {
                        "coordinate": "angstrom",
                        "energy": "hartree",
                        "force": "hartree/angstrom",
                    },
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    if (args.public_records_base is None) != (args.public_records_output is None):
        parser.error("--public-records-base and --public-records-output must be provided together")
    if args.public_records_output is not None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from xtbflow.data.records import PublicRecord, canonical_hash, load_jsonl, write_jsonl

        historical = [
            record for record in load_jsonl(args.public_records_base)
            if record.source_dataset != "spice2_openff_v1.1_pilot"
        ]
        public_rows = []
        for row in output_rows:
            system = row["system"]
            locator = f"{args.source.name}#/{row['parent_record_id']}/positions/{row['config_index']}"
            public_rows.append(PublicRecord(
                source_dataset="spice2_openff_v1.1_pilot",
                source_revision=args.source_sha256.lower(),
                record_id=row["source_record_id"],
                parent_reaction_id=row["parent_record_id"],
                split_group=row["parent_record_id"],
                geometry_locator=locator,
                geometry_hash=canonical_hash({"symbols": system["symbols"], "coordinates": system["coordinates"]}),
                geometry_origin="published_spice2_openff_configuration",
                reference_protocol_id=row["reference_protocol_id"],
                units={"coordinates": "angstrom", "energy": "hartree", "forces": "hartree/angstrom"},
                available_label_mask={"energy": True, "forces": True, "product": False, "ts_geometry": False},
                charge_spin_evidence="mapped_explicit_h_smiles_zero_radicals_even_electron_qcschema_singlet_workflow",
                license_record="SPICE/curated SPICE2 OpenFF source distributed under CC0-1.0; source archive hash pinned",
                pretraining_overlap_audit="not_applicable",
                record_status="observed_reference_energy_forces",
                admission=row["split"],
                input_data={"symbols": system["symbols"], "reactant_coordinates": system["coordinates"], "charge": system["charge"], "multiplicity": system["multiplicity"], "reactant_geometry_locator": locator},
                label_data={"energy": row["reference"]["energy_hartree"], "forces": row["reference"]["forces_hartree_per_angstrom"]},
                claim_limit="64-parent-record development E/F pilot only; not an independent reaction-discovery or final paper test set",
            ))
        args.public_records_output.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.public_records_output, [*historical, *public_rows])
    evidence = {
        "schema": "xtbflow-spice2-openff-admission/v1",
        "source": args.source.name,
        "source_sha256": args.source_sha256.lower(),
        "source_reference": "modelforge curated SPICE2 OpenFF v1.1 ntc_1000_HCNOFClS",
        "reference_method": "B3LYP-D3BJ/DZVP",
        "source_units": unit_contract,
        "canonical_conversion": {
            "hartree_kj_per_mol": HARTREE_KJ_PER_MOL,
            "angstrom_per_nm": ANGSTROM_PER_NM,
        },
        "observed_source_records": total_records,
        "observed_source_configs": total_configs,
        "eligible_records": len(candidates),
        "rejected_records": len(rejected),
        "selected_records": len(output_rows),
        "selected_configs": len(output_rows),
        "selection_policy": "all eligible charged records first, then sulfur-bearing records, then remaining records; lexical record-id order within strata; config_index=0; no result-based selection",
        "split_seed": SPLIT_SEED,
        "split_ratios": SPLIT_RATIOS,
        "split_contract": "same SHA-256 seed/group bucket rule as xtbflow.data.splits.assign_group_splits",
        "split_counts": {name: sum(row["split"] == name for row in output_rows) for name in ("train", "validation", "test")},
        "charge_counts": {str(charge): sum(row["system"]["charge"] == charge for row in output_rows) for charge in (-1, 0, 1)},
        "sulfur_records": sum("S" in row["system"]["symbols"] for row in output_rows),
        "admission_status": "pilot_admitted",
        "limits": [
            "This is a 64-independent-record pilot, one configuration per parent record.",
            "Records with varying charge, charge outside -1/0/+1, radicals, odd electron count, or elements outside H/C/N/O/S are excluded.",
            "Multiplicity=1 is admitted only for radical-free explicit-H SMILES with even electron count under the documented SPICE/QCSchema closed-shell workflow.",
            "Source labels are reference E/F; no new DFT is generated.",
        ],
        "rejections": rejected,
    }
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: evidence[key] for key in ("eligible_records", "selected_records", "split_counts", "charge_counts", "sulfur_records")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
