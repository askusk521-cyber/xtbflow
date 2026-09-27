#!/usr/bin/env python3
"""Audit a bounded UniTS-Lib slice without changing or reserializing labels.

The published ``UniTS_Lib.npy`` object array contains useful coordinates and
labels, but does not embed a complete per-record calculator protocol.  This
command therefore separates three questions that are often conflated:

* are the record fields structurally readable and finite;
* can the source and reported method be traced to fixed artefacts; and
* are coordinate, energy, and force units explicit enough for training.

The output is a compact JSON evidence file.  It never rewrites the source
array, infers missing charge/spin values, or promotes a record to the public
training manifest.  Unknown units deliberately keep the decision at
``diagnostic_only`` even when every sampled row passes the structural checks.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ATOMIC_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _symbols(atoms: Any) -> tuple[str, ...]:
    values = tuple(int(value) for value in atoms)
    if any(value < 1 or value > len(ATOMIC_SYMBOLS) for value in values):
        raise ValueError("atoms contains an unsupported atomic number")
    return tuple(ATOMIC_SYMBOLS[value - 1] for value in values)


def _stats(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    numeric = np.issubdtype(array.dtype, np.number)
    result: dict[str, Any] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    if numeric and array.size:
        finite = np.isfinite(array)
        result.update(
            {
                "finite": bool(finite.all()),
                "min": float(np.nanmin(array)),
                "max": float(np.nanmax(array)),
                "max_abs": float(np.nanmax(np.abs(array))),
            }
        )
    else:
        result["finite"] = False
    return result


def _input_hash(symbols: tuple[str, ...], coordinates: np.ndarray, charge: int, multiplicity: int) -> str:
    payload = {
        "symbols": list(symbols),
        "coordinates": [[float(component) for component in row] for row in coordinates],
        "charge": charge,
        "multiplicity": multiplicity,
        "environment": {},
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _graph_checks(record: Mapping[str, Any], atom_count: int, charge: int, multiplicity: int) -> dict[str, Any]:
    checks: dict[str, Any] = {"present": "graph_feat" in record}
    if "graph_feat" not in record:
        return checks
    try:
        features = record["graph_feat"]
        node_attr = np.asarray(features[0])
        edge_index = np.asarray(features[1])
        edge_attr = np.asarray(features[2])
        checks.update(
            {
                "node_shape": list(node_attr.shape),
                "edge_index_shape": list(edge_index.shape),
                "edge_attr_shape": list(edge_attr.shape),
                "node_shape_ok": bool(node_attr.shape == (atom_count, 10)),
                "edge_index_bounds_ok": bool(
                    edge_index.size == 0
                    or (np.isfinite(edge_index).all() and edge_index.min() >= 0 and edge_index.max() < atom_count)
                ),
                # UniTS stores global charge and multiplicity in every node
                # row.  Check the published encoding; never repair it here.
                "charge_column_consistent": bool(node_attr.shape == (atom_count, 10) and np.all(node_attr[:, 2] == charge + 3)),
                "multiplicity_column_consistent": bool(node_attr.shape == (atom_count, 10) and np.all(node_attr[:, 9] == multiplicity - 1)),
            }
        )
    except (IndexError, TypeError, ValueError) as exc:
        checks["error"] = f"{type(exc).__name__}: {exc}"
    return checks


def _audit_row(record: Mapping[str, Any], index: int, source_sha256: str) -> dict[str, Any]:
    required = {"atoms", "inpt_orien", "forces", "energy", "chrg", "mult", "rct_idx"}
    missing = sorted(required - set(record))
    row: dict[str, Any] = {"record_index": index, "record_id": f"units-lib:{source_sha256[:16]}:row-{index}"}
    if missing:
        row.update({"status": "invalid", "missing_fields": missing})
        return row
    try:
        symbols = _symbols(record["atoms"])
        coordinates = np.asarray(record["inpt_orien"], dtype=np.float64)
        forces = np.asarray(record["forces"], dtype=np.float64)
        energy = float(record["energy"])
        charge = int(record["chrg"])
        multiplicity = int(record["mult"])
        reactive = tuple(int(value) for value in record["rct_idx"])
        coordinates_ok = coordinates.shape == (len(symbols), 3) and bool(np.isfinite(coordinates).all())
        forces_ok = forces.shape == (len(symbols), 3) and bool(np.isfinite(forces).all())
        energy_ok = bool(np.isfinite(energy))
        state_ok = type(record["chrg"]) is int and type(record["mult"]) is int and multiplicity >= 1
        reactive_ok = len(set(reactive)) == len(reactive) and all(0 <= value < len(symbols) for value in reactive)
        graph = _graph_checks(record, len(symbols), charge, multiplicity)
        try:
            molecule_atoms = int(record["rdmol"].GetNumAtoms())
            mol_count_ok = molecule_atoms == len(symbols)
        except (AttributeError, TypeError, ValueError):
            molecule_atoms, mol_count_ok = None, False
        structure_ok = bool(coordinates_ok and forces_ok and energy_ok and state_ok and reactive_ok and mol_count_ok)
        row.update(
            {
                "status": "structurally_valid" if structure_ok else "invalid",
                "symbols": list(symbols),
                "atom_count": len(symbols),
                "input_hash": _input_hash(symbols, coordinates, charge, multiplicity),
                "charge": charge,
                "multiplicity": multiplicity,
                "reactive_indices_zero_based": list(reactive),
                "rdmol_atom_count": molecule_atoms,
                "checks": {
                    "coordinates": coordinates_ok,
                    "forces": forces_ok,
                    "energy": energy_ok,
                    "charge_and_multiplicity": state_ok,
                    "reactive_indices": reactive_ok,
                    "rdmol_atom_count": mol_count_ok,
                },
                "coordinate_stats": _stats(coordinates),
                "energy": energy,
                "force_stats": _stats(forces),
                "graph_checks": graph,
            }
        )
    except (TypeError, ValueError, OverflowError) as exc:
        row.update({"status": "invalid", "error": f"{type(exc).__name__}: {exc}"})
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="verified UniTS_Lib.npy object array")
    parser.add_argument("--output", required=True, type=Path, help="new JSON evidence file")
    parser.add_argument("--count", type=int, default=64, help="number of contiguous records to inspect")
    parser.add_argument("--start", type=int, default=0, help="first record index")
    parser.add_argument("--source-url", default="https://modelscope.cn/models/XuLiCheng2025/UniTS-Gen-v1")
    parser.add_argument("--source-revision", default="08bd230e0826b6075431b7e123ee07a8e56c32be")
    parser.add_argument("--method-identity", default="B3LYP-D3(BJ)/def2-SVP")
    parser.add_argument("--method-evidence", default="UniTS-Gen paper/SI reports the dataset-level method; raw records do not embed a per-record calculator protocol")
    parser.add_argument("--coordinate-unit", default="unknown")
    parser.add_argument("--energy-unit", default="unknown")
    parser.add_argument("--force-unit", default="unknown")
    args = parser.parse_args()
    if args.count < 1 or args.start < 0:
        raise SystemExit("--count must be positive and --start must be nonnegative")
    if not args.source.is_file():
        raise SystemExit(f"source does not exist: {args.source}")

    source_sha256 = _sha256(args.source)
    dataset = np.load(args.source, allow_pickle=True)
    total = int(len(dataset))
    stop = min(total, args.start + args.count)
    rows = [_audit_row(dataset[index], index, source_sha256) for index in range(args.start, stop)]
    status_counts = Counter(row["status"] for row in rows)
    all_units_explicit = all(value != "unknown" for value in (args.coordinate_unit, args.energy_unit, args.force_unit))
    structural_pass = sum(value == "structurally_valid" for value in status_counts.elements())
    report = {
        "schema": "xtbflow-units-provenance-audit/v1",
        "source": {
            # Keep host-specific mount paths out of committed evidence.  The
            # byte identity, URL and pinned revision are sufficient to recover
            # the source without exposing an execution-machine layout.
            "path": "data/raw/units/UniTS_Lib.npy",
            "size_bytes": args.source.stat().st_size,
            "sha256": source_sha256,
            "url": args.source_url,
            "revision": args.source_revision,
            "total_records": total,
        },
        "slice": {"start": args.start, "count_requested": args.count, "count_observed": len(rows)},
        "method": {
            "identity": args.method_identity,
            "status": "reported_externally_not_embedded",
            "evidence": args.method_evidence,
        },
        "units": {
            "coordinate": {"value": args.coordinate_unit, "status": "explicit" if args.coordinate_unit != "unknown" else "unknown"},
            "energy": {"value": args.energy_unit, "status": "explicit" if args.energy_unit != "unknown" else "unknown"},
            "force": {"value": args.force_unit, "status": "explicit" if args.force_unit != "unknown" else "unknown"},
        },
        "structural_summary": {
            "status_counts": dict(sorted(status_counts.items())),
            "structurally_valid_count": structural_pass,
            "graph_charge_consistent_count": sum(bool(row.get("graph_checks", {}).get("charge_column_consistent")) for row in rows),
            "graph_multiplicity_consistent_count": sum(bool(row.get("graph_checks", {}).get("multiplicity_column_consistent")) for row in rows),
        },
        "decision": {
            "admission": "diagnostic_only" if not all_units_explicit or structural_pass != len(rows) else "candidate_for_quarantine_pairing",
            "public_manifest_update": False,
            "reason": "Per-record energy/force units are not explicit in the raw object array; preserve this slice as evidence until a source-level unit statement is linked to the exact bytes.",
        },
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "total_records": total, "observed": len(rows), "status_counts": dict(sorted(status_counts.items())), "admission": report["decision"]["admission"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
