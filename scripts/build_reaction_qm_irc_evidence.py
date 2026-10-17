#!/usr/bin/env python3
"""Add IRC evidence to the Reaction-QM task funnel (issues #76 / #94).

Second stage of ``build_reaction_qm_task_funnel.py``: it reads the funnel's
``manifest.jsonl`` (hash-bound in the output), then for every *identity
verifiable* record opens the IRC path in ``B3LYPD3_TZVP_IRC.h5`` and checks

* atom order and TS frame against the main HDF5;
* energy/force mutual consistency from the work identity;
* whether the two IRC ends reproduce the formed/broken sigma bonds named by the
  event (step-pairing evidence).

Read-only; no quantum-chemistry call, no GPU, no download.  Outputs go to
``--output-dir`` (a run directory, never the repository).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import rdkit

from xtbflow.data.reaction_qm_funnel import SOURCE_REVISION, bond_table, split_side
from xtbflow.data.reaction_qm_irc import (
    HARTREE_TO_EV,
    IrcInput,
    analyse_irc,
    build_irc_report,
    result_row,
)

EXPECTED_MD5 = "782a4e5e8099de8f2b8e0e90128028cb"  # Zenodo-published checksum of the IRC file


def _hashes(path: Path, chunk: int = 16 * 1024 * 1024) -> tuple[str, str]:
    sha, md5 = hashlib.sha256(), hashlib.md5()  # noqa: S324 - md5 only to match Zenodo's published checksum
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            sha.update(block)
            md5.update(block)
    return sha.hexdigest(), md5.hexdigest()


def _chunk_index(handle: h5py.File) -> dict[str, str]:
    """record id -> chunk group; empty if the file is flat."""

    index: dict[str, str] = {}
    for key in handle.keys():
        item = handle[key]
        if isinstance(item, h5py.Group) and not key.startswith("RXN_"):
            for rid in item.keys():
                index[rid] = key
    return index


def _group(handle: h5py.File, index: dict[str, str], rid: str):
    if rid in handle and "atomic_numbers" in handle[rid]:
        return handle[rid]
    chunk = index.get(rid)
    return handle[chunk][rid] if chunk and rid in handle[chunk] else None


def _sigma_edits(ts_smiles: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Formed / broken sigma bonds (0 -> >0, >0 -> 0) from the TS reaction string."""

    reactant_side, product_side = ts_smiles.split(">>")
    r, p = bond_table(split_side(reactant_side)), bond_table(split_side(product_side))
    formed = [k for k in sorted(set(r) | set(p)) if r.get(k, 0.0) == 0.0 and p.get(k, 0.0) > 0.0]
    broken = [k for k in sorted(set(r) | set(p)) if r.get(k, 0.0) > 0.0 and p.get(k, 0.0) == 0.0]
    return formed, broken


def _work(args: tuple[str, str, list[str]]) -> list[dict[str, Any]]:
    main_path, irc_path, ids = args
    rows: list[dict[str, Any]] = []
    with h5py.File(main_path, "r") as main, h5py.File(irc_path, "r") as irc_file:
        main_index, irc_index = _chunk_index(main), _chunk_index(irc_file)
        for rid in ids:
            ts_group = main[main_index[rid]][rid]["TS"]
            ts_smiles = ts_group["smiles"][()]
            ts_smiles = ts_smiles.decode("utf-8") if isinstance(ts_smiles, bytes) else str(ts_smiles)
            formed, broken = _sigma_edits(ts_smiles)
            group = _group(irc_file, irc_index, rid)
            irc = None
            if group is not None:
                irc = IrcInput(
                    atomic_numbers=tuple(int(z) for z in np.asarray(group["atomic_numbers"][()]).ravel()),
                    coordinates=np.asarray(group["coordinates"][()], dtype=np.float64),
                    energies=np.asarray(group["energies"][()], dtype=np.float64).ravel(),
                    forces=np.asarray(group["forces"][()], dtype=np.float64),
                )
            result = analyse_irc(
                rid,
                irc,
                ts_atomic_numbers=tuple(int(z) for z in np.asarray(ts_group["atomic_numbers"][()]).ravel()),
                ts_coordinates=np.asarray(ts_group["coordinates"][()], dtype=np.float64),
                ts_energy_hartree=float(np.asarray(ts_group["EHG"][()]).ravel()[0]),
                formed=formed,
                broken=broken,
            )
            rows.append(result_row(result))
    return rows


def _git_commit(repo: Path) -> dict[str, Any]:
    import subprocess

    try:
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:  # noqa: BLE001
        return {"commit": None, "dirty": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True, help="main B3LYPD3_TZVP.h5")
    parser.add_argument("--irc-h5", type=Path, required=True)
    parser.add_argument("--funnel-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N verified ids (smoke runs)")
    parser.add_argument("--verify-hashes", action="store_true", help="sha256 + md5 of the IRC file (md5 must match Zenodo)")
    args = parser.parse_args()
    if args.workers <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--workers and --limit must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sha = md5 = None
    if args.verify_hashes:
        sha, md5 = _hashes(args.irc_h5)
        if md5 != EXPECTED_MD5:
            print(f"IRC md5 mismatch: {md5}", file=sys.stderr)
            return 2

    manifest_rows = [json.loads(line) for line in args.funnel_manifest.open(encoding="utf-8")]
    verified = [r for r in manifest_rows if r["identity_verified"]]
    if args.limit is not None:
        verified = verified[: args.limit]
    ids = [r["record_id"] for r in verified]
    groups = {r["record_id"]: r["parent_group"] for r in manifest_rows if r.get("parent_group")}
    event_ok = {r["record_id"] for r in verified if r["tasks"]["event_only"]}

    slices = [ids[i :: args.workers] for i in range(args.workers)]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for part in pool.map(_work, [(str(args.h5), str(args.irc_h5), s) for s in slices if s]):
            rows.extend(part)
    order = {rid: k for k, rid in enumerate(ids)}
    rows.sort(key=lambda r: order[r["record_id"]])
    for row in rows:
        row["parent_group"] = groups.get(row["record_id"])
        row["event_only"] = row["record_id"] in event_ok

    manifest = args.output_dir / "irc_manifest.jsonl"
    with manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    lists = {
        "energy_force": lambda r: r["irc_energy_force_ok"],
        "step_pairing_by_irc": lambda r: r["irc_path_pairing_ok"],
        "path_paired_joint": lambda r: r["irc_path_pairing_ok"] and r["event_only"],
    }
    list_hashes: dict[str, dict[str, Any]] = {}
    list_dir = args.output_dir / "task_ids"
    list_dir.mkdir()
    for name, pick in lists.items():
        chosen = [r["record_id"] for r in rows if pick(r)]
        path = list_dir / f"{name}.txt"
        path.write_text("".join(i + "\n" for i in chosen), encoding="utf-8", newline="\n")
        list_hashes[name] = {"records": len(chosen), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    base = {}
    funnel_report = args.funnel_manifest.parent / "funnel_report.json"
    if funnel_report.is_file():
        base = {k: v for k, v in json.loads(funnel_report.read_text(encoding="utf-8"))["tasks"].items()}
    report = build_irc_report(rows, groups=groups, event_ok_ids=event_ok, base_tasks=base)
    report["run_scope"] = {
        "verified_records_in_funnel": sum(1 for r in manifest_rows if r["identity_verified"]),
        "records_examined": len(rows),
        "smoke_limit": args.limit,
        "complete": args.limit is None,
    }
    report["inputs"] = {
        "funnel_manifest_sha256": hashlib.sha256(args.funnel_manifest.read_bytes()).hexdigest(),
        "irc_file": args.irc_h5.name,
        "irc_sha256": sha,
        "irc_md5": md5,
        "irc_md5_expected_from_zenodo": EXPECTED_MD5,
        "source_revision": SOURCE_REVISION,
    }
    report["irc_manifest"] = {"file": "irc_manifest.jsonl", "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "rows": len(rows)}
    report["task_id_lists"] = list_hashes
    report["identities"] = {
        "rdkit": rdkit.__version__,
        "python": sys.version.split()[0],
        "code": _git_commit(Path(__file__).resolve().parents[1]),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hartree_to_ev": HARTREE_TO_EV,
    }
    (args.output_dir / "irc_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "irc_present": report["irc_present"],
        "energy_force": {k: v for k, v in report["energy_force"].items() if k != "unit_consistency"},
        "unit_consistency": report["energy_force"]["unit_consistency"],
        "step_pairing_by_irc": report["step_pairing_by_irc"],
        "path_paired_joint": report["path_paired_joint"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
