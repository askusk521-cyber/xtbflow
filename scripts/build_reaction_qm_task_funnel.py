#!/usr/bin/env python3
"""Build the Reaction-QM task admission funnel (issues #76 / #94).

Read-only on the pinned HDF5.  It performs no quantum-chemistry call, no
download and no GPU work.  Outputs go to ``--output-dir`` (a run directory,
never the repository):

* ``manifest.jsonl``   one compact row per record (flags, reasons, group id);
* ``funnel_report.json`` aggregate counts (records *and* parent groups), reasons
  with next actions, mapping statistics and claim limits;
* ``task_ids/<task>.txt`` record ids available for each task (for lane 2);
* ``run.json``         code/asset/library identities so the run can be replayed.

Only the aggregate report is meant to be committed; the per-record manifest is
hash-bound in the report so it can be re-verified without publishing it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import rdkit

from xtbflow.data.reaction_qm_funnel import (
    GROUPING_RULE,
    SCHEMA,
    SOURCE_REVISION,
    TASKS,
    ReactionInput,
    SpeciesInput,
    analyse_reaction,
    assign_parent_groups,
    build_report,
    record_row,
)

EXPECTED_H5_SHA256 = "3d0fc655819a9a2747f554a9025cd36cbdffd1175c4a1d40934b6fe5530af82a"
_RXN_NUMBER = re.compile(r"(?:RXN_)?(\d+)$")


def _sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(item: h5py.Group, key: str) -> int | None:
    return int(item[key][()]) if key in item else None


def species_from_h5(name: str, item: h5py.Group) -> SpeciesInput | None:
    """Read one species group; ``None`` if a mandatory field is absent."""

    for key in ("smiles", "atomic_numbers", "coordinates"):
        if key not in item:
            return None
    smiles = item["smiles"][()]
    smiles = smiles.decode("utf-8") if isinstance(smiles, bytes) else str(smiles)
    coordinates = np.asarray(item["coordinates"][()])
    energies = tuple(float(x) for x in np.asarray(item["EHG"][()]).ravel()) if "EHG" in item else ()
    return SpeciesInput(
        name=name,
        smiles=smiles,
        atomic_numbers=tuple(int(z) for z in np.asarray(item["atomic_numbers"][()]).ravel()),
        charge=_scalar(item, "charge"),
        multiplicity=_scalar(item, "multiplicity"),
        energies=energies,
        coordinate_shape=tuple(int(x) for x in coordinates.shape),
        coordinates_finite=bool(np.isfinite(coordinates).all()),
    )


def _process_chunk(args: tuple[str, str, int | None]) -> list[dict[str, Any]]:
    path, chunk_name, limit = args
    rows: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        group = handle[chunk_name]
        names = sorted(group.keys())
        if limit is not None:
            names = names[:limit]
        for record_id in names:
            species: dict[str, SpeciesInput] = {}
            missing = False
            for name in group[record_id].keys():
                parsed = species_from_h5(name, group[record_id][name])
                if parsed is None:
                    missing = True
                    break
                species[name] = parsed
            if missing:
                species = {}
            result = analyse_reaction(ReactionInput(record_id, species))
            rows.append(record_row(result))
    return rows


def _reaction_sort_key(name: str) -> tuple[int, str]:
    match = _RXN_NUMBER.fullmatch(name)
    return (int(match.group(1)), name) if match else (2**63 - 1, name)


def _official_split_crosscheck(splits_dir: Path, rows: list[dict[str, Any]], groups: dict[str, str]) -> dict[str, Any]:
    """How often does the project-derived parent group straddle the official
    reaction-ID partition?  Informational: the official split is kept separate
    and is not used to define groups."""

    assignment: dict[str, str] = {}
    for label, filename in (("train", "B3LYP-RXN_train.csv"), ("valid", "B3LYP-RXN_valid.csv"), ("test", "B3LYP-RXN_test.csv")):
        path = splits_dir / filename
        if not path.is_file():
            return {"available": False, "reason": f"missing {filename}"}
        with path.open(encoding="utf-8") as handle:
            next(handle)
            for line in handle:
                assignment[line.split(",", 1)[0]] = label
    by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        gid, split = groups.get(row["record_id"]), assignment.get(row["record_id"])
        if gid and split and row["identity_verified"]:
            by_group[gid].add(split)
    straddling = sum(1 for splits in by_group.values() if len(splits) > 1)
    return {
        "available": True,
        "records_in_official_split": len(assignment),
        "verified_scope_groups": len(by_group),
        "verified_scope_groups_spanning_multiple_official_partitions": straddling,
        "note": "Official partition is by reaction ID only; groups spanning it indicate that ID-level splits can leak a parent system.",
    }


def _git_commit(repo: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:  # noqa: BLE001 - provenance is best effort, never fatal
        return {"commit": None, "dirty": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-per-chunk", type=int, default=None, help="Process only the first N records of every HDF5 chunk (smoke runs)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--official-splits-dir", type=Path, default=None)
    parser.add_argument("--verify-sha256", action="store_true")
    args = parser.parse_args()
    if args.limit_per_chunk is not None and args.limit_per_chunk <= 0:
        parser.error("--limit-per-chunk must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    out_dir = args.output_dir
    if out_dir.exists() and any(out_dir.iterdir()):
        parser.error(f"refusing to overwrite non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    asset_sha = _sha256(args.h5) if args.verify_sha256 else None
    if asset_sha is not None and asset_sha != EXPECTED_H5_SHA256:
        print(f"asset hash mismatch: {asset_sha}", file=sys.stderr)
        return 2

    with h5py.File(args.h5, "r") as handle:
        chunks = sorted(handle.keys())
        total_in_file = sum(len(handle[c]) for c in chunks)
    jobs = [(str(args.h5), chunk, args.limit_per_chunk) for chunk in chunks]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for part in pool.map(_process_chunk, jobs):
            rows.extend(part)
    rows.sort(key=lambda r: _reaction_sort_key(r["record_id"]))

    groups = assign_parent_groups(rows)
    for row in rows:
        row["parent_group"] = groups.get(row["record_id"])

    manifest = out_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    task_dir = out_dir / "task_ids"
    task_dir.mkdir()
    task_hashes: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        ids = [r["record_id"] for r in rows if r["tasks"].get(task)]
        path = task_dir / f"{task}.txt"
        path.write_text("".join(i + "\n" for i in ids), encoding="utf-8", newline="\n")
        task_hashes[task] = {"records": len(ids), "sha256": _sha256(path)}

    report = build_report(rows, groups, source_records=len(rows))
    report["run_scope"] = {
        "records_in_hdf5": total_in_file,
        "records_processed": len(rows),
        "smoke_limit_per_chunk": args.limit_per_chunk,
        "complete": args.limit_per_chunk is None and len(rows) == total_in_file,
    }
    report["manifest"] = {"file": "manifest.jsonl", "sha256": _sha256(manifest), "rows": len(rows)}
    report["task_id_lists"] = task_hashes
    if args.official_splits_dir is not None:
        report["official_split_crosscheck"] = _official_split_crosscheck(args.official_splits_dir, rows, groups)
    report["identities"] = {
        "source_revision": SOURCE_REVISION,
        "h5_sha256_expected": EXPECTED_H5_SHA256,
        "h5_sha256_computed": asset_sha,
        "grouping_rule": GROUPING_RULE,
        "rdkit": rdkit.__version__,
        "python": sys.version.split()[0],
        "code": _git_commit(Path(__file__).resolve().parents[1]),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "funnel_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "run.json").write_text(
        json.dumps({"schema": SCHEMA, "argv": sys.argv[1:], **report["identities"], **report["run_scope"]}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {s["stage"]: (s["records"], s["parent_groups"]) for s in report["funnel"]}
    summary.update({f"task:{k}": (v["records"], v["parent_groups"]) for k, v in report["tasks"].items()})
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
