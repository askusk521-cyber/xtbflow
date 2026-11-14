#!/usr/bin/env python3
"""Prepare an immutable n2-only first-wave task manifest from completed minima."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import socket

import numpy as np
from gqc_tautomer_pilot import digest, write_json


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if socket.gethostname() != "node2":
        raise RuntimeError("source coordinates are read only on n2")
    a.out.mkdir(parents=True, exist_ok=False)
    verified = json.loads((a.pilot / "verify/report.json").read_text())
    minima = json.loads((a.pilot / "minima/report.json").read_text())
    if digest(a.pilot / "verify/report.json") != minima["verified_sha256"]:
        raise ValueError("pilot identity mismatch")
    rows = {r["start_id"]: r for r in verified["rows"]}
    selected = [rows[minima["selected_start_ids"][k]] for k in minima["selected_start_ids"] if not k.startswith("control_")]
    points, tasks = {}, []
    for row in selected:
        pid = row["id"]
        points[pid] = {k: row[k] for k in ("id", "group", "form", "formula", "stereo_smiles", "charge", "multiplicity", "symbols", "coordinates_angstrom")}
        x = np.asarray(row["coordinates_angstrom"])
        direction = np.sin(np.arange(x.size) + 0.17).reshape(x.shape)
        direction -= direction.mean(axis=0)
        direction /= np.linalg.norm(direction)
        points[pid + "_distorted"] = {**points[pid], "id": pid + "_distorted", "coordinates_angstrom": (x + 0.08 * direction).tolist()}
        for suffix, basis in [("_distorted", "def2-svp"), ("", "def2-tzvp")]:
            tasks.append({"id": f"audit_{pid}_{basis}", "kind": "audit", "point": pid + suffix, "method": "b3lyp-rks", "xc": "b3lyp", "basis": basis, "gpu": 1, "cpus": 4, "mem_gib": 16, "minutes": 25, "calls": 8})
        if pid in ("R6_enol", "R6_carbonyl"):
            tasks.append({"id": "audit_m06_" + pid, "kind": "audit", "point": pid, "method": "m06-2x-rks", "xc": "m06-2x", "basis": "def2-svp", "gpu": 1, "cpus": 4, "mem_gib": 16, "minutes": 25, "calls": 8})
        tasks.append({"id": "ccsdt_" + pid, "kind": "ccsd_t", "point": pid, "basis": "cc-pvdz", "gpu": 0, "cpus": 8, "mem_gib": 32, "minutes": 120, "calls": 5})
        for solvent, epsilon in (("water", 78.3553), ("acetonitrile", 35.688)):
            tasks.append({"id": f"pcm_{solvent}_{pid}", "kind": "solvent", "point": pid, "xc": "b3lyp", "basis": "def2-tzvp", "solvent": solvent, "epsilon": epsilon, "gpu": 0, "cpus": 4, "mem_gib": 16, "minutes": 40, "calls": 2})
    manifest = {"schema": "xtbflow-qc-tasks/v1", "config": json.loads(a.config.read_text()), "config_sha256": digest(a.config),
                "source_verified_sha256": digest(a.pilot / "verify/report.json"), "source_minima_sha256": digest(a.pilot / "minima/report.json"),
                "points": points, "tasks": tasks, "input_scope": "exposed numerical/chemical development cohort"}
    write_json(a.out / "tasks.json", manifest)
    summary = {"task_count": len(tasks), "gpu_tasks": sum(t["gpu"] for t in tasks), "cpu_tasks": sum(not t["gpu"] for t in tasks),
               "gpu_hour_ceiling": sum(t["gpu"] * t["minutes"] / 60 for t in tasks),
               "cpu_core_hour_ceiling": sum(t["cpus"] * t["minutes"] / 60 for t in tasks),
               "operation_ceiling": sum(t["calls"] for t in tasks), "manifest_sha256": digest(a.out / "tasks.json")}
    write_json(a.out / "summary.json", summary)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
