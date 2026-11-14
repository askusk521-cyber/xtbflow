#!/usr/bin/env python3
"""Register four new cc-pVTZ references after the observed R6 basis sensitivity."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import socket
import subprocess
from gqc_tautomer_pilot import ROOT, digest, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    a = ap.parse_args()
    if socket.gethostname() != "node2":
        raise RuntimeError("source coordinates remain on n2")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("preparation source must be clean")
    one = json.loads((a.root / "wave1/tasks.json").read_text())
    two = json.loads((a.root / "wave2/tasks.json").read_text())
    evidence = [a.root / "results-wave2" / ("ccsdt_tz_R6_" + form) / "report.json" for form in ("enol", "carbonyl")]
    if any(json.loads(p.read_text())["status"] != "completed_finite_basis_reference" for p in evidence):
        raise RuntimeError("basis followup requires completed R6 references")
    points = {pid: p for pid, p in one["points"].items() if pid in ("R30_enol", "R30_carbonyl", "R51_enol", "R51_carbonyl")}
    if len(points) != 4:
        raise ValueError("fixed source cohort incomplete")
    tasks = [{"id": "ccsdt_tz_" + pid, "point": pid, "kind": "ccsd_t", "basis": "cc-pvtz",
              "gpu": 0, "cpus": 8, "mem_gib": 48, "minutes": 105, "calls": 5} for pid in points]
    all_tasks = one["tasks"] + two["tasks"] + tasks
    ceilings = {"gpu_allocation_hours_ceiling": sum(t["gpu"] * t["minutes"] / 60 for t in all_tasks),
                "cpu_core_hours_ceiling": sum(t["cpus"] * t["minutes"] / 60 for t in all_tasks),
                "quantum_operations_ceiling": sum(t["calls"] for t in all_tasks)}
    if any(v > one["config"]["resources"][k] for k, v in ceilings.items()):
        raise ValueError("followup exceeds total authorized ceiling")
    out = a.root / "wave3"
    out.mkdir(exist_ok=False)
    pack = {"schema": "xtbflow-qc-tasks/v1", "config": one["config"], "config_sha256": one["config_sha256"],
            "preparation_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "preparation_script_sha256": digest(Path(__file__)), "wave1_manifest_sha256": digest(a.root / "wave1/tasks.json"),
            "wave2_manifest_sha256": digest(a.root / "wave2/tasks.json"), "trigger_report_sha256": [digest(p) for p in evidence],
            "input_scope": "same exposed old geometries, four new larger-basis calculations; not retries",
            "points": points, "tasks": tasks}
    write_json(out / "tasks.json", pack)
    summary = {"new_task_count": 4, "total_task_count": len(all_tasks), "manifest_sha256": digest(out / "tasks.json"),
               "source_commit": pack["preparation_source_commit"], "aggregate_ceilings": ceilings,
               "new_cpu_core_hour_ceiling": 56, "new_quantum_operation_ceiling": 20,
               "trigger_report_sha256": pack["trigger_report_sha256"]}
    write_json(out / "summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
