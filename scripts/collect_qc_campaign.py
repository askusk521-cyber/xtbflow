#!/usr/bin/env python3
"""Read n2 raw task reports and emit a coordinate/path-free campaign summary."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import socket
import time
from gqc_tautomer_pilot import digest, write_json


def collect(root):
    tasks = []
    manifests = {}
    resource = Counter()
    for wave in ("wave1", "wave2", "wave3"):
        path = root / wave / "tasks.json"
        if not path.exists():
            continue
        manifest = json.loads(path.read_text())
        manifests[wave] = {"sha256": digest(path), "config_sha256": manifest["config_sha256"]}
        for index, task in enumerate(manifest["tasks"]):
            entry = {"wave": wave, "index": index, "id": task["id"], "kind": task["kind"], "status": "report_not_present"}
            report_path = root / ("results-" + wave) / task["id"] / "report.json"
            if report_path.exists():
                report = json.loads(report_path.read_text())
                entry.update(report_sha256=digest(report_path))
                for key in ("status", "stage", "source_commit", "slurm_job_id", "host", "python", "environment_versions", "error_type", "resources",
                            "energy_hartree", "reference_energy_hartree", "cross_energy_error_hartree", "cross_force_error_hartree_per_bohr", "energy_pass", "force_pass", "fd",
                            "ccsdt_hartree", "t1_frobenius_norm", "basis", "epsilon", "frequencies_cm1", "max_force_hartree_per_bohr", "harmonic_gibbs_hartree", "identity",
                            "gate_evidence", "upstream", "local_electronic_barrier_kcal_mol", "initial_hessian_probe"):
                    if key in report:
                        entry[key] = report[key]
                ledger = report_path.parent / "operations.jsonl"
                if ledger.exists():
                    counts = Counter(json.loads(line)["status"] for line in ledger.read_text().splitlines())
                    entry["journal"] = dict(counts)
                    entry["journal"]["unsettled"] = counts["started"] - counts["completed"] - counts["failed"]
                    resource.update(counts)
                if "transition_state" in report:
                    entry["ts_frequencies_cm1"] = report["transition_state"]["frequencies_cm1"]
                if "neb" in report:
                    entry["neb_converged"] = report["neb"]["converged"]
            tasks.append(entry)
    results = {t["id"]: t for t in tasks}
    pairs = []
    for group in ("R6", "R30", "R51"):
        pair = {"group": group, "sign": "enol minus the previously selected carbonyl; positive favors that carbonyl"}
        for prefix, field, output in [("ccsdt_", "ccsdt_hartree", "ccsdt_ccpvdz"), ("ccsdt_tz_", "ccsdt_hartree", "ccsdt_ccpvtz"),
                                      ("pcm_water_", "energy_hartree", "b3lyp_tz_water"), ("pcm_acetonitrile_", "energy_hartree", "b3lyp_tz_acetonitrile"),
                                      ("gas_tz_", "energy_hartree", "b3lyp_tz_gas")]:
            r, p = results.get(prefix + group + "_enol", {}), results.get(prefix + group + "_carbonyl", {})
            if field in r and field in p:
                pair[output + "_delta_e_kcal_mol"] = (r[field] - p[field]) * 627.509474
        pairs.append(pair)
    return {"schema": "xtbflow-qc-campaign-summary/v1", "host": socket.gethostname(), "collected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "manifests": manifests, "status_counts": dict(Counter(t["status"] for t in tasks)), "tasks": tasks, "pairs": pairs,
            "quantum_journal_counts": dict(resource), "allocated_gpu_seconds": "unknown: Slurm accounting unavailable",
            "claim_limits": ["Read per-task status; queue disappearance is not success.", "No coordinates, weights, source raw bytes or private absolute paths are exported.",
                             "GQC agreement is scoped to exact tested native build/method/basis; CCSD(T) remains a finite-basis reference.",
                             "New path hypotheses require a qualified saddle and bidirectional verified endpoints; a NEB maximum alone is not a reaction."]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if socket.gethostname() != "node2":
        raise RuntimeError("raw task results are read only on n2")
    if a.out.exists():
        raise FileExistsError("do not replace prior evidence")
    summary = collect(a.root)
    write_json(a.out, summary)
    print(json.dumps({k: summary[k] for k in ("status_counts", "quantum_journal_counts", "pairs")}))


if __name__ == "__main__":
    main()
