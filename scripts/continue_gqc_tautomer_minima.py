#!/usr/bin/env python3
"""One explicitly budgeted internal-coordinate continuation of unresolved starts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np

from gqc_tautomer_pilot import ROOT, BOHR_ANGSTROM, BudgetExceeded, Meter, build_identity, calculator, digest, gqc_eval, write_json


def merge_rows(inputs, previous, attempted):
    merged = {r["start_id"]: r for r in previous}
    merged.update({r["start_id"]: r for r in attempted})
    return [merged.get(r["start_id"], {**{k: r[k] for k in ("id", "group", "form", "start_id", "formula", "stereo_smiles", "charge", "multiplicity")}, "status": "not_run_budget"}) for r in inputs]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "continuation", "inputs", "optimized", "gate", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    if socket.gethostname() != "node2" or not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("continuation requires n2 Slurm GPU")
    a.out.mkdir(parents=True, exist_ok=False)
    cfg, policy, pack, old, gate = [json.loads(x.read_text()) for x in (a.config, a.continuation, a.inputs, a.optimized, a.gate)]
    identity = build_identity()
    if pack["config"] != cfg or old["config_sha256"] != digest(a.config) or old["input_sha256"] != digest(a.inputs) or old["build_identity"] != identity or gate["build_identity"] != identity or gate["status"] != "qualified_for_bounded_pilot":
        raise ValueError("continuation identity mismatch")
    if old["resources"]["calls_unsettled"]:
        raise ValueError("unsettled prior work must be reconciled before continuation")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("continuation source must be clean")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    prior = old["resources"]["calls_started"] + gate["resources"]["calls_started"]
    cap = min(policy["max_new_calls"], policy["shared_qc_ceiling"] - prior - policy["reserved_followup_calls"])
    meter = Meter(a.out, max(cap, 0), policy["wall_seconds"])
    report = {"schema": "xtbflow-gqc-tautomer-stage/v1", "stage": "registered_continuation", "status": "started",
              "host": socket.gethostname(), "slurm_job_id": os.environ["SLURM_JOB_ID"], "source_commit": source,
              "config_sha256": digest(a.config), "input_sha256": digest(a.inputs), "build_identity": identity,
              "continuation_policy_sha256": digest(a.continuation), "previous_report_sha256": digest(a.optimized),
              "prior_optimization_resources": old["resources"], "remaining_call_cap": cap, "rows": []}
    cpu_start = time.process_time()
    old_rows = {r["start_id"]: r for r in old["rows"]}
    try:
        from geometric.engine import Engine
        from geometric.molecule import Molecule
        from geometric.optimize import run_optimizer
        calc = calculator(cfg)
        for row in pack["rows"]:
            sid = row["start_id"]
            previous = old_rows.get(sid)
            if previous and previous["status"] == "force_converged_identity_pending":
                report["rows"].append(previous)
                continue
            entry = {k: row[k] for k in ("id", "group", "form", "start_id", "formula", "stereo_smiles", "charge", "multiplicity", "symbols")}
            entry.update(status="continuation_started", previous_status=previous["status"] if previous else "not_reached_before_budget", continuation_attempt=1)
            report["rows"].append(entry)
            before = meter.calls
            try:
                x = np.asarray(previous["coordinates_angstrom"] if previous and "coordinates_angstrom" in previous else row["coordinates_angstrom"])
                mol = Molecule()
                mol.elem, mol.xyzs, mol.comms = row["symbols"], [x], [sid]
                mol.build_topology()
                class NativeEngine(Engine):
                    def calc_new(self, coords, dirname):
                        e, f = gqc_eval(calc, row, np.asarray(coords), meter, sid + ":TRIC")
                        return {"energy": e, "gradient": -f.ravel()}
                engine = NativeEngine(mol)
                working = a.out / sid
                working.mkdir()
                cwd = os.getcwd()
                os.chdir(working)
                try:
                    progress = run_optimizer(customengine=engine, input="gqc.in", prefix="opt", coordsys="tric", maxiter=policy["max_iterations_per_start"],
                        convergence_energy=1e-7, convergence_grms=3e-5, convergence_gmax=3e-5,
                        convergence_drms=6e-4, convergence_dmax=1e-3)
                finally:
                    os.chdir(cwd)
                xyz = np.asarray(progress.xyzs[-1])
                e, f = gqc_eval(calc, row, xyz / BOHR_ANGSTROM, meter, sid + ":final")
                fmax = float(np.max(np.abs(f)))
                entry.update(energy_hartree=e, coordinates_angstrom=xyz.tolist(), max_force_hartree_per_bohr=fmax,
                             status="force_converged_identity_pending" if fmax <= policy["force_max_hartree_per_bohr"] else "optimization_unresolved")
            except BudgetExceeded:
                entry["status"] = "continuation_budget_exhausted"
                raise
            except Exception as exc:
                import traceback
                traceback.print_exc()
                entry.update(status="continuation_failed", error_type=type(exc).__name__)
            finally:
                entry["continuation_calls_started"] = meter.calls - before
                write_json(a.out / "partial.json", {**report, "new_resources": meter.summary()})
        report["status"] = "continuation_complete_identity_pending"
    except Exception as exc:
        report.update(status="budget_exhausted" if isinstance(exc, BudgetExceeded) else "execution_failed", error_type=type(exc).__name__)
    finally:
        report["rows"] = merge_rows(pack["rows"], old["rows"], report["rows"])
        report["new_resources"] = {**meter.summary(), "process_cpu_seconds": time.process_time() - cpu_start}
        report["resources"] = {k: old["resources"].get(k, 0) + meter.summary().get(k, 0) for k in ("calls_started", "calls_completed", "calls_failed", "calls_unsettled")}
        write_json(a.out / "report.json", report)
        print(json.dumps({"status": report["status"], "new_resources": report["new_resources"]}), flush=True)
    return 0 if report["status"] == "continuation_complete_identity_pending" else 1


if __name__ == "__main__":
    raise SystemExit(main())
