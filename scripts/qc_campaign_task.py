#!/usr/bin/env python3
"""One immutable Slurm task, with finite resources and durable failed-work accounting."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import time

import numpy as np
from gqc_tautomer_pilot import ROOT, BOHR_ANGSTROM, BudgetExceeded, build_identity, digest, write_json


class Ledger:
    def __init__(self, out: Path, task: dict):
        self.out, self.task, self.start = out, task, time.monotonic()
        self.started = self.completed = self.failed = 0

    def call(self, label, function):
        if self.started >= self.task["calls"] or time.monotonic() - self.start >= self.task["minutes"] * 60 - 60:
            raise BudgetExceeded("task call or soft wall limit")
        if sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file()) > 4 * 1024**3:
            raise BudgetExceeded("task disk limit")
        if self.started % 10 == 0 and sum(p.stat().st_size for p in self.out.parent.parent.rglob("*") if p.is_file()) > 32 * 1024**3:
            raise BudgetExceeded("campaign retained-disk checkpoint limit")
        self.started += 1
        index, start = self.started, time.monotonic()
        self.log({"operation": index, "label": label, "status": "started", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        try:
            value = function()
        except Exception as exc:
            self.failed += 1
            self.log({"operation": index, "label": label, "status": "failed", "error_type": type(exc).__name__, "seconds": time.monotonic() - start})
            raise
        self.completed += 1
        self.log({"operation": index, "label": label, "status": "completed", "seconds": time.monotonic() - start})
        return value

    def log(self, value):
        with (self.out / "operations.jsonl").open("a") as f:
            f.write(json.dumps(value) + "\n")

    def summary(self):
        return {"started": self.started, "completed": self.completed, "failed": self.failed,
                "unsettled": self.started - self.completed - self.failed, "wall_seconds": time.monotonic() - self.start}


def mol_and_mf(point, task, xyz_bohr=None):
    from pyscf import gto, dft, scf, lib
    lib.num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", task["cpus"])))
    xyz = np.asarray(point["coordinates_angstrom"]) / BOHR_ANGSTROM if xyz_bohr is None else xyz_bohr
    mol = gto.M(atom=list(zip(point["symbols"], xyz.tolist())), unit="Bohr", basis=task["basis"],
                charge=point["charge"], spin=point["multiplicity"] - 1, cart=False,
                max_memory=int(task["mem_gib"] * 1024 * 0.7), verbose=0)
    mf = scf.RHF(mol) if task["kind"] == "ccsd_t" else dft.RKS(mol)
    if task["kind"] != "ccsd_t":
        mf.xc, mf.grids.level = task["xc"], 3
    mf.conv_tol, mf.max_cycle = 1e-11, 150
    return mol, mf


def run_audit(task, point, cfg, ledger, report):
    from generativeqc import Calculator
    params = {**cfg["gqc"], "method": task["method"], "basis": task["basis"]}
    calc = Calculator(**params)
    xyz = np.asarray(point["coordinates_angstrom"]) / BOHR_ANGSTROM
    def evaluate(x, forces):
        def call():
            r = calc.singlepoint(list(zip(point["symbols"], x.tolist())), charge=point["charge"], multiplicity=point["multiplicity"], properties=("energy", "forces") if forces else ("energy",))
            if not r.converged or "cuda" not in str(r.executed_backend).lower():
                raise RuntimeError("GQC convergence/backend failure")
            energy = float(r.energy)
            force = np.array(r.forces, dtype=float, copy=True) if forces else None
            if not np.isfinite(energy) or (forces and (force.shape != (len(point["symbols"]), 3) or not np.isfinite(force).all())):
                raise RuntimeError("nonfinite or invalid native energy/force")
            return energy, force
        return ledger.call("gqc_EF" if forces else "gqc_E", call)
    energy, force = evaluate(xyz, True)
    direction = np.sin(np.arange(xyz.size) + 0.31).reshape(xyz.shape)
    direction -= direction.mean(axis=0)
    direction /= np.linalg.norm(direction)
    analytic = float(np.sum(force * direction))
    fd = []
    for h in cfg["qualification"]["fd_steps_bohr"]:
        ep, _ = evaluate(xyz + h * direction, False)
        em, _ = evaluate(xyz - h * direction, False)
        fd.append({"step_bohr": h, "abs_error": abs(-(ep - em) / (2 * h) - analytic)})
    mol, mf = mol_and_mf(point, task, xyz)
    def ref():
        e = float(mf.kernel())
        if not mf.converged:
            raise RuntimeError("reference SCF failed")
        return e, -np.asarray(mf.nuc_grad_method().kernel())
    er, fr = ledger.call("pyscf_EF", ref)
    de, df = abs(energy - er), float(np.max(np.abs(force - fr)))
    q = cfg["qualification"]
    report.update(energy_hartree=energy, reference_energy_hartree=er, fd=fd, cross_energy_error_hartree=de,
                  cross_force_error_hartree_per_bohr=df, energy_pass=de <= q["cross_energy_tolerance_hartree"],
                  force_pass=max(r["abs_error"] for r in fd) <= q["fd_force_tolerance"] and df <= q["cross_force_tolerance_hartree_per_bohr"])
    report["status"] = "qualification_passed" if report["energy_pass"] and report["force_pass"] else "qualification_failed"


def run_cc(task, point, ledger, report):
    from pyscf import cc
    mol, mf = mol_and_mf(point, task)
    mol.incore_anyway = True
    mf.chkfile = str(ledger.out / "hf.chk")
    ehf = ledger.call("RHF", lambda: float(mf.kernel()))
    if not mf.converged:
        raise RuntimeError("RHF did not converge")
    calc = cc.CCSD(mf)
    calc.max_cycle, calc.conv_tol, calc.max_memory = 50, 1e-8, 24000
    calc.incore_complete = True
    calc.diis_file = str(ledger.out / "ccdiis.h5")
    ecorr, t1, t2 = ledger.call("CCSD", calc.kernel)
    if not calc.converged:
        raise RuntimeError("CCSD did not converge")
    triples = ledger.call("perturbative_triples", lambda: float(calc.ccsd_t()))
    report.update(status="completed_finite_basis_reference", rhf_hartree=ehf,
                  ccsd_hartree=float(calc.e_tot), triples_correction_hartree=triples,
                  ccsdt_hartree=float(calc.e_tot + triples), frozen_core=False,
                  t1_frobenius_norm=float(np.linalg.norm(t1)), basis=task["basis"], nao=mol.nao,
                  claim_limit="All-electron CCSD(T)/cc-pVDZ single point; finite-basis, single-reference assumptions remain.")


def run_solvent(task, point, ledger, report):
    mol, mf = mol_and_mf(point, task)
    mf = mf.PCM()
    mf.with_solvent.method, mf.with_solvent.eps = "C-PCM", task["epsilon"]
    energy = ledger.call("CPCM_single_point", lambda: float(mf.kernel()))
    if not mf.converged:
        raise RuntimeError("CPCM SCF failed")
    report.update(status="completed_solvent_single_point", energy_hartree=energy, solvent_model="C-PCM", epsilon=task["epsilon"],
                  claim_limit="Electrostatic continuum single point at gas geometry; not a solvent free-energy or rate prediction.")


def run_single_point(task, point, ledger, report):
    _, mf = mol_and_mf(point, task)
    energy = ledger.call("gas_single_point", lambda: float(mf.kernel()))
    if not mf.converged or not np.isfinite(energy):
        raise RuntimeError("gas reference SCF failed")
    report.update(status="completed_gas_single_point", energy_hartree=energy)


def numerical_gate(task, manifest, out, report):
    if "gate" not in task:
        return True
    wave, name = task["gate"].split("/")
    path = out.parent.parent / ("results-" + wave) / name / "report.json"
    if not path.exists():
        report.update(status="blocked_numerical_gate", gate_evidence={"id": task["gate"], "status": "missing"})
        return False
    gate = json.loads(path.read_text())
    expected_manifest = manifest["wave1_manifest_sha256"] if wave == "wave1" else task["_manifest_sha256"]
    native = next((r["sha256"] for r in gate.get("build_identity", {}).get("native_libraries", []) if r["name"] == "lib/libgenerativeqc.so"), None)
    allowed = gate["status"] == "qualification_passed" and gate["manifest_sha256"] == expected_manifest and native == manifest["config"]["native_sha256"]
    allowed = allowed and gate["task"]["method"] == "b3lyp-rks" and gate["task"]["basis"] == "def2-svp"
    report["gate_evidence"] = {"id": task["gate"], "status": gate["status"], "report_sha256": digest(path), "identity_valid": allowed}
    if not allowed:
        report["status"] = "blocked_numerical_gate"
    return allowed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if socket.gethostname() != "node2" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("QC campaign tasks require n2 Slurm")
    manifest = json.loads(a.manifest.read_text())
    task = dict(manifest["tasks"][a.index])
    point = manifest["points"][task["point"]]
    out = a.out / task["id"]
    out.mkdir(parents=True, exist_ok=False)
    scratch = out / "scratch"
    scratch.mkdir()
    os.environ["TMPDIR"] = str(scratch)
    import tempfile
    tempfile.tempdir = str(scratch)
    if point["multiplicity"] != 1:
        raise ValueError("this frozen campaign only contains explicit singlets")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("campaign source must be clean")
    report = {"schema": "xtbflow-qc-task-result/v1", "task": task, "status": "started", "source_commit": source,
              "manifest_sha256": digest(a.manifest), "config_sha256": manifest["config_sha256"],
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(), "python": platform.python_version(),
              "input_point": {k: v for k, v in point.items() if k not in ("symbols", "coordinates_angstrom")},
              "claim_limits": manifest["config"]["claim_limits"]}
    write_json(out / "report.json", report)
    ledger = Ledger(out, task)
    task["_manifest_sha256"] = digest(a.manifest)
    cpu = time.process_time()
    def terminating(signum, frame):
        raise BudgetExceeded("scheduler termination signal")
    signal.signal(signal.SIGTERM, terminating)
    try:
        import importlib.metadata
        report["environment_versions"] = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pyscf", "geometric")}
        report["allocated_cpus"] = int(os.environ["SLURM_CPUS_PER_TASK"])
        if task["gpu"]:
            if not os.environ.get("CUDA_VISIBLE_DEVICES"):
                raise RuntimeError("missing Slurm GPU visibility")
            library = Path(os.environ["GQC_NATIVE_LIBRARY"])
            if digest(library) != manifest["config"]["native_sha256"]:
                raise ValueError("unqualified native library")
            os.environ["GENERATIVEQC_LIBRARY"] = str(library)
            report["build_identity"] = build_identity()
        if not numerical_gate(task, manifest, out, report):
            pass
        elif task["kind"] == "audit":
            run_audit(task, point, manifest["config"], ledger, report)
        elif task["kind"] == "ccsd_t":
            run_cc(task, point, ledger, report)
        elif task["kind"] == "solvent":
            run_solvent(task, point, ledger, report)
        elif task["kind"] == "single_point":
            run_single_point(task, point, ledger, report)
        elif task["kind"] in ("opt", "frequency", "path"):
            from qc_campaign_paths import run_opt, run_frequency, run_path
            if task["kind"] == "opt":
                run_opt(task, point, manifest["config"], ledger, report)
            elif task["kind"] == "frequency":
                run_frequency(task, point, ledger, report)
            else:
                run_path(task, point, manifest, ledger, report)
        else:
            raise ValueError("unregistered task kind")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        report.update(status="budget_exhausted" if isinstance(exc, BudgetExceeded) else "execution_failed", error_type=type(exc).__name__, error_message=str(exc))
    finally:
        report["resources"] = {**ledger.summary(), "process_cpu_seconds": time.process_time() - cpu}
        write_json(out / "report.json", report)
        print(json.dumps({"task": task["id"], "status": report["status"], "resources": report["resources"]}), flush=True)
    return 1 if report["status"] in ("execution_failed", "budget_exhausted") else 0


if __name__ == "__main__":
    raise SystemExit(main())
