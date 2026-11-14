#!/usr/bin/env python3
"""Independently check GQC minima and finite-cohort energy ordering on n2."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np

from gqc_tautomer_pilot import BOHR_ANGSTROM, HARTREE_KCAL, ROOT, BudgetExceeded, Meter, digest, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != "node2" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("real frequency calculations require n2 Slurm")
    args.out.mkdir(parents=True, exist_ok=False)
    cfg, verified, optimized, gate = (json.loads(p.read_text()) for p in (args.config, args.verified, args.optimized, args.gate))
    for key in ("config_sha256", "input_sha256", "build_identity"):
        if not (verified[key] == optimized[key] == gate[key]):
            raise ValueError("source stage identity mismatch")
    prior_calls = optimized["resources"]["calls_started"] + gate["resources"]["calls_started"]
    cap = min(cfg["caps"]["qc_operations"], cfg["shared_qc_operation_ceiling"] - prior_calls)
    meter = Meter(args.out, max(0, cap), cfg["caps"]["wall_seconds"], cfg["caps"]["hessians"])
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("frequency source must be clean")
    report = {"schema": "xtbflow-gqc-tautomer-minima/v1", "host": socket.gethostname(), "source_commit": source,
              "config_sha256": digest(args.config), "verified_sha256": digest(args.verified),
              "optimized_sha256": digest(args.optimized), "gate_sha256": digest(args.gate),
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "prior_qc_operations": prior_calls,
              "remaining_operation_cap": cap, "status": "started", "rows": [],
              "start_status_counts": dict(Counter(r["status"] for r in verified["rows"])),
              "claim_limits": cfg["claim_limits"]}
    cpu_start = time.process_time()
    selected = {}
    for row in verified["rows"]:
        if row["status"] == "geometry_identity_verified" and (row["id"] not in selected or row["energy_hartree"] < selected[row["id"]]["energy_hartree"]):
            selected[row["id"]] = row
    report["selected_start_ids"] = {key: row["start_id"] for key, row in selected.items()}
    probe_species = next((sid for sid in cfg["hessian_probe"]["species_priority"] if sid in selected), None)
    try:
        import pyscf
        from pyscf import dft, gto, lib
        from pyscf.hessian import thermo
        lib.num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
        report["pyscf_version"] = pyscf.__version__
        report["thermo_code_sha256"] = digest(Path(thermo.__file__))

        def scf(row, xyz, xc, basis, grid, label, gradient=True):
            def calculate():
                mol = gto.M(atom=list(zip(row["symbols"], xyz.tolist())), unit="Bohr", basis=basis,
                            charge=row["charge"], spin=row["multiplicity"] - 1, cart=False, verbose=0)
                mf = dft.RKS(mol)
                mf.xc, mf.grids.level = xc, grid
                mf.conv_tol, mf.max_cycle = cfg["scf_tolerance"], cfg["scf_max_cycle"]
                energy = float(mf.kernel())
                if not mf.converged:
                    raise RuntimeError("reference SCF did not converge")
                grad = np.asarray(mf.nuc_grad_method().kernel()) if gradient else None
                return energy, grad, mf
            return meter.run(label, calculate)

        for sid, row in selected.items():
            entry = {k: row[k] for k in ("id", "group", "form", "start_id", "formula", "charge", "multiplicity", "stereo_smiles", "energy_hartree")}
            entry["status"] = "started"
            report["rows"].append(entry)
            try:
                xyz = np.asarray(row["coordinates_angstrom"]) / BOHR_ANGSTROM
                energy, grad, mf = scf(row, xyz, cfg["xc"], cfg["basis"], cfg["grids_level"], sid + ":reference_EF")
                entry.update(pyscf_energy_hartree=energy, cross_energy_abs_error_hartree=abs(energy - row["energy_hartree"]),
                             reference_max_force_hartree_per_bohr=float(np.max(np.abs(grad))))
                hessian = meter.run(sid + ":pyscf_hessian", lambda: np.asarray(mf.Hessian().kernel()), hessian=True)
                vibration = thermo.harmonic_analysis(mf.mol, hessian)
                raw_freq = np.asarray(vibration["freq_wavenumber"])
                frequencies = np.where(np.abs(raw_freq.imag) > 1e-8, -np.abs(raw_freq.imag), raw_freq.real)
                entry["frequencies_cm1"] = frequencies.tolist()
                entry["minimum_frequency_cm1"] = float(frequencies.min())
                entry["hessian_backend"] = "PySCF CPU analytic B3LYP/def2-SVP"
                if np.min(frequencies) < -cfg["significant_imaginary_cm1"]:
                    entry["status"] = "not_a_minimum"
                elif np.min(frequencies) <= 0:
                    entry["status"] = "frequency_gray_zone"
                elif entry["reference_max_force_hartree_per_bohr"] > cfg["cross_force_max_hartree_per_bohr"]:
                    entry["status"] = "reference_gradient_unresolved"
                else:
                    entry["status"] = "minimum_within_recorded_protocol"
                    thermal = thermo.thermo(mf, vibration["freq_au"], temperature=cfg["temperature_k"], pressure=cfg["pressure_pa"])
                    entry["harmonic_gibbs_hartree"] = float(thermal["G_tot"][0])
                    entry["zpe_hartree"] = float(thermal["ZPE"][0])
                if sid == probe_species:
                    direction = np.sin(np.arange(xyz.size) + 0.3).reshape(xyz.shape)
                    direction -= direction.mean(axis=0)
                    direction /= np.linalg.norm(direction)
                    matrix = hessian.transpose(0, 2, 1, 3).reshape(xyz.size, xyz.size)
                    analytic = matrix @ direction.ravel()
                    probe = []
                    for step in cfg["hessian_probe"]["steps_bohr"]:
                        _, gp, _ = scf(row, xyz + step * direction, cfg["xc"], cfg["basis"], cfg["grids_level"], sid + ":Hfd_plus")
                        _, gm, _ = scf(row, xyz - step * direction, cfg["xc"], cfg["basis"], cfg["grids_level"], sid + ":Hfd_minus")
                        numerical = ((gp - gm) / (2 * step)).ravel()
                        probe.append({"step_bohr": step, "relative_error": float(np.linalg.norm(numerical - analytic) / max(np.linalg.norm(numerical), 1e-12))})
                    entry["hessian_direction_probe"] = probe
                    report["hessian_probe_passed"] = all(r["relative_error"] <= cfg["hessian_probe"]["relative_error_tolerance"] for r in probe)
                sensitivity = cfg["sensitivity"]
                alternative, _, _ = scf(row, xyz, sensitivity["xc"], sensitivity["basis"], sensitivity["grids_level"], sid + ":method_sensitivity", False)
                entry["m06_2x_def2_tzvp_single_point_hartree"] = alternative
            except BudgetExceeded:
                entry["status"] = "budget_exhausted"
                raise
            except Exception as exc:
                entry.update(status="calculation_failed", error_type=type(exc).__name__)
                import traceback
                traceback.print_exc()
            finally:
                write_json(args.out / "partial.json", {**report, "resources": meter.summary()})
        report["pairs"] = []
        for group in ("R6", "R30", "R51"):
            entries = {r["form"]: r for r in report["rows"] if r["group"] == group}
            if set(entries) != {"enol", "carbonyl"}:
                report["pairs"].append({"group": group, "status": "missing_qualified_form"})
                continue
            enol, carbonyl = entries["enol"], entries["carbonyl"]
            if enol["formula"] != carbonyl["formula"] or enol["charge"] != carbonyl["charge"] or enol["multiplicity"] != carbonyl["multiplicity"]:
                raise ValueError("invalid pair composition or electronic state")
            pair = {"group": group, "sign_convention": "enol minus carbonyl; positive favors carbonyl", "status": "finite_cohort_comparison", "forms_status": [enol["status"], carbonyl["status"]]}
            for field, output in [("energy_hartree", "gqc_delta_e_kcal_mol"), ("m06_2x_def2_tzvp_single_point_hartree", "m06_2x_sp_delta_e_kcal_mol")]:
                if field in enol and field in carbonyl:
                    pair[output] = (enol[field] - carbonyl[field]) * HARTREE_KCAL
            if report.get("hessian_probe_passed") and all(r["status"] == "minimum_within_recorded_protocol" for r in (enol, carbonyl)):
                pair["single_conformer_harmonic_delta_g_kcal_mol"] = (enol["harmonic_gibbs_hartree"] - carbonyl["harmonic_gibbs_hartree"]) * HARTREE_KCAL
            report["pairs"].append(pair)
        report["status"] = "completed_with_recorded_per_structure_status"
    except Exception as exc:
        import traceback
        traceback.print_exc()
        report.update(status="budget_exhausted" if isinstance(exc, BudgetExceeded) else "execution_failed", error_type=type(exc).__name__)
    finally:
        report["resources"] = {**meter.summary(), "process_cpu_seconds": time.process_time() - cpu_start}
        report["campaign_qc_operations"] = prior_calls + meter.calls
        write_json(args.out / "report.json", report)
        print(json.dumps({"status": report["status"], "resources": report["resources"], "pairs": report.get("pairs", [])}), flush=True)
    return 0 if report["status"] == "completed_with_recorded_per_structure_status" else 1


if __name__ == "__main__":
    raise SystemExit(main())
