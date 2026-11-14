"""Bounded GQC minima/NEB with independent CPU PySCF saddle and IRC checks.

Every backend operation is charged to the shared per-task journal. Hessians
are explicitly PySCF analytic derivatives, never claimed as GQC Hessians.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from gqc_tautomer_pilot import ROOT, BOHR_ANGSTROM, HARTREE_KCAL, digest, write_json
from qc_campaign_task import mol_and_mf

HARTREE_EV = 27.211386245988


def checkpoint(report, ledger, stage):
    report["stage"] = stage
    report["resources"] = ledger.summary()
    write_json(ledger.out / "report.json", report)


def identity(point, xyz):
    command = [os.environ["QC_GRAPH_PYTHON"], str(ROOT / "scripts/qc_campaign_geometry.py")]
    completed = subprocess.run(command, input=json.dumps({"point": point, "coordinates_angstrom": np.asarray(xyz).tolist()}),
                               text=True, capture_output=True, check=True, timeout=60)
    return json.loads(completed.stdout)


def load_upstream(task, ledger, name, required):
    path = ledger.out.parent / name / "report.json"
    if not path.exists():
        return None, {"id": name, "status": "missing"}
    value = json.loads(path.read_text())
    if value["manifest_sha256"] != task["_manifest_sha256"]:
        raise ValueError("upstream manifest mismatch")
    evidence = {"id": name, "report_sha256": digest(path), "status": value["status"]}
    return (value if value["status"] in required else None), evidence


class Native:
    def __init__(self, point, cfg, ledger):
        from generativeqc import Calculator
        self.point, self.ledger = point, ledger
        self.calc = Calculator(**cfg["gqc"])

    def evaluate(self, xyz, label):
        def calculation():
            result = self.calc.singlepoint(list(zip(self.point["symbols"], (np.asarray(xyz) / BOHR_ANGSTROM).tolist())),
                                           charge=self.point["charge"], multiplicity=self.point["multiplicity"], properties=("energy", "forces"))
            energy, force = float(result.energy), np.array(result.forces, dtype=float, copy=True)
            if not result.converged or "cuda" not in str(result.executed_backend).lower():
                raise RuntimeError("GQC SCF/backend failure")
            if not np.isfinite(energy) or force.shape != (len(self.point["symbols"]), 3) or not np.isfinite(force).all():
                raise RuntimeError("invalid GQC energy/force")
            return energy, -force
        return self.ledger.call(label, calculation)


class Reference:
    def __init__(self, point, task, ledger):
        self.point, self.task, self.ledger = point, task, ledger

    def evaluate(self, xyz, label):
        def calculation():
            _, mf = mol_and_mf(self.point, self.task, np.asarray(xyz) / BOHR_ANGSTROM)
            energy = float(mf.kernel())
            if not mf.converged:
                raise RuntimeError("PySCF SCF not converged")
            gradient = np.asarray(mf.nuc_grad_method().kernel())
            if not np.isfinite(energy) or not np.isfinite(gradient).all():
                raise RuntimeError("nonfinite reference energy/force")
            return energy, gradient, mf
        energy, gradient, self.mf = self.ledger.call(label, calculation)
        return energy, gradient

    def hessian(self, xyz, label, probe=False):
        from pyscf.hessian import thermo
        energy, gradient = self.evaluate(xyz, label + ":EF")
        mf = self.mf
        hess = self.ledger.call(label + ":Hessian", lambda: np.asarray(mf.Hessian().kernel()))
        if not np.isfinite(hess).all():
            raise RuntimeError("nonfinite Hessian")
        matrix = hess.transpose(0, 2, 1, 3).reshape(np.asarray(xyz).size, -1)
        vibration = thermo.harmonic_analysis(mf.mol, hess)
        raw = np.asarray(vibration["freq_wavenumber"])
        freq = np.where(np.abs(raw.imag) > 1e-8, -np.abs(raw.imag), raw.real)
        result = {"energy_hartree": energy, "max_force_hartree_per_bohr": float(np.max(np.abs(gradient))),
                  "frequencies_cm1": freq.tolist(), "hessian_backend": "PySCF CPU analytic B3LYP/def2-SVP",
                  "hessian_unit": "hartree/bohr^2", "thermo_code_sha256": digest(Path(thermo.__file__))}
        if probe:
            direction = np.sin(np.arange(np.asarray(xyz).size) + 0.3).reshape(np.asarray(xyz).shape)
            direction -= direction.mean(0)
            direction /= np.linalg.norm(direction)
            h = 0.0005
            _, gp = self.evaluate(xyz + h * BOHR_ANGSTROM * direction, label + ":Hfd_plus")
            _, gm = self.evaluate(xyz - h * BOHR_ANGSTROM * direction, label + ":Hfd_minus")
            numeric = ((gp - gm) / (2 * h)).ravel()
            error = float(np.linalg.norm(numeric - matrix @ direction.ravel()) / max(np.linalg.norm(numeric), 1e-12))
            result["hessian_direction_probe"] = {"step_bohr": h, "relative_error": error, "passed": error <= 0.02}
        if np.all(freq > 0):
            thermal = thermo.thermo(mf, vibration["freq_au"], temperature=298.15, pressure=101325.0)
            result["harmonic_gibbs_hartree"] = float(thermal["G_tot"][0])
            result["thermal_convention"] = "gas ideal RRHO 298.15 K, 1 atm, single conformer; soft modes unresolved physically"
        return result, matrix


def optimize_geometry(point, xyz, evaluator, directory, label, maxiter, **kwargs):
    from geometric.engine import Engine
    from geometric.molecule import Molecule
    from geometric.optimize import run_optimizer
    mol = Molecule()
    mol.elem, mol.xyzs, mol.comms = point["symbols"], [np.asarray(xyz)], [label]
    mol.build_topology()

    class Backend(Engine):
        def calc_new(self, coords, dirname):
            x = np.asarray(coords).reshape(-1, 3) * BOHR_ANGSTROM
            energy, gradient = evaluator.evaluate(x, label)
            # Durable last evaluated geometry survives an optimizer/native failure.
            write_json(directory / "last_evaluation.json", {"coordinates_angstrom": x.tolist(), "energy_hartree": energy,
                       "max_force_hartree_per_bohr": float(np.max(np.abs(gradient)))})
            return {"energy": energy, "gradient": gradient.ravel()}

    directory.mkdir(parents=True, exist_ok=False)
    engine = Backend(mol)
    cwd = os.getcwd()
    os.chdir(directory)
    try:
        progress = run_optimizer(customengine=engine, input="quantum.in", prefix=label, coordsys="tric", maxiter=maxiter,
                                 convergence_energy=1e-7, convergence_grms=3e-5, convergence_gmax=3e-5,
                                 convergence_drms=6e-4, convergence_dmax=1e-3, **kwargs)
        return [np.asarray(x) for x in progress.xyzs]
    finally:
        os.chdir(cwd)


def run_opt(task, point, cfg, ledger, report):
    native = Native(point, cfg, ledger)
    checkpoint(report, ledger, "GQC_optimization")
    frames = optimize_geometry(point, point["coordinates_angstrom"], native, ledger.out / "opt", "opt", cfg["optimization"]["max_iterations"])
    xyz = frames[-1]
    energy, gradient = native.evaluate(xyz, "opt_final")
    found = identity(point, xyz)
    max_force = float(np.max(np.abs(gradient)))
    report.update(energy_hartree=energy, coordinates_angstrom=xyz.tolist(), identity=found, max_force_hartree_per_bohr=max_force,
                  status="optimized_identity_passed" if max_force <= cfg["optimization"]["max_force_hartree_per_bohr"] and found["status"] == "identity_passed" else "optimization_or_identity_unresolved")


def run_frequency(task, point, ledger, report):
    old, evidence = load_upstream(task, ledger, task["upstream"], {"optimized_identity_passed"})
    report["upstream"] = evidence
    if old is None:
        report["status"] = "blocked_upstream_optimization"
        return
    checkpoint(report, ledger, "PySCF_minimum_frequency")
    result, matrix = Reference(point, task, ledger).hessian(np.asarray(old["coordinates_angstrom"]), "frequency")
    np.save(ledger.out / "hessian.npy", matrix)
    report.update(result)
    minimum = min(result["frequencies_cm1"])
    report["status"] = "minimum_verified" if minimum > 0 and result["max_force_hartree_per_bohr"] <= 3e-4 else "frequency_or_reference_gradient_unresolved"


def run_path(task, point, manifest, ledger, report):
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.mep import NEB
    from ase.optimize import FIRE
    cfg, path = manifest["config"], task["path"]
    endpoints = []
    report["upstream"] = []
    for pid in (path["reactant"], path["product"]):
        opt, eo = load_upstream(task, ledger, "opt_" + pid, {"optimized_identity_passed"})
        freq, ef = load_upstream(task, ledger, "freq_" + pid, {"minimum_verified"})
        report["upstream"].extend([eo, ef])
        if opt is None or freq is None:
            report["status"] = "blocked_upstream_endpoint"
            return
        endpoints.append(np.asarray(opt["coordinates_angstrom"]))
    # Proper rigid alignment; atom identities are never permuted.
    from numpy.linalg import svd
    heavy = [i for i, s in enumerate(point["symbols"]) if s != "H"]
    r, p = endpoints
    cr, cp = r[heavy].mean(0), p[heavy].mean(0)
    u, _, vt = svd((p[heavy] - cp).T @ (r[heavy] - cr))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    endpoints[1] = (p - cp) @ (u @ correction @ vt) + cr
    native = Native(point, cfg, ledger)

    class ASEBackend(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            energy, gradient = native.evaluate(atoms.positions, "GQC_NEB_EF")
            self.results = {"energy": energy * HARTREE_EV, "forces": -gradient * HARTREE_EV / BOHR_ANGSTROM}

    images = [Atoms(point["symbols"], positions=endpoints[0]) for _ in range(cfg["paths"]["images"])]
    images[-1].positions = endpoints[1]
    neb = NEB(images, climb=False, method="improvedtangent", remove_rotation_and_translation=True)
    checkpoint(report, ledger, "IDPP_then_GQC_NEB")
    neb.interpolate(method="idpp")
    for im in images:
        im.calc = ASEBackend()
    optimizer = FIRE(neb, logfile=str(ledger.out / "neb.log"), trajectory=str(ledger.out / "neb.traj"), maxstep=0.1)
    optimizer.run(fmax=cfg["paths"]["neb_fmax_ev_angstrom"], steps=cfg["paths"]["neb_relax_steps"])
    neb.climb = True
    converged = bool(optimizer.run(fmax=cfg["paths"]["neb_fmax_ev_angstrom"], steps=cfg["paths"]["neb_climb_steps"]))
    energies = [float(im.get_potential_energy() / HARTREE_EV) for im in images]
    maximum = 1 + int(np.argmax(energies[1:-1]))
    xyz = images[maximum].positions.copy()
    report["neb"] = {"converged": converged, "image_energies_hartree": energies, "selected_image": maximum,
                     "status": "candidate_only_not_a_TS", "image_coordinates_angstrom": [im.positions.tolist() for im in images]}
    checkpoint(report, ledger, "PySCF_transition_state")
    reference = Reference(point, task, ledger)
    initial, matrix = reference.hessian(xyz, "TS_initial", probe=True)
    report["initial_hessian_probe"] = initial["hessian_direction_probe"]
    if not initial["hessian_direction_probe"]["passed"]:
        report["status"] = "hessian_qualification_failed"
        return
    hfile = ledger.out / "initial_hessian.txt"
    np.savetxt(hfile, matrix)
    frames = optimize_geometry(point, xyz, reference, ledger.out / "ts", "ts", cfg["paths"]["ts_steps"], transition=True, hessian="file:" + str(hfile))
    xyz = frames[-1]
    result, matrix = reference.hessian(xyz, "TS_final")
    report["transition_state"] = {**result, "coordinates_angstrom": xyz.tolist()}
    freq = np.asarray(result["frequencies_cm1"])
    if np.sum(freq < cfg["paths"]["maximum_negative_frequency_cm1"]) != 1 or np.sum(freq < 0) != 1 or result["max_force_hartree_per_bohr"] > 3e-5:
        report["status"] = "saddle_not_verified"
        return
    hfile = ledger.out / "final_hessian.txt"
    np.savetxt(hfile, matrix)
    checkpoint(report, ledger, "PySCF_bidirectional_IRC")
    frames = optimize_geometry(point, xyz, reference, ledger.out / "irc", "irc", cfg["paths"]["irc_steps"],
                               irc=True, irc_direction="both", hessian="file:" + str(hfile), trust=0.2, tmax=0.2)
    report["irc"] = {"frames": len(frames), "status": "optimizer_returned"}
    report["irc_endpoints"] = []
    matched = []
    for side, start in enumerate((frames[0], frames[-1])):
        checkpoint(report, ledger, "PySCF_IRC_endpoint_" + str(side))
        relaxed = optimize_geometry(point, start, reference, ledger.out / ("end_" + str(side)), "end", 80)[-1]
        result, _ = reference.hessian(relaxed, "endpoint_" + str(side))
        ids = {pid: identity(manifest["points"][pid], relaxed) for pid in (path["reactant"], path["product"])}
        record = {**result, "coordinates_angstrom": relaxed.tolist(), "identities": ids}
        record["minimum_passed"] = min(result["frequencies_cm1"]) > 0 and result["max_force_hartree_per_bohr"] <= 3e-5
        report["irc_endpoints"].append(record)
        matched.append([pid for pid, value in ids.items() if value["status"] == "identity_passed"])
    connected = all(e["minimum_passed"] for e in report["irc_endpoints"]) and (
        path["reactant"] in matched[0] and path["product"] in matched[1] or path["reactant"] in matched[1] and path["product"] in matched[0])
    report["status"] = "connected_path_verified" if connected else "IRC_endpoint_unresolved_or_different"
    if connected:
        ri = 0 if path["reactant"] in matched[0] else 1
        report["local_electronic_barrier_kcal_mol"] = (report["transition_state"]["energy_hartree"] - report["irc_endpoints"][ri]["energy_hartree"]) * HARTREE_KCAL
        report["barrier_reference"] = "the verified local reactant minimum of this finite gas/cluster path; not separated-species solution free energy"
