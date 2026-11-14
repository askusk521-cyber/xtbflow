#!/usr/bin/env python3
"""Run a bounded n2 preparation, exact-build force gate and minimum screen.

Use the xtbflow environment for prepare/verify and the prebuilt GenerativeQC
venv for qualify/optimize. All QC requires a Slurm allocation. Raw coordinates
and journals stay in a new n2 output directory. The public summary strips them.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import time
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BOHR_ANGSTROM = 0.529177210903
HARTREE_KCAL = 627.509474


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024**2):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


class BudgetExceeded(RuntimeError):
    pass


class Meter:
    def __init__(self, out: Path, cap: int, seconds: int, hessian_cap: int = 0):
        self.out, self.cap, self.seconds = out, cap, seconds
        self.hessian_cap = hessian_cap
        self.start = time.monotonic()
        self.calls = self.completed = self.failed = self.hessians = 0

    def run(self, label: str, function: Callable[[], Any], *, hessian: bool = False) -> Any:
        if time.monotonic() - self.start >= self.seconds or self.calls >= self.cap:
            raise BudgetExceeded("call or wall limit reached")
        if hessian and self.hessians >= self.hessian_cap:
            raise BudgetExceeded("Hessian limit reached")
        if sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file()) > 256 * 1024**2:
            raise BudgetExceeded("output limit reached")
        self.calls += 1
        self.hessians += int(hessian)
        t0 = time.monotonic()
        with (self.out / "calls.jsonl").open("a") as stream:
            stream.write(json.dumps({"call": self.calls, "label": label, "status": "started", "hessian": hessian}) + "\n")
            stream.flush()
        try:
            value = function()
        except Exception:
            self.failed += 1
            with (self.out / "calls.jsonl").open("a") as stream:
                stream.write(json.dumps({"call": self.calls, "label": label, "status": "failed", "seconds": time.monotonic() - t0}) + "\n")
            raise
        self.completed += 1
        with (self.out / "calls.jsonl").open("a") as stream:
            stream.write(json.dumps({"call": self.calls, "label": label, "status": "completed", "seconds": time.monotonic() - t0}) + "\n")
        return value

    def summary(self) -> dict[str, Any]:
        return {"calls_started": self.calls, "calls_completed": self.completed,
                "calls_failed": self.failed, "calls_unsettled": self.calls - self.completed - self.failed,
                "hessians_started": self.hessians, "wall_seconds": time.monotonic() - self.start}


def prepare(config: dict[str, Any], out: Path) -> dict[str, Any]:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
    opts = config["preparation"]
    rows, attempted = [], 0
    for species in config["species"]:
        graph = Chem.MolFromSmiles(species["smiles"])
        if graph is None or Chem.GetFormalCharge(graph) != species["charge"]:
            raise ValueError("invalid input graph or explicit charge")
        stereos = sorted(EnumerateStereoisomers(graph, options=StereoEnumerationOptions(
            onlyUnassigned=True, unique=True, maxIsomers=opts["max_stereoisomers_per_graph"])), key=Chem.MolToSmiles)
        for si, stereo in enumerate(stereos):
            mol = Chem.AddHs(stereo)
            if sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) != 0 or species["multiplicity"] != 1:
                raise ValueError("pilot only admits explicitly specified closed-shell graphs")
            if attempted + opts["conformers_per_stereoisomer"] > opts["max_initial_conformers"]:
                raise BudgetExceeded("initial conformer cap")
            params = AllChem.ETKDGv3()
            params.randomSeed = opts["seed"] + attempted
            params.numThreads = 1
            ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=opts["conformers_per_stereoisomer"], params=params))
            attempted += opts["conformers_per_stereoisomer"]
            if not ids:
                raise RuntimeError("geometry embedding failed")
            energies = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=opts["mmff_max_iterations"])
            ci = min(range(len(ids)), key=lambda i: (energies[i][0] != 0, energies[i][1], i))
            row = {**species, "start_id": f"{species['id']}_stereo{si}",
                   "stereo_smiles": Chem.MolToSmiles(stereo, isomericSmiles=True),
                   "formula": rdMolDescriptors.CalcMolFormula(mol),
                   "symbols": [a.GetSymbol() for a in mol.GetAtoms()],
                   "mapped_atom_ids": list(range(1, mol.GetNumAtoms() + 1)),
                   "coordinates_angstrom": mol.GetConformer(ids[ci]).GetPositions().tolist(),
                   "mmff_converged": energies[ci][0] == 0, "embedded_conformers": len(ids)}
            rows.append(row)
    pack = {"schema": "xtbflow-independent-tautomer-input/v1", "source": "explicit graph/ETKDG/MMFF; no source geometries or labels",
            "config": config, "rows": rows, "initial_conformers_attempted": attempted}
    write_json(out / "inputs.json", pack)
    return {"input_sha256": digest(out / "inputs.json"), "starts": len(rows), "initial_conformers_attempted": attempted,
            "rows": [{k: r[k] for k in ("start_id", "stereo_smiles", "formula", "charge", "multiplicity", "mmff_converged")} for r in rows]}


def build_identity() -> dict[str, Any]:
    import generativeqc
    package = Path(generativeqc.__file__).resolve().parent
    libs = sorted(package.rglob("*.so"))
    if not libs:
        raise RuntimeError("native library identity unavailable")
    return {"source_commit_declared": os.environ.get("GQC_SOURCE_SHA", "unknown"),
            "calculator_sha256": digest(package / "calculator.py"),
            "native_libraries": [{"name": str(p.relative_to(package)), "bytes": p.stat().st_size, "sha256": digest(p)} for p in libs]}


def calculator(config: dict[str, Any]) -> Any:
    from generativeqc import Calculator
    return Calculator(**config["generativeqc"])


def gqc_eval(calc: Any, row: dict[str, Any], x: np.ndarray, meter: Meter, label: str,
             forces: bool = True) -> tuple[float, np.ndarray | None]:
    def call() -> tuple[float, np.ndarray | None]:
        result = calc.singlepoint(list(zip(row["symbols"], x.reshape(-1, 3).tolist())),
                                  charge=row["charge"], multiplicity=row["multiplicity"],
                                  properties=("energy", "forces") if forces else ("energy",))
        if not result.converged or "cuda" not in str(result.executed_backend).lower():
            raise RuntimeError("SCF did not converge on the requested CUDA backend")
        energy = float(result.energy)
        force = np.array(result.forces, dtype=float, copy=True) if forces else None
        if not np.isfinite(energy) or (force is not None and not np.isfinite(force).all()):
            raise RuntimeError("nonfinite QC result")
        return energy, force
    return meter.run(label, call)


def reference(row: dict[str, Any], x: np.ndarray, config: dict[str, Any], meter: Meter,
              label: str) -> tuple[float, np.ndarray, Any]:
    from pyscf import dft, gto, lib
    lib.num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    settings = config["reference"]
    mol = gto.M(atom=list(zip(row["symbols"], x.reshape(-1, 3).tolist())), unit="Bohr",
                basis=settings["basis"], charge=row["charge"], spin=row["multiplicity"] - 1,
                cart=settings["cart"], verbose=0)
    mf = dft.RKS(mol)
    mf.xc, mf.grids.level = settings["xc"], settings["grids_level"]
    mf.conv_tol, mf.max_cycle = settings["conv_tol"], settings["max_cycle"]
    def call() -> tuple[float, np.ndarray, Any]:
        energy = float(mf.kernel())
        if not mf.converged:
            raise RuntimeError("independent reference SCF failed")
        return energy, -np.asarray(mf.nuc_grad_method().kernel()), mf
    return meter.run(label, call)


def qualify(config: dict[str, Any], pack: dict[str, Any], out: Path, meter: Meter,
            result: dict[str, Any]) -> None:
    gate = config["qualification"]
    calc = calculator(config)
    water = {"start_id": "water_perturbed", "symbols": ["O", "H", "H"], "charge": 0, "multiplicity": 1,
             "coordinates_angstrom": [[0, 0, 0], [0.78, 0.12, 0.54], [-0.74, -0.04, 0.61]]}
    cohort = [water] + [next(r for r in pack["rows"] if r["id"] == sid) for sid in ("R6_enol", "R6_carbonyl")]
    for row in cohort:
        x = np.array(row["coordinates_angstrom"]) / BOHR_ANGSTROM
        e, f = gqc_eval(calc, row, x, meter, row["start_id"] + ":gqc")
        direction = np.sin(np.arange(x.size) + 0.3).reshape(x.shape)
        direction -= direction.mean(axis=0)
        direction /= np.linalg.norm(direction)
        analytic = float(np.sum(f * direction))
        finite = []
        for step in gate["fd_steps_bohr"]:
            ep, _ = gqc_eval(calc, row, x + step * direction, meter, row["start_id"] + ":fd_plus", False)
            em, _ = gqc_eval(calc, row, x - step * direction, meter, row["start_id"] + ":fd_minus", False)
            fd = -(ep - em) / (2 * step)
            finite.append({"step_bohr": step, "force": fd, "abs_error": abs(fd - analytic)})
        re, rf, _ = reference(row, x, config, meter, row["start_id"] + ":pyscf")
        de, df = abs(re - e), float(np.max(np.abs(rf - f)))
        passed = max(v["abs_error"] for v in finite) <= gate["fd_abs_tolerance_hartree_per_bohr"] and de <= gate["cross_energy_tolerance_hartree"] and df <= gate["cross_force_tolerance_hartree_per_bohr"]
        result["rows"].append({"id": row["start_id"], "gqc_energy_hartree": e, "analytic_direction_force": analytic,
                               "finite_differences": finite, "cross_energy_error_hartree": de,
                               "cross_force_error_hartree_per_bohr": df, "passed": passed})
        write_json(out / "partial.json", {**result, "resources": meter.summary()})
    result["status"] = "qualified_for_bounded_pilot" if all(r["passed"] for r in result["rows"]) else "force_qualification_failed"


def optimize(config: dict[str, Any], pack: dict[str, Any], out: Path, meter: Meter,
             result: dict[str, Any]) -> None:
    from scipy.optimize import minimize
    settings = config["optimization"]
    calc = calculator(config)
    for row in pack["rows"]:
        sid = row["start_id"]
        entry: dict[str, Any] = {k: row[k] for k in ("id", "group", "form", "start_id", "formula", "stereo_smiles", "charge", "multiplicity")}
        before = meter.calls
        entry["status"] = "started"
        result["rows"].append(entry)
        write_json(out / "partial.json", {**result, "resources": meter.summary()})
        try:
            def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
                energy, force = gqc_eval(calc, row, flat, meter, sid + ":optimization")
                return energy, -force.ravel()
            x = np.asarray(row["coordinates_angstrom"]) / BOHR_ANGSTROM
            opt = minimize(objective, x.ravel(), jac=True, method="BFGS", options={
                "gtol": settings["force_max_hartree_per_bohr"], "maxiter": settings["max_iterations_per_start"], "norm": np.inf})
            entry.update(energy_hartree=float(opt.fun), max_force_hartree_per_bohr=float(np.max(np.abs(opt.jac))),
                         iterations=int(opt.nit), optimizer_success=bool(opt.success), optimizer_message=str(opt.message),
                         coordinates_angstrom=(opt.x.reshape(-1, 3) * BOHR_ANGSTROM).tolist(), symbols=row["symbols"])
            entry["status"] = "force_converged_identity_pending" if entry["max_force_hartree_per_bohr"] <= settings["force_max_hartree_per_bohr"] else "optimization_unresolved"
        except BudgetExceeded:
            entry["status"] = "budget_exhausted"
            raise
        except Exception as exc:
            entry.update(status="calculation_failed", error_type=type(exc).__name__)
        finally:
            entry["calls_started"] = meter.calls - before
            write_json(out / "partial.json", {**result, "resources": meter.summary()})
    result["status"] = "optimization_complete_identity_pending"


def verify(pack: dict[str, Any], optimized: dict[str, Any], out: Path) -> dict[str, Any]:
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    rows = []
    for row in optimized["rows"]:
        r = dict(row)
        if row["status"] == "force_converged_identity_pending":
            xyz = str(len(row["symbols"])) + "\nindependent generated geometry\n" + "\n".join(
                s + " " + " ".join(f"{v:.12f}" for v in p) for s, p in zip(row["symbols"], row["coordinates_angstrom"])) + "\n"
            try:
                mol = Chem.MolFromXYZBlock(xyz)
                rdDetermineBonds.DetermineBonds(mol, charge=row["charge"], allowChargedFragments=True, embedChiral=True)
                observed = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
                expected = Chem.MolToSmiles(Chem.MolFromSmiles(row["stereo_smiles"]), isomericSmiles=True)
                r.update(observed_smiles=observed, identity_passed=observed == expected)
                r["status"] = "geometry_identity_verified" if observed == expected else "geometry_identity_changed"
            except Exception as exc:
                r.update(status="geometry_identity_unresolved", error_type=type(exc).__name__)
        rows.append(r)
    return {"status": "identity_checks_complete", "rows": rows, "config_sha256": optimized["config_sha256"],
            "input_sha256": optimized["input_sha256"], "build_identity": optimized["build_identity"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "qualify", "optimize", "verify"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--optimized", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != "node2":
        raise RuntimeError("real chemistry preparation and calculations require node2")
    args.out.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    result: dict[str, Any] = {"schema": "xtbflow-gqc-tautomer-stage/v1", "stage": args.mode, "host": socket.gethostname(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": digest(args.config), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": platform.python_version(), "status": "started", "rows": [], "claim_limits": config["claim_limits"]}
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("source checkout must be clean")
    cpu0, wall0 = time.process_time(), time.monotonic()
    meter = None
    try:
        if args.mode == "prepare":
            result.update(prepare(config, args.out), status="prepared_no_qc")
        else:
            pack = json.loads(args.inputs.read_text())
            result["input_sha256"] = digest(args.inputs)
            if pack["config"] != config:
                raise ValueError("prepared input protocol mismatch")
            if args.mode == "verify":
                optimized = json.loads(args.optimized.read_text())
                if optimized["config_sha256"] != result["config_sha256"] or optimized["input_sha256"] != result["input_sha256"]:
                    raise ValueError("optimization/input/protocol identity mismatch")
                result.update(verify(pack, optimized, args.out))
            else:
                if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
                    raise RuntimeError("QC requires a GPU Slurm allocation")
                import cupy as cp
                result["gpu"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
                result["build_identity"] = build_identity()
                result["packages"] = {}
                for name in ("generativeqc", "numpy", "scipy", "pyscf", "cupy-cuda12x"):
                    try:
                        result["packages"][name] = importlib.metadata.version(name)
                    except importlib.metadata.PackageNotFoundError:
                        result["packages"][name] = "no_distribution_metadata"
                settings = config["qualification" if args.mode == "qualify" else "optimization"]
                meter = Meter(args.out, settings["max_calls"], settings["wall_seconds"], settings.get("max_hessians", 0))
                if args.mode == "qualify":
                    qualify(config, pack, args.out, meter, result)
                else:
                    gate = json.loads(args.gate.read_text())
                    if gate["status"] != "qualified_for_bounded_pilot" or gate["config_sha256"] != result["config_sha256"] or gate["input_sha256"] != result["input_sha256"] or gate["build_identity"] != result["build_identity"]:
                        raise ValueError("missing exact-build/input/protocol force qualification")
                    result["gate_sha256"] = digest(args.gate)
                    optimize(config, pack, args.out, meter, result)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        result["status"] = "budget_exhausted" if isinstance(exc, BudgetExceeded) else "execution_failed"
        result["error_type"] = type(exc).__name__
    finally:
        result["resources"] = {**(meter.summary() if meter else {"calls_started": 0}),
            "driver_wall_seconds": time.monotonic() - wall0, "process_cpu_seconds": time.process_time() - cpu0}
        write_json(args.out / "report.json", result)
        print(json.dumps({"stage": args.mode, "status": result["status"], "resources": result["resources"]}), flush=True)
    return 1 if result["status"] in ("execution_failed", "budget_exhausted", "force_qualification_failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
