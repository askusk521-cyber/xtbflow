#!/usr/bin/env python3
"""Bounded GFN2 unit/interface qualification on explicit small-molecule fixtures."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xtbflow.calculators import (  # noqa: E402
    BOHR_IN_ANGSTROM,
    CalculatorProtocol,
    MolecularSystem,
    XTBOracleAdapter,
)

PROTOCOL = {
    "accuracy": 1.0,
    "max_iterations": 250,
    "electronic_temperature": 300.0,
    "solvent": None,
}
FD_STEP_ANGSTROM = 1.0e-4
THRESHOLDS = {
    "adapter_native_energy_abs_hartree": 1.0e-12,
    "adapter_native_force_max_abs_hartree_per_angstrom": 1.0e-10,
    "native_ase_energy_abs_hartree": 1.0e-10,
    "native_ase_force_max_abs_hartree_per_angstrom": 1.0e-9,
    "finite_difference_abs_hartree_per_angstrom": 5.0e-5,
    "xtb_energy_abs_hartree": 1.0e-4,
    "xtb_force_max_abs_hartree_per_angstrom": 1.0e-3,
    "repeat_energy_abs_hartree": 1.0e-12,
    "repeat_force_max_abs_hartree_per_angstrom": 1.0e-10,
}

ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16}

FIXTURES = (
    {
        "id": "water_perturbed",
        "symbols": ("O", "H", "H"),
        "coordinates": ((0.0, 0.0, 0.0), (0.97, 0.03, 0.0), (-0.25, 0.91, 0.04)),
        "charge": 0,
        "multiplicity": 1,
        "fd_coordinate": (1, 0),
    },
    {
        "id": "ammonia_perturbed",
        "symbols": ("N", "H", "H", "H"),
        "coordinates": ((0.0, 0.0, 0.08), (0.94, 0.0, -0.28), (-0.47, 0.82, -0.34), (-0.43, -0.78, -0.25)),
        "charge": 0,
        "multiplicity": 1,
        "fd_coordinate": (1, 0),
    },
    {
        "id": "hydrogen_sulfide_perturbed",
        "symbols": ("S", "H", "H"),
        "coordinates": ((0.0, 0.0, 0.0), (1.33, 0.05, 0.0), (-0.04, 1.34, 0.05)),
        "charge": 0,
        "multiplicity": 1,
        "fd_coordinate": (1, 0),
    },
    {
        "id": "hydrogen_cyanide_perturbed",
        "symbols": ("H", "C", "N"),
        "coordinates": ((0.0, 0.0, 0.0), (1.05, 0.02, 0.0), (2.20, -0.01, 0.03)),
        "charge": 0,
        "multiplicity": 1,
        "fd_coordinate": (1, 0),
    },
    {
        "id": "hydronium_perturbed",
        "symbols": ("O", "H", "H", "H"),
        "coordinates": ((0.0, 0.0, 0.0), (0.98, 0.0, 0.05), (-0.49, 0.85, -0.02), (-0.46, -0.79, 0.12)),
        "charge": 1,
        "multiplicity": 1,
        "fd_coordinate": (1, 0),
    },
    {
        "id": "hydroxide_perturbed",
        "symbols": ("O", "H"),
        "coordinates": ((0.0, 0.0, 0.0), (0.99, 0.03, 0.02)),
        "charge": -1,
        "multiplicity": 1,
        "fd_coordinate": (1, 0),
    },
)


def _protocol(implementation: str) -> CalculatorProtocol:
    return CalculatorProtocol(
        protocol_id=f"{implementation}-gfn2-qualification-v1",
        calculator="xtb_oracle",
        method="GFN2-xTB",
        backend="cpu",
        parameters={**PROTOCOL, "implementation": implementation},
    )


def _system(fixture: dict[str, Any]) -> MolecularSystem:
    return MolecularSystem(
        tuple(fixture["symbols"]),
        tuple(tuple(float(v) for v in row) for row in fixture["coordinates"]),
        int(fixture["charge"]),
        int(fixture["multiplicity"]),
        {"periodic": False, "qualification_fixture": fixture["id"]},
        fixture["id"],
    )


def _max_force_delta(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.max(np.abs(first - second)))


def _adapter_result(adapter: XTBOracleAdapter, system: MolecularSystem) -> tuple[float, np.ndarray, float]:
    start = time.perf_counter()
    result = adapter.evaluate(system, operation="energy_forces")
    elapsed = time.perf_counter() - start
    if result.energy is None or result.forces is None:
        raise RuntimeError("adapter did not return energy and forces")
    return float(result.energy), np.asarray(result.forces, dtype=np.float64), elapsed


def _tblite_native(system: MolecularSystem) -> tuple[float, np.ndarray, float]:
    from ase.units import Hartree, kB
    from tblite.interface import Calculator

    numbers = np.asarray([ATOMIC_NUMBERS[symbol] for symbol in system.symbols], dtype=np.int32)
    positions = np.asarray(system.coordinates, dtype=np.float64) / BOHR_IN_ANGSTROM
    calc = Calculator("GFN2-xTB", numbers, positions, charge=system.charge, uhf=system.multiplicity - 1)
    calc.set("verbosity", 0)
    calc.set("accuracy", PROTOCOL["accuracy"])
    calc.set("max-iter", PROTOCOL["max_iterations"])
    calc.set("temperature", PROTOCOL["electronic_temperature"] * kB / Hartree)
    start = time.perf_counter()
    result = calc.singlepoint()
    elapsed = time.perf_counter() - start
    energy = float(result.get("energy"))
    forces = -np.asarray(result.get("gradient"), dtype=np.float64) / BOHR_IN_ANGSTROM
    return energy, forces, elapsed


def _tblite_ase(system: MolecularSystem) -> tuple[float, np.ndarray, float]:
    from ase import Atoms
    from ase.units import Hartree
    from tblite.ase import TBLite

    atoms = Atoms(symbols=list(system.symbols), positions=np.asarray(system.coordinates, dtype=np.float64))
    atoms.calc = TBLite(
        method="GFN2-xTB",
        charge=system.charge,
        multiplicity=system.multiplicity,
        accuracy=PROTOCOL["accuracy"],
        max_iterations=PROTOCOL["max_iterations"],
        electronic_temperature=PROTOCOL["electronic_temperature"],
        verbosity=0,
        cache_api=False,
    )
    start = time.perf_counter()
    energy = float(atoms.get_potential_energy()) / Hartree
    forces = np.asarray(atoms.get_forces(), dtype=np.float64) / Hartree
    elapsed = time.perf_counter() - start
    return energy, forces, elapsed


def _finite_difference(
    adapter: XTBOracleAdapter,
    system: MolecularSystem,
    atom: int,
    axis: int,
) -> tuple[float, float]:
    coordinates = np.asarray(system.coordinates, dtype=np.float64)
    plus = coordinates.copy()
    minus = coordinates.copy()
    plus[atom, axis] += FD_STEP_ANGSTROM
    minus[atom, axis] -= FD_STEP_ANGSTROM
    start = time.perf_counter()
    e_plus = adapter.evaluate(system.with_coordinates(tuple(map(tuple, plus))), operation="energy").energy
    e_minus = adapter.evaluate(system.with_coordinates(tuple(map(tuple, minus))), operation="energy").energy
    elapsed = time.perf_counter() - start
    if e_plus is None or e_minus is None:
        raise RuntimeError("finite-difference energy call returned no energy")
    return -float(e_plus - e_minus) / (2.0 * FD_STEP_ANGSTROM), elapsed


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tblite_adapter = XTBOracleAdapter(protocol=_protocol("tblite"), implementation="tblite")
    xtb_adapter = XTBOracleAdapter(protocol=_protocol("xtb"), implementation="xtb")
    tblite_cap = tblite_adapter.capabilities
    xtb_cap = xtb_adapter.capabilities
    rows: list[dict[str, Any]] = []
    actual_calls = 0

    for fixture in FIXTURES:
        system = _system(fixture)
        row: dict[str, Any] = {
            "fixture_id": fixture["id"],
            "symbols": list(system.symbols),
            "coordinates_angstrom": [list(item) for item in system.coordinates],
            "charge": system.charge,
            "multiplicity": system.multiplicity,
            "input_hash": system.input_hash,
            "fd_coordinate": list(fixture["fd_coordinate"]),
        }
        try:
            adapter_energy, adapter_forces, adapter_seconds = _adapter_result(tblite_adapter, system)
            native_energy, native_forces, native_seconds = _tblite_native(system)
            ase_energy, ase_forces, ase_seconds = _tblite_ase(system)
            xtb_energy, xtb_forces, xtb_seconds = _adapter_result(xtb_adapter, system)
            fd_force, fd_seconds = _finite_difference(tblite_adapter, system, *fixture["fd_coordinate"])
            actual_calls += 6
            atom, axis = fixture["fd_coordinate"]
            deltas = {
                "adapter_native_energy_abs_hartree": abs(adapter_energy - native_energy),
                "adapter_native_force_max_abs_hartree_per_angstrom": _max_force_delta(adapter_forces, native_forces),
                "native_ase_energy_abs_hartree": abs(native_energy - ase_energy),
                "native_ase_force_max_abs_hartree_per_angstrom": _max_force_delta(native_forces, ase_forces),
                "finite_difference_abs_hartree_per_angstrom": abs(fd_force - float(adapter_forces[atom, axis])),
                "xtb_energy_abs_hartree": abs(adapter_energy - xtb_energy),
                "xtb_force_max_abs_hartree_per_angstrom": _max_force_delta(adapter_forces, xtb_forces),
            }
            checks = {name: value <= THRESHOLDS[name] for name, value in deltas.items()}
            row.update(
                {
                    "status": "success",
                    "energies_hartree": {
                        "tblite_adapter": adapter_energy,
                        "tblite_native": native_energy,
                        "tblite_ase": ase_energy,
                        "xtb": xtb_energy,
                    },
                    "forces_hartree_per_angstrom": {
                        "tblite_adapter": adapter_forces.tolist(),
                        "tblite_native": native_forces.tolist(),
                        "tblite_ase": ase_forces.tolist(),
                        "xtb": xtb_forces.tolist(),
                    },
                    "finite_difference_force_hartree_per_angstrom": fd_force,
                    "deltas": deltas,
                    "checks": checks,
                    "timings_seconds": {
                        "tblite_adapter": adapter_seconds,
                        "tblite_native": native_seconds,
                        "tblite_ase": ase_seconds,
                        "xtb": xtb_seconds,
                        "finite_difference_pair": fd_seconds,
                    },
                }
            )
        except Exception as exc:
            row.update({"status": "failure", "error": f"{type(exc).__name__}: {exc}", "checks": {}})
        rows.append(row)

    repeat = {"status": "not_run"}
    if rows and rows[0].get("status") == "success":
        system = _system(FIXTURES[0])
        first_energy = rows[0]["energies_hartree"]["tblite_adapter"]
        first_forces = np.asarray(rows[0]["forces_hartree_per_angstrom"]["tblite_adapter"])
        repeat_energy, repeat_forces, repeat_seconds = _adapter_result(tblite_adapter, system)
        actual_calls += 1
        repeat_deltas = {
            "repeat_energy_abs_hartree": abs(first_energy - repeat_energy),
            "repeat_force_max_abs_hartree_per_angstrom": _max_force_delta(first_forces, repeat_forces),
        }
        repeat = {
            "status": "success",
            "fixture_id": FIXTURES[0]["id"],
            "deltas": repeat_deltas,
            "checks": {name: value <= THRESHOLDS[name] for name, value in repeat_deltas.items()},
            "timing_seconds": repeat_seconds,
        }

    all_case_checks = [value for row in rows for value in row.get("checks", {}).values()]
    interface_checks = [
        row["checks"][key]
        for row in rows
        if row.get("status") == "success"
        for key in (
            "adapter_native_energy_abs_hartree",
            "adapter_native_force_max_abs_hartree_per_angstrom",
            "native_ase_energy_abs_hartree",
            "native_ase_force_max_abs_hartree_per_angstrom",
        )
    ]
    fd_checks = [row["checks"]["finite_difference_abs_hartree_per_angstrom"] for row in rows if row.get("status") == "success"]
    xtb_checks = [
        row["checks"][key]
        for row in rows
        if row.get("status") == "success"
        for key in ("xtb_energy_abs_hartree", "xtb_force_max_abs_hartree_per_angstrom")
    ]
    complete = len(rows) == len(FIXTURES) and all(row.get("status") == "success" for row in rows)
    passed = complete and bool(all_case_checks) and all(all_case_checks) and all(repeat.get("checks", {}).values())
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report = {
        "schema": "xtbflow-gfn2-qualification/v1",
        "status": "qualified_limited_scope" if passed else "qualification_failed",
        "scientific_qualification": passed,
        "source_commit": source_commit,
        "script_sha256": _sha256(Path(__file__).resolve()),
        "runtime": {
            "host": platform.node(),
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "tblite": importlib.metadata.version("tblite"),
            "xtb": importlib.metadata.version("xtb"),
            "ase": importlib.metadata.version("ase"),
        },
        "protocol": PROTOCOL,
        "finite_difference_step_angstrom": FD_STEP_ANGSTROM,
        "thresholds_predeclared": THRESHOLDS,
        "backend_identity": {
            "tblite": {
                "version": tblite_cap.version,
                "build_hash": tblite_cap.build_hash,
                "qualification": tblite_cap.qualification,
            },
            "xtb": {
                "version": xtb_cap.version,
                "build_hash": xtb_cap.build_hash,
                "qualification": xtb_cap.qualification,
            },
        },
        "gates": {
            "all_fixtures_completed": complete,
            "tblite_interface_parity": complete and bool(interface_checks) and all(interface_checks),
            "angstrom_finite_difference": complete and bool(fd_checks) and all(fd_checks),
            "independent_xtb_parity": complete and bool(xtb_checks) and all(xtb_checks),
            "repeatability": repeat.get("status") == "success" and all(repeat.get("checks", {}).values()),
        },
        "calculator_calls_recorded": actual_calls,
        "planned_max_calls": 37,
        "fixtures": rows,
        "repeatability": repeat,
        "qualified_scope": (
            "CPU GFN2-xTB energy/force unit and numerical parity for the tested small, non-periodic, "
            "closed-shell CHNOS neutral/cation/anion fixtures under the recorded protocol."
            if passed
            else "none"
        ),
        "limits": [
            "Fixtures are explicit qualification geometries, not training data and not equilibrium claims.",
            "tblite adapter/native/ASE are three unit/interface paths over the same implementation, not independent physical methods.",
            "xTB is the independent GFN2 implementation check; passing does not establish broad chemical accuracy.",
            "No CUDA, CP2K, transition-state, dataset-admission, model-training, or mechanism result is claimed.",
            "The tested scope may seed bounded same-domain pairing only; untested charge, spin, periodic, solvent, and larger-domain cases remain unknown.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": report["status"], "gates": report["gates"], "calls": actual_calls}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
