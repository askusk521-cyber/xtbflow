#!/usr/bin/env python3
"""Cross-check tblite analytic forces against independent xTB energy finite differences."""
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.run_gfn2_qualification as q  # noqa: E402

STEP_ANGSTROM = 1.0e-4
ENERGY_THRESHOLD_HARTREE = 1.0e-4
FORCE_THRESHOLD_HARTREE_PER_ANGSTROM = 5.0e-5


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _energy(adapter, system) -> float:
    result = adapter.evaluate(system, operation="energy")
    if result.energy is None:
        raise RuntimeError("energy call returned no energy")
    return float(result.energy)


def _all_fd_forces(adapter, system) -> tuple[np.ndarray, int, float]:
    coordinates = np.asarray(system.coordinates, dtype=np.float64)
    forces = np.zeros_like(coordinates)
    calls = 0
    start = time.perf_counter()
    for atom in range(coordinates.shape[0]):
        for axis in range(3):
            plus = coordinates.copy()
            minus = coordinates.copy()
            plus[atom, axis] += STEP_ANGSTROM
            minus[atom, axis] -= STEP_ANGSTROM
            e_plus = _energy(adapter, system.with_coordinates(tuple(map(tuple, plus))))
            e_minus = _energy(adapter, system.with_coordinates(tuple(map(tuple, minus))))
            forces[atom, axis] = -(e_plus - e_minus) / (2.0 * STEP_ANGSTROM)
            calls += 2
    return forces, calls, time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    tblite = q.XTBOracleAdapter(protocol=q._protocol("tblite"), implementation="tblite")
    xtb = q.XTBOracleAdapter(protocol=q._protocol("xtb"), implementation="xtb")
    rows = []
    actual_calls = 0

    for fixture in q.FIXTURES:
        system = q._system(fixture)
        start = time.perf_counter()
        tbl_result = tblite.evaluate(system, operation="energy_forces")
        xtb_result = xtb.evaluate(system, operation="energy_forces")
        baseline_seconds = time.perf_counter() - start
        actual_calls += 2
        if tbl_result.energy is None or tbl_result.forces is None or xtb_result.energy is None or xtb_result.forces is None:
            raise RuntimeError("baseline energy/force result is incomplete")
        xtb_fd, fd_calls, fd_seconds = _all_fd_forces(xtb, system)
        actual_calls += fd_calls
        tbl_forces = np.asarray(tbl_result.forces, dtype=np.float64)
        xtb_analytic = np.asarray(xtb_result.forces, dtype=np.float64)
        energy_delta = abs(float(tbl_result.energy) - float(xtb_result.energy))
        fd_delta = np.abs(tbl_forces - xtb_fd)
        analytic_delta = np.abs(tbl_forces - xtb_analytic)
        worst_fd = np.unravel_index(np.argmax(fd_delta), fd_delta.shape)
        worst_analytic = np.unravel_index(np.argmax(analytic_delta), analytic_delta.shape)
        rows.append(
            {
                "fixture_id": fixture["id"],
                "input_hash": system.input_hash,
                "charge": system.charge,
                "multiplicity": system.multiplicity,
                "atom_count": len(system.symbols),
                "energy_abs_delta_hartree": energy_delta,
                "energy_pass": energy_delta <= ENERGY_THRESHOLD_HARTREE,
                "xtb_fd_vs_tblite_force_max_abs_hartree_per_angstrom": float(fd_delta[worst_fd]),
                "xtb_fd_force_pass": float(fd_delta[worst_fd]) <= FORCE_THRESHOLD_HARTREE_PER_ANGSTROM,
                "xtb_fd_worst_component": [int(worst_fd[0]), int(worst_fd[1])],
                "xtb_analytic_vs_tblite_force_max_abs_hartree_per_angstrom": float(analytic_delta[worst_analytic]),
                "xtb_analytic_worst_component": [int(worst_analytic[0]), int(worst_analytic[1])],
                "xtb_analytic_vs_xtb_fd_force_max_abs_hartree_per_angstrom": float(np.max(np.abs(xtb_analytic - xtb_fd))),
                "timings_seconds": {"baseline_pair": baseline_seconds, "xtb_full_fd": fd_seconds},
                "calculator_calls": 2 + fd_calls,
            }
        )

    energy_gate = all(row["energy_pass"] for row in rows)
    force_gate = all(row["xtb_fd_force_pass"] for row in rows)
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report = {
        "schema": "xtbflow-gfn2-independent-fd/v1",
        "status": "pass" if energy_gate and force_gate else "fail",
        "source_commit": source_commit,
        "script_sha256": _sha256(Path(__file__).resolve()),
        "runtime": {
            "host": platform.node(),
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "tblite": importlib.metadata.version("tblite"),
            "xtb": importlib.metadata.version("xtb"),
        },
        "protocol": q.PROTOCOL,
        "step_angstrom": STEP_ANGSTROM,
        "thresholds_predeclared": {
            "energy_abs_hartree": ENERGY_THRESHOLD_HARTREE,
            "independent_xtb_fd_force_max_abs_hartree_per_angstrom": FORCE_THRESHOLD_HARTREE_PER_ANGSTROM,
        },
        "gates": {
            "independent_xtb_energy_parity": energy_gate,
            "independent_xtb_energy_fd_force_parity": force_gate,
        },
        "fixtures": rows,
        "calculator_calls_recorded": actual_calls,
        "planned_max_calls": 126,
        "interpretation": (
            "Independent xTB energy surfaces and numerical energy derivatives agree with tblite in the tested scope."
            if energy_gate and force_gate
            else "Independent xTB energy/finite-difference parity did not meet the fixed threshold."
        ),
        "limits": [
            "xTB analytic gradients are recorded as diagnostics but are not used as the force-parity gate because the preceding qualification run found self-inconsistency on several components.",
            "The force gate uses only independent xTB energy evaluations and central finite differences at the predeclared 1e-4 Angstrom step.",
            "This does not qualify xTB analytic gradients, CUDA, open-shell states, solvent, periodic systems, CP2K, or broader chemical accuracy.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": report["status"], "gates": report["gates"], "calls": actual_calls}, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
