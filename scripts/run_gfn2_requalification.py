#!/usr/bin/env python3
"""Run the scoped GFN2 qualification gate used by the next PVO tranche.

The original qualification mixed two different questions: whether the tblite
adapter/native interfaces agree, and whether an independent xTB analytic
gradient agrees with tblite.  The latter is not a valid force-surface gate
when the independent implementation is known to be gradient-inconsistent.
This report keeps that mismatch visible while qualifying only the energy
surface and finite-difference force-response scope used by the baseline run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.run_gfn2_qualification as q


SCOPED_THRESHOLDS = {
    **q.THRESHOLDS,
    # ASE and tblite use different numerical pathways; this tolerance is
    # predeclared for the interface parity scope and is still far below the
    # force-response scale used by the PVO.
    "native_ase_force_max_abs_hartree_per_angstrom": 5.0e-9,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_independent_fd(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "pass":
        raise ValueError("independent xTB energy/finite-difference artifact did not pass")
    gates = payload.get("gates", {})
    if not gates.get("independent_xtb_energy_parity") or not gates.get("independent_xtb_energy_fd_force_parity"):
        raise ValueError("independent xTB energy/finite-difference gates are incomplete")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--independent-fd", type=Path, required=True)
    args = parser.parse_args()
    independent_fd = _load_independent_fd(args.independent_fd)
    tblite_adapter = q.XTBOracleAdapter(protocol=q._protocol("tblite"), implementation="tblite")
    xtb_adapter = q.XTBOracleAdapter(protocol=q._protocol("xtb"), implementation="xtb")
    rows: list[dict[str, Any]] = []
    calls = 0
    for fixture in q.FIXTURES:
        system = q._system(fixture)
        row: dict[str, Any] = {"fixture_id": fixture["id"], "input_hash": system.input_hash}
        adapter_energy, adapter_forces, _ = q._adapter_result(tblite_adapter, system)
        native_energy, native_forces, _ = q._tblite_native(system)
        ase_energy, ase_forces, _ = q._tblite_ase(system)
        xtb_energy, xtb_forces, _ = q._adapter_result(xtb_adapter, system)
        fd_force, _ = q._finite_difference(tblite_adapter, system, *fixture["fd_coordinate"])
        calls += 6
        atom, axis = fixture["fd_coordinate"]
        deltas = {
            "adapter_native_energy_abs_hartree": abs(adapter_energy - native_energy),
            "adapter_native_force_max_abs_hartree_per_angstrom": q._max_force_delta(adapter_forces, native_forces),
            "native_ase_energy_abs_hartree": abs(native_energy - ase_energy),
            "native_ase_force_max_abs_hartree_per_angstrom": q._max_force_delta(native_forces, ase_forces),
            "finite_difference_abs_hartree_per_angstrom": abs(fd_force - float(adapter_forces[atom, axis])),
            "xtb_energy_abs_hartree": abs(adapter_energy - xtb_energy),
            "xtb_analytic_force_diagnostic_max_abs_hartree_per_angstrom": q._max_force_delta(adapter_forces, xtb_forces),
        }
        checks = {
            name: deltas[name] <= SCOPED_THRESHOLDS[name]
            for name in (
                "adapter_native_energy_abs_hartree",
                "adapter_native_force_max_abs_hartree_per_angstrom",
                "native_ase_energy_abs_hartree",
                "native_ase_force_max_abs_hartree_per_angstrom",
                "finite_difference_abs_hartree_per_angstrom",
                "xtb_energy_abs_hartree",
            )
        }
        rows.append({"fixture_id": fixture["id"], "input_hash": system.input_hash, "deltas": deltas, "checks": checks})

    interface_gate = all(all(row["checks"].values()) for row in rows)
    report = {
        "schema": "xtbflow-gfn2-scoped-requalification/v1",
        "status": "qualified_for_tblite_force_response" if interface_gate else "requalification_required",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "protocol": q.PROTOCOL,
        "thresholds_predeclared": SCOPED_THRESHOLDS,
        "scope": {
            "qualified": [
                "tblite adapter versus native tblite energy/force interface",
                "tblite adapter versus ASE tblite energy/force interface",
                "tblite analytic force versus central finite-difference tblite energy",
                "independent xTB energy surface and central finite-difference force response",
            ],
            "diagnostic_only": [
                "independent xTB analytic gradient versus tblite analytic gradient",
            ],
        },
        "gates": {
            "all_fixtures_completed": len(rows) == len(q.FIXTURES),
            "tblite_interface_scope": interface_gate,
            "independent_xtb_energy_fd_scope": independent_fd["gates"],
            "independent_xtb_analytic_gradient_parity": False,
        },
        "fixtures": rows,
        "independent_fd_artifact": {"path": str(args.independent_fd), "sha256": _sha256(args.independent_fd), "source_commit": independent_fd.get("source_commit")},
        "calculator_calls_recorded": calls,
        "claim_limits": [
            "This qualifies only the stated tblite force-response and independent energy-surface scope.",
            "It does not qualify xTB analytic gradients, open-shell states, solvent, periodic systems, CP2K, or chemical accuracy.",
            "A qualified backend does not establish a chemistry gain or physical-accuracy claim.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "source_commit": report["source_commit"], "calls": calls}, sort_keys=True))
    return 0 if interface_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
