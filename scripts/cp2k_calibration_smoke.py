#!/usr/bin/env python3
"""Run a bounded CP2K state/element calibration smoke suite for xtbflow."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from xtbflow.calculators import CP2KAdapter, CP2KProtocol, MolecularSystem


def _version(executable: str) -> tuple[str, str]:
    completed = subprocess.run([executable, "-v"], capture_output=True, text=True, check=False)
    text = f"{completed.stdout}\n{completed.stderr}"
    version = re.search(r"CP2K version\s+([0-9][^\s]*)", text)
    revision = re.search(r"Source code revision\s+([0-9a-f]+)", text, re.IGNORECASE)
    return (version.group(1) if version else "unknown", revision.group(1) if revision else "unknown")


def _cases() -> tuple[MolecularSystem, ...]:
    return (
        MolecularSystem(
            ("O", "H", "H"),
            ((0.0, 0.0, 0.0), (0.758602, 0.0, 0.504284), (-0.758602, 0.0, 0.504284)),
            0,
            1,
            system_id="water-neutral-singlet",
        ),
        MolecularSystem(
            ("H", "H"),
            ((0.0, 0.0, 0.0), (0.74, 0.0, 0.0)),
            1,
            2,
            system_id="hydrogen-cation-doublet",
        ),
        MolecularSystem(
            ("S", "H", "H"),
            ((0.0, 0.0, 0.0), (1.34, 0.0, 0.0), (-0.45, 1.26, 0.0)),
            0,
            1,
            system_id="hydrogen-sulfide-neutral-singlet",
        ),
    )


def _protocol(data_dir: Path, executable: str, version: str, revision: str, threads: int, cutoff: float) -> CP2KProtocol:
    elements = ("H", "C", "N", "O", "S")
    return CP2KProtocol(
        protocol_id="cp2k-reference-v0.1-pbe-d3bj",
        functional="PBE",
        basis_set="DZVP-MOLOPT-GTH",
        pseudopotential="GTH-PBE",
        dispersion="DFTD3(BJ)",
        cutoff_ry=cutoff,
        relative_cutoff_ry=50.0,
        scf_epsilon=1e-7,
        max_scf=80,
        charge=0,
        multiplicity=1,
        build_hash=f"cp2k-{version}-{revision}",
        cp2k_version=version,
        parameters={
            "basis_set_file": str(data_dir / "BASIS_MOLOPT"),
            "pseudopotential_file": str(data_dir / "GTH_POTENTIALS"),
            "dispersion_parameter_file": str(data_dir / "dftd3.dat"),
            "reference_functional": "PBE",
            "basis_by_element": {element: "DZVP-MOLOPT-GTH" for element in elements},
            "pseudopotential_by_element": {element: "GTH-PBE" for element in elements},
        },
    )


def _result(system: MolecularSystem, protocol: CP2KProtocol, result: Any) -> dict[str, Any]:
    return {
        "system_id": system.system_id,
        "symbols": list(system.symbols),
        "charge": system.charge,
        "multiplicity": system.multiplicity,
        "input_hash": system.input_hash,
        "protocol_id": protocol.protocol_id,
        "protocol_identity": protocol.identity,
        "status": result.status,
        "energy_hartree": result.energy,
        "forces_hartree_per_angstrom": result.forces,
        "calculator_calls": result.calculator_calls,
        "error_category": result.error_category,
        "error_message": result.error_message,
        "input_file_hash": result.input_file_hash,
        "metadata": result.metadata,
    }


def _failure(system: MolecularSystem, protocol: CP2KProtocol, error: Exception) -> dict[str, Any]:
    return {
        "system_id": system.system_id,
        "symbols": list(system.symbols),
        "charge": system.charge,
        "multiplicity": system.multiplicity,
        "input_hash": system.input_hash,
        "protocol_id": protocol.protocol_id,
        "protocol_identity": protocol.identity,
        "status": "failure",
        "energy_hartree": None,
        "forces_hartree_per_angstrom": None,
        "calculator_calls": 1,
        "error_category": getattr(error, "category", "execution"),
        "error_message": f"{type(error).__name__}: {error}",
        "input_file_hash": None,
        "metadata": {},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", default=os.environ.get("XTBFlow_CP2K_EXECUTABLE"))
    parser.add_argument("--data-dir", default=os.environ.get("XTBFlow_CP2K_DATA_DIR"))
    parser.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    parser.add_argument("--cutoff", type=float, default=600.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(os.environ.get("XTBFlow_CP2K_ARTIFACT_DIR", "runs/cp2k-artifacts/calibration")),
        help="Persistent root for input/stdout/stderr/output artifacts.",
    )
    args = parser.parse_args()
    executable = args.executable or shutil.which("cp2k.psmp") or shutil.which("cp2k.popt")
    if not executable:
        parser.error("CP2K executable not found; set XTBFlow_CP2K_EXECUTABLE")
    executable = str(Path(executable).expanduser().resolve())
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else Path(executable).parent.parent / "share" / "cp2k" / "data"
    required = [data_dir / "BASIS_MOLOPT", data_dir / "GTH_POTENTIALS", data_dir / "dftd3.dat"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("missing CP2K data files: " + ", ".join(missing))
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    version, revision = _version(executable)
    base = _protocol(data_dir, executable, version, revision, args.threads, args.cutoff)
    records: list[dict[str, Any]] = []
    for system in _cases():
        protocol = base.for_state(system.charge, system.multiplicity)
        try:
            result = CP2KAdapter(
                protocol,
                executable=executable,
                timeout_seconds=args.timeout,
                artifact_dir=args.artifact_dir / system.system_id,
                require_artifacts=True,
            ).evaluate(system)
        except Exception as error:
            records.append(_failure(system, protocol, error))
        else:
            records.append(_result(system, protocol, result))
    payload = {
        "schema": "xtbflow-cp2k-calibration-smoke/v1",
        "status": "pass" if all(record["status"] == "success" for record in records) else "fail",
        "calculator": "cp2k",
        "cp2k_version": version,
        "source_revision": revision,
        "build_hash": base.build_hash,
        "protocol_family": base.protocol_id,
        "cutoff_ry": base.cutoff_ry,
        "runtime_threads": args.threads,
        "artifact_root": str(args.artifact_dir),
        "cases": records,
        "scope": "installation and state/element E/F smoke only; not convergence or path qualification",
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
