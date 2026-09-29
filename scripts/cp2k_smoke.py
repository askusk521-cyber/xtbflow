#!/usr/bin/env python3
"""Run one bounded CP2K energy/force smoke calculation for xtbflow."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from xtbflow.calculators import CP2KAdapter, CP2KProtocol, MolecularSystem


def _version(executable: str) -> tuple[str, str]:
    completed = subprocess.run([executable, "-v"], capture_output=True, text=True, check=False)
    text = f"{completed.stdout}\n{completed.stderr}"
    version = re.search(r"CP2K version\s+([0-9][^\s]*)", text)
    revision = re.search(r"Source code revision\s+([0-9a-f]+)", text, re.IGNORECASE)
    return (version.group(1) if version else "unknown", revision.group(1) if revision else "unknown")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", default=os.environ.get("XTBFlow_CP2K_EXECUTABLE"), help="Path to cp2k.psmp")
    parser.add_argument("--data-dir", default=os.environ.get("XTBFlow_CP2K_DATA_DIR"), help="CP2K data directory")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cutoff", type=float, default=600.0)
    parser.add_argument("--timeout", type=float, default=180.0)
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
    version, revision = _version(executable)
    protocol = CP2KProtocol(
        protocol_id="cp2k-reference-v0.1-pbe-d3bj",
        functional="PBE",
        basis_set="DZVP-MOLOPT-GTH",
        pseudopotential="GTH-PBE",
        dispersion="DFTD3(BJ)",
        cutoff_ry=args.cutoff,
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
            "basis_by_element": {symbol: "DZVP-MOLOPT-GTH" for symbol in ("H", "C", "N", "O", "S")},
            "pseudopotential_by_element": {symbol: "GTH-PBE" for symbol in ("H", "C", "N", "O", "S")},
            "threads": args.threads,
            "executable": executable,
        },
    )
    system = MolecularSystem(
        ("O", "H", "H"),
        ((0.0, 0.0, 0.0), (0.758602, 0.0, 0.504284), (-0.758602, 0.0, 0.504284)),
        0,
        1,
        system_id="water-smoke",
    )
    result = CP2KAdapter(protocol, timeout_seconds=args.timeout).evaluate(system)
    print(json.dumps({
        "status": result.status,
        "energy_hartree": result.energy,
        "forces_hartree_per_angstrom": result.forces,
        "calculator_build_hash": result.calculator_build_hash,
        "input_file_hash": result.input_file_hash,
        "protocol_identity": protocol.identity,
        "path_status": result.path_status,
        "metadata": result.metadata,
    }, indent=2, sort_keys=True))
    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
