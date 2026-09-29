#!/usr/bin/env python3
"""Run a metered, persistent CP2K cutoff-convergence ladder."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xtbflow.calculators import (  # noqa: E402
    CP2KAdapter,
    MolecularSystem,
    build_cp2k_protocol,
    inspect_cp2k_runtime,
    load_cp2k_protocol_document,
    physical_protocol_identity,
    physical_protocol_record,
    protocol_document_sha256,
)
from xtbflow.runtime import RunLedger, StageBudget  # noqa: E402

FIXTURE = {
    "system_id": "water-perturbed-cutoff-ladder",
    "symbols": ("O", "H", "H"),
    "coordinates": ((0.0, 0.0, 0.0), (0.97, 0.03, 0.0), (-0.25, 0.91, 0.04)),
    "charge": 0,
    "multiplicity": 1,
}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _document_for_cutoff(
    document: Mapping[str, Any], cutoff: float
) -> dict[str, Any]:
    payload = json.loads(json.dumps(document, allow_nan=False))
    protocol = payload["protocol"]
    protocol["cutoff_ry"] = float(cutoff)
    protocol["protocol_id"] = (
        f"cp2k-reference-v0.1-pbe-d3bj-cutoff-{float(cutoff):g}"
    )
    return payload


def _max_force_delta(left: Any, right: Any) -> float:
    return max(
        abs(float(a) - float(b))
        for row_a, row_b in zip(left, right)
        for a, b in zip(row_a, row_b)
    )


def _comparison(run: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cutoff_ry": run["cutoff_ry"],
        "reference_cutoff_ry": reference["cutoff_ry"],
        "abs_energy_delta_hartree": abs(
            float(run["energy_hartree"]) - float(reference["energy_hartree"])
        ),
        "max_force_delta_hartree_per_angstrom": _max_force_delta(
            run["forces_hartree_per_angstrom"],
            reference["forces_hartree_per_angstrom"],
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/calculators/cp2k/protocol_v0.1.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--cp2k-executable", default=os.environ.get("XTBFlow_CP2K_EXECUTABLE")
    )
    parser.add_argument(
        "--cp2k-data-dir", default=os.environ.get("XTBFlow_CP2K_DATA_DIR")
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--cutoffs", type=float, nargs="+", default=(500.0, 600.0, 700.0)
    )
    parser.add_argument("--energy-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--force-tolerance", type=float, default=1.0e-4)
    args = parser.parse_args()

    if not args.cp2k_executable:
        parser.error("set --cp2k-executable or XTBFlow_CP2K_EXECUTABLE")
    if args.threads < 1 or args.timeout <= 0:
        parser.error("threads and timeout must be positive")
    if len(args.cutoffs) < 2 or any(
        not math.isfinite(value) or value <= 0 for value in args.cutoffs
    ):
        parser.error("at least two positive finite cutoffs are required")
    if list(args.cutoffs) != sorted(set(args.cutoffs)):
        parser.error("cutoffs must be unique and strictly increasing")
    if args.energy_tolerance <= 0 or args.force_tolerance <= 0:
        parser.error("energy and force tolerances must be positive")
    if args.ledger.exists():
        parser.error(f"refusing to reuse an existing convergence ledger: {args.ledger}")

    protocol_path = args.protocol.expanduser().resolve()
    document = load_cp2k_protocol_document(protocol_path)
    runtime = inspect_cp2k_runtime(
        args.cp2k_executable, data_dir=args.cp2k_data_dir
    )
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    system = MolecularSystem(
        tuple(FIXTURE["symbols"]),
        tuple(FIXTURE["coordinates"]),
        int(FIXTURE["charge"]),
        int(FIXTURE["multiplicity"]),
        {"periodic": False, "qualification_fixture": FIXTURE["system_id"]},
        str(FIXTURE["system_id"]),
    )

    ledger = RunLedger(
        StageBudget(
            "P2a-cp2k-cutoff-convergence",
            max_calculator_calls=len(args.cutoffs),
            max_concurrent_jobs=1,
            max_retries_per_job=0,
        )
    )
    token = ledger.issue_calculator_token(
        "c2-cp2k-cutoff-ladder",
        len(args.cutoffs),
        metadata={
            "system_id": system.system_id,
            "cutoffs_ry": list(args.cutoffs),
        },
        persist_path=args.ledger,
    )

    runs: list[dict[str, Any]] = []
    artifact_root = args.artifact_dir.expanduser().resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    for cutoff in args.cutoffs:
        calls_before = token.consumed_calls
        cutoff_document = _document_for_cutoff(document, cutoff)
        protocol_id = cutoff_document["protocol"]["protocol_id"]
        protocol = build_cp2k_protocol(
            cutoff_document,
            runtime,
            charge=system.charge,
            multiplicity=system.multiplicity,
            protocol_id=protocol_id,
        )
        started = time.monotonic()
        try:
            result = CP2KAdapter(
                protocol,
                executable=runtime.executable,
                timeout_seconds=args.timeout,
                require_budget_token=True,
                artifact_dir=artifact_root / f"cutoff-{cutoff:g}",
                require_artifacts=True,
            ).evaluate(system, operation="energy_forces", budget_token=token)
        except Exception as exc:
            runs.append(
                {
                    "cutoff_ry": cutoff,
                    "protocol_id": protocol.protocol_id,
                    "physical_protocol_identity": physical_protocol_identity(
                        cutoff_document,
                        runtime,
                        charge=system.charge,
                        multiplicity=system.multiplicity,
                        protocol_id=protocol.protocol_id,
                    ),
                    "status": "failure",
                    "calculator_calls": token.consumed_calls - calls_before,
                    "wall_seconds": round(time.monotonic() - started, 3),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            continue
        runs.append(
            {
                "cutoff_ry": cutoff,
                "relative_cutoff_ry": protocol.relative_cutoff_ry,
                "protocol_id": protocol.protocol_id,
                "physical_protocol": physical_protocol_record(
                    cutoff_document,
                    runtime,
                    charge=system.charge,
                    multiplicity=system.multiplicity,
                    protocol_id=protocol.protocol_id,
                ),
                "physical_protocol_identity": physical_protocol_identity(
                    cutoff_document,
                    runtime,
                    charge=system.charge,
                    multiplicity=system.multiplicity,
                    protocol_id=protocol.protocol_id,
                ),
                "runtime_bound_protocol_identity": protocol.identity,
                "status": result.status,
                "energy_hartree": result.energy,
                "forces_hartree_per_angstrom": result.forces,
                "input_file_hash": result.input_file_hash,
                "metadata": result.metadata,
                "calculator_calls": result.calculator_calls,
                "wall_seconds": round(time.monotonic() - started, 3),
            }
        )
    token.settle(recorded_calls=token.consumed_calls, persist_path=args.ledger)

    successful = [run for run in runs if run["status"] == "success"]
    reference = successful[-1] if successful else None
    comparisons = (
        [_comparison(run, reference) for run in successful]
        if reference is not None
        else []
    )
    thresholds_pass = bool(comparisons) and all(
        item["abs_energy_delta_hartree"] <= args.energy_tolerance
        and item["max_force_delta_hartree_per_angstrom"]
        <= args.force_tolerance
        for item in comparisons
    )
    passed = len(successful) == len(runs) and thresholds_pass
    report = {
        "schema": "xtbflow-cp2k-convergence-smoke/v2",
        "status": "pass" if passed else "fail",
        "scientific_qualification": False,
        "source_commit": _source_commit(),
        "script_sha256": protocol_document_sha256(Path(__file__)),
        "protocol_document_sha256": protocol_document_sha256(protocol_path),
        "runtime": runtime.public_record(),
        "system": {
            "system_id": system.system_id,
            "symbols": list(system.symbols),
            "coordinates_angstrom": [list(row) for row in system.coordinates],
            "charge": system.charge,
            "multiplicity": system.multiplicity,
            "input_hash": system.input_hash,
        },
        "runtime_threads": args.threads,
        "energy_tolerance_hartree": args.energy_tolerance,
        "force_tolerance_hartree_per_angstrom": args.force_tolerance,
        "cutoffs_ry": list(args.cutoffs),
        "artifact_root": str(artifact_root),
        "runs": runs,
        "comparisons_to_highest_cutoff": comparisons,
        "calculator_calls": ledger.committed_calculator_calls,
        "scope": (
            "one non-equilibrium water geometry; cutoff convergence only, "
            "not basis, SCF, TS, path, or broad chemical qualification"
        ),
        "claim_limits": [
            "The ladder is one fixed geometry and one CP2K build only.",
            "Failure is a valid gate result and blocks promotion of the provisional cutoff.",
            "Passing would not establish basis convergence, broad chemical accuracy, transition-state validity, or endpoint connectivity.",
        ],
    }
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "calculator_calls": report["calculator_calls"],
                "successful_runs": len(successful),
                "reference_cutoff_ry": reference["cutoff_ry"] if reference else None,
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
