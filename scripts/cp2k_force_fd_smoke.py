#!/usr/bin/env python3
"""Validate one CP2K analytic force component by central finite difference."""
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
    "system_id": "water-perturbed-force-fd",
    "symbols": ("O", "H", "H"),
    "coordinates": ((0.0, 0.0, 0.0), (0.97, 0.03, 0.0), (-0.25, 0.91, 0.04)),
    "charge": 0,
    "multiplicity": 1,
    "atom_index": 1,
    "axis": 0,
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


def _public_result(result: Any) -> dict[str, Any]:
    payload = asdict(result)
    metadata = dict(payload.get("metadata", {}))
    metadata.pop("artifact_directory", None)
    metadata["artifact_directory_recorded_in_private_run"] = True
    payload["metadata"] = metadata
    return payload


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


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
    parser.add_argument("--step", type=float, default=1.0e-3)
    parser.add_argument("--threshold", type=float, default=5.0e-4)
    args = parser.parse_args()
    if not args.cp2k_executable:
        parser.error("set --cp2k-executable or XTBFlow_CP2K_EXECUTABLE")
    if args.threads < 1 or args.timeout <= 0:
        parser.error("threads and timeout must be positive")
    if not math.isfinite(args.step) or args.step <= 0:
        parser.error("step must be finite and positive")
    if not math.isfinite(args.threshold) or args.threshold <= 0:
        parser.error("threshold must be finite and positive")
    if args.ledger.exists():
        parser.error(f"refusing to reuse an existing force-FD ledger: {args.ledger}")

    protocol_path = args.protocol.expanduser().resolve()
    document = load_cp2k_protocol_document(protocol_path)
    runtime = inspect_cp2k_runtime(
        args.cp2k_executable, data_dir=args.cp2k_data_dir
    )
    protocol = build_cp2k_protocol(
        document,
        runtime,
        charge=int(FIXTURE["charge"]),
        multiplicity=int(FIXTURE["multiplicity"]),
        protocol_id="cp2k-reference-v0.1-force-fd",
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

    atom = int(FIXTURE["atom_index"])
    axis = int(FIXTURE["axis"])
    coordinates = [list(row) for row in system.coordinates]
    plus_coordinates = [row[:] for row in coordinates]
    minus_coordinates = [row[:] for row in coordinates]
    plus_coordinates[atom][axis] += args.step
    minus_coordinates[atom][axis] -= args.step
    plus_system = system.with_coordinates(tuple(tuple(row) for row in plus_coordinates))
    minus_system = system.with_coordinates(tuple(tuple(row) for row in minus_coordinates))

    ledger = RunLedger(
        StageBudget(
            "P2a-cp2k-force-fd",
            max_calculator_calls=3,
            max_concurrent_jobs=1,
            max_retries_per_job=0,
        )
    )
    token = ledger.issue_calculator_token(
        "cp2k-force-fd-water",
        3,
        metadata={"protocol_id": protocol.protocol_id, "step_angstrom": args.step},
        persist_path=args.ledger,
    )
    adapter = CP2KAdapter(
        protocol,
        executable=runtime.executable,
        timeout_seconds=args.timeout,
        require_budget_token=True,
        artifact_dir=args.artifact_dir.expanduser().resolve(),
        require_artifacts=True,
    )

    error: dict[str, str] | None = None
    baseline = plus = minus = None
    try:
        baseline = adapter.evaluate(
            system, operation="energy_forces", budget_token=token
        )
        plus = adapter.evaluate(plus_system, operation="energy", budget_token=token)
        minus = adapter.evaluate(minus_system, operation="energy", budget_token=token)
    except Exception as exc:
        error = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    finally:
        token.settle(recorded_calls=token.consumed_calls, persist_path=args.ledger)

    passed = False
    analytic_force = finite_difference_force = absolute_error = None
    if error is None:
        if baseline is None or plus is None or minus is None:
            error = {
                "error_type": "RuntimeError",
                "error_message": "one or more CP2K results are missing",
            }
        elif baseline.status != "success" or plus.status != "success" or minus.status != "success":
            error = {
                "error_type": "RuntimeError",
                "error_message": "one or more CP2K calls did not return success",
            }
        elif baseline.forces is None or plus.energy is None or minus.energy is None:
            error = {
                "error_type": "RuntimeError",
                "error_message": "CP2K finite-difference inputs are incomplete",
            }
        else:
            analytic_force = float(baseline.forces[atom][axis])
            finite_difference_force = -float(plus.energy - minus.energy) / (2.0 * args.step)
            absolute_error = abs(analytic_force - finite_difference_force)
            passed = absolute_error <= args.threshold

    report = {
        "schema": "xtbflow-cp2k-force-finite-difference/v1",
        "status": "passed" if passed else "failed",
        "scientific_qualification": False,
        "source_commit": _source_commit(),
        "script_sha256": protocol_document_sha256(Path(__file__)),
        "protocol_document_sha256": protocol_document_sha256(protocol_path),
        "protocol_id": protocol.protocol_id,
        "physical_protocol": physical_protocol_record(
            document,
            runtime,
            charge=system.charge,
            multiplicity=system.multiplicity,
            protocol_id=protocol.protocol_id,
        ),
        "physical_protocol_identity": physical_protocol_identity(
            document,
            runtime,
            charge=system.charge,
            multiplicity=system.multiplicity,
            protocol_id=protocol.protocol_id,
        ),
        "runtime_bound_protocol_identity": protocol.identity,
        "runtime": runtime.public_record(),
        "fixture": {
            "system_id": system.system_id,
            "symbols": list(system.symbols),
            "coordinates_angstrom": [list(row) for row in system.coordinates],
            "charge": system.charge,
            "multiplicity": system.multiplicity,
            "input_hash": system.input_hash,
            "atom_index": atom,
            "axis": axis,
        },
        "finite_difference_step_angstrom": args.step,
        "threshold_abs_hartree_per_angstrom": args.threshold,
        "analytic_force_hartree_per_angstrom": analytic_force,
        "finite_difference_force_hartree_per_angstrom": finite_difference_force,
        "absolute_error_hartree_per_angstrom": absolute_error,
        "calculator_calls": ledger.committed_calculator_calls,
        "error": error,
        "results": {
            "baseline": _public_result(baseline) if baseline is not None else None,
            "plus": _public_result(plus) if plus is not None else None,
            "minus": _public_result(minus) if minus is not None else None,
        },
        "claim_limits": [
            "Passing checks one force component, one geometry, and one frozen CP2K runtime only.",
            "This does not establish basis/cutoff convergence, broad chemical accuracy, or reference-label admission.",
            "No transition-state, mode, path, endpoint-connectivity, or mechanism result is claimed.",
        ],
    }
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "absolute_error_hartree_per_angstrom": absolute_error,
                "calculator_calls": ledger.committed_calculator_calls,
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
