#!/usr/bin/env python3
"""Run bounded tblite GFN2 E/F calls for a frozen SPICE2 OpenFF pilot manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xtbflow.calculators import CalculatorProtocol, MolecularSystem, XTBOracleAdapter  # noqa: E402
from xtbflow.runtime import RunLedger, StageBudget  # noqa: E402

PROTOCOL = CalculatorProtocol(
    protocol_id="tblite-gfn2-spice2-openff-pilot-v1",
    calculator="xtb_oracle",
    method="GFN2-xTB",
    backend="cpu",
    parameters={
        "accuracy": 1.0,
        "max_iterations": 250,
        "electronic_temperature": 300.0,
        "implementation": "tblite",
        "solvent": None,
    },
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, choices=(64, 256), default=64)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != args.expected_count:
        parser.error(f"pilot manifest must contain exactly {args.expected_count} rows")
    ledger = RunLedger(StageBudget(f"P1-spice2-pilot-{args.expected_count}", max_calculator_calls=args.expected_count, max_system_evaluations=args.expected_count))
    token = ledger.issue_calculator_token(
        f"spice2-openff-pilot-{args.expected_count}",
        args.expected_count,
        metadata={"protocol_id": PROTOCOL.protocol_id, "source_rows": len(rows)},
        persist_path=args.ledger,
    )
    adapter = XTBOracleAdapter(protocol=PROTOCOL, implementation="tblite", require_budget_token=True)
    output = []
    timings = []
    success = 0
    failure = 0
    for row in rows:
        system_payload = row["system"]
        system = MolecularSystem(
            tuple(system_payload["symbols"]),
            tuple(tuple(float(value) for value in xyz) for xyz in system_payload["coordinates"]),
            int(system_payload["charge"]),
            int(system_payload["multiplicity"]),
            dict(system_payload["environment"]),
            str(system_payload["system_id"]),
        )
        started = time.perf_counter()
        result = adapter.evaluate(system, operation="energy_forces", budget_token=token)
        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        observed = {
            "source_record_id": row["source_record_id"],
            "parent_record_id": row["parent_record_id"],
            "split": row["split"],
            "system": system_payload,
            "semi_empirical_protocol_id": PROTOCOL.protocol_id,
            "reference_protocol_id": row["reference_protocol_id"],
        }
        if result.status == "success":
            success += 1
            observed["semi_empirical"] = {
                "calculator": result.calculator,
                "protocol_id": result.protocol_id,
                "input_hash": result.input_hash,
                "charge": result.charge,
                "multiplicity": result.multiplicity,
                "operation": "energy_forces",
                "status": result.status,
                "converged": result.converged,
                "energy": result.energy,
                "forces": result.forces,
                "calculator_calls": result.calculator_calls,
                "calculator_build_hash": result.calculator_build_hash,
                "metadata": dict(result.metadata),
            }
            observed["reference"] = {
                "calculator": "spice2_openff",
                "protocol_id": row["reference_protocol_id"],
                "input_hash": system.input_hash,
                "charge": system.charge,
                "multiplicity": system.multiplicity,
                "operation": "energy_forces",
                "status": "success",
                "converged": True,
                "energy": row["reference"]["energy_hartree"],
                "forces": row["reference"]["forces_hartree_per_angstrom"],
                "calculator_calls": 0,
                "calculator_build_hash": row["source_file_sha256"],
                "metadata": {
                    "source_collection": row["source_collection"],
                    "source_file_sha256": row["source_file_sha256"],
                    "source_units": row["source_units"],
                    "source_config_index": row["config_index"],
                },
            }
        else:
            failure += 1
            observed["error"] = {
                "category": result.error_category or result.status,
                "message": result.error_message or result.status,
            }
            observed["semi_empirical"] = {
                "protocol_id": result.protocol_id,
            }
            observed["reference"] = {
                "protocol_id": row["reference_protocol_id"],
            }
        output.append(observed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    token.settle(recorded_calls=token.consumed_calls, persist_path=args.ledger)
    accounting = ledger.to_dict()
    summary = {
        "schema": "xtbflow-spice2-openff-gfn2-run/v1",
        "protocol_id": PROTOCOL.protocol_id,
        "protocol_identity": PROTOCOL.identity,
        "rows": len(rows),
        "success": success,
        "failure": failure,
        "calculator_calls": len(rows),
        "wall_seconds_total": sum(timings),
        "wall_seconds_mean": sum(timings) / len(timings),
        "wall_seconds_max": max(timings),
        "ledger": str(args.ledger),
        "calculator_accounting": {
            "committed_calculator_calls": accounting["committed_calculator_calls"],
            "committed_tokens": accounting["committed_tokens"],
            "pending_reservations": accounting["reservations"],
        },
        "output": str(args.output),
        "backend_capabilities": adapter.capabilities.__dict__,
        "scope": "bounded CPU tblite pairing pilot; observed calls only, no new DFT",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if failure == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
