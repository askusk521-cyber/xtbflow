#!/usr/bin/env python3
"""Run one bounded real GFN2 ASE-dimer development pilot."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xtbflow.calculators import (  # noqa: E402
    CalculatorProtocol,
    MolecularSystem,
    XTBOracleAdapter,
)
from xtbflow.runtime import RunLedger, StageBudget  # noqa: E402
from xtbflow.validation.ase_dimer import (  # noqa: E402
    ASEDimerConfig,
    run_ase_dimer_search,
    run_dimer_preflight,
)
from xtbflow.validation.evidence import index_artifacts, sha256_file  # noqa: E402


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
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


def _protocol(payload: Mapping[str, Any]) -> CalculatorProtocol:
    return CalculatorProtocol(
        protocol_id=str(payload["protocol_id"]),
        calculator=str(payload["calculator"]),
        method=str(payload["method"]),
        backend=str(payload["backend"]),
        parameters=dict(payload["parameters"]),
    )


def _system(payload: Mapping[str, Any]) -> MolecularSystem:
    return MolecularSystem(
        tuple(str(item) for item in payload["symbols"]),
        tuple(
            tuple(float(value) for value in row)
            for row in payload["coordinates_angstrom"]
        ),
        int(payload["charge"]),
        int(payload["multiplicity"]),
        dict(payload["environment"]),
        str(payload["system_id"]),
    )


def _search_config(payload: Mapping[str, Any]) -> ASEDimerConfig:
    return ASEDimerConfig(
        fmax_eV_per_angstrom=float(payload["fmax_eV_per_angstrom"]),
        max_steps=int(payload["max_steps"]),
        dimer_separation_angstrom=float(payload["dimer_separation_angstrom"]),
        max_num_rot=int(payload["max_num_rot"]),
        maximum_translation_angstrom=float(
            payload["maximum_translation_angstrom"]
        ),
        trial_translation_step_angstrom=float(
            payload["trial_translation_step_angstrom"]
        ),
        random_seed=int(payload["random_seed"]),
        remove_rigid_body_modes=payload["remove_rigid_body_modes"],
        rigid_body_tolerance=float(payload["rigid_body_tolerance"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/validation/gfn2_dimer_pilot_v0.1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument(
        "--preflight-ledger",
        type=Path,
        help="separate durable ledger for an optional force/curvature preflight",
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()

    document = json.loads(args.config.read_text(encoding="utf-8"))
    if document.get("schema") != "xtbflow-gfn2-dimer-pilot/v1":
        parser.error("unsupported pilot config schema")

    preflight_document = document.get("preflight")
    preflight_ledger_path: Path | None = None
    if preflight_document is not None:
        suffix = args.ledger.suffix or ".json"
        preflight_ledger_path = args.preflight_ledger or args.ledger.with_name(
            f"{args.ledger.stem}.preflight{suffix}"
        )
    elif args.preflight_ledger is not None:
        parser.error("--preflight-ledger requires a config with a preflight block")

    guarded_paths = [args.output, args.ledger]
    if preflight_ledger_path is not None:
        guarded_paths.append(preflight_ledger_path)
    if len({str(item.expanduser().resolve()) for item in guarded_paths}) != len(
        guarded_paths
    ):
        parser.error("output, dimer ledger, and preflight ledger must be distinct")
    if any(item.exists() for item in guarded_paths):
        parser.error("refusing to overwrite an existing output or ledger")
    if args.artifact_dir.exists() and any(args.artifact_dir.iterdir()):
        parser.error("refusing to mix with a nonempty artifact directory")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    protocol = _protocol(document["protocol"])
    system = _system(document["system"])
    search = _search_config(document["search"])
    adapter = XTBOracleAdapter(
        protocol=protocol,
        implementation="tblite",
        require_budget_token=True,
    )
    source_commit = _source_commit()
    report: dict[str, Any] = {
        "schema": "xtbflow-gfn2-dimer-pilot-result/v1",
        "status": "initializing",
        "execution_stage": "preflight" if preflight_document is not None else "dimer",
        "preflight_gate_status": (
            "pending" if preflight_document is not None else "not_configured"
        ),
        "search_status": "not_started",
        "search_execution": {
            "status": "not_started",
            "initial_guess_eligible": False,
        },
        "scientific_qualification": False,
        "source_commit": source_commit,
        "config_sha256": sha256_file(args.config),
        "script_sha256": sha256_file(Path(__file__)),
        "ase_dimer_driver_sha256": sha256_file(
            ROOT / "src/xtbflow/validation/ase_dimer.py"
        ),
        "xtb_oracle_adapter_sha256": sha256_file(
            ROOT / "src/xtbflow/calculators/xtb_oracle.py"
        ),
        "protocol": {
            "protocol_id": protocol.protocol_id,
            "protocol_identity": protocol.identity,
            "method": protocol.method,
            "backend": protocol.backend,
            "parameters": dict(protocol.parameters),
        },
        "runtime": {
            "implementation": "tblite",
            "version": adapter.version,
            "build_hash": adapter.build_hash,
            "capabilities": asdict(adapter.capabilities),
        },
        "input": {
            "system": document["system"],
            "input_hash": system.input_hash,
            "electronic_state": {
                "charge": system.charge,
                "multiplicity": system.multiplicity,
                "source": "frozen_config",
            },
            "initial_mode": document["initial_mode"],
            "search": document["search"],
            "preflight": preflight_document,
            "gate": document["gate"],
        },
        "preflight": None,
        "driver_result": None,
        "artifact_index": [],
        "calculator_accounting": None,
        "claim_limits": list(document["claim_limits"]),
    }

    if preflight_document is not None:
        assert preflight_ledger_path is not None
        preflight_budget = preflight_document["budget"]
        preflight_ledger = RunLedger(
            StageBudget(
                str(preflight_budget["phase"]),
                max_calculator_calls=int(
                    preflight_budget["max_calculator_calls"]
                ),
                max_concurrent_jobs=int(
                    preflight_budget["max_concurrent_jobs"]
                ),
                max_retries_per_job=int(
                    preflight_budget["max_retries_per_job"]
                ),
            )
        )
        preflight_token = preflight_ledger.issue_calculator_token(
            "gfn2-dimer-preflight",
            int(preflight_budget["max_calculator_calls"]),
            metadata={
                "protocol_id": protocol.protocol_id,
                "system_id": system.system_id,
                "driver_config_identity": search.identity,
                "hvp_step_angstrom": float(
                    preflight_document["hvp_step_angstrom"]
                ),
            },
            persist_path=preflight_ledger_path,
        )
        try:
            preflight_result = run_dimer_preflight(
                adapter,
                system,
                document["initial_mode"],
                budget_token=preflight_token,
                step_angstrom=float(preflight_document["hvp_step_angstrom"]),
                config=search,
                require_durable_token=True,
            )
            preflight_result_payload: dict[str, Any] = asdict(preflight_result)
        except Exception as error:
            preflight_result_payload = {
                "status": "failure",
                "calculator_calls": preflight_token.consumed_calls,
                "error": f"{type(error).__name__}: {error}",
            }

        preflight_gate = preflight_document["gate"]
        curvature = preflight_result_payload.get(
            "mode_curvature_eV_per_angstrom2"
        )
        force_norm = preflight_result_payload.get(
            "projected_force_norm_hartree_per_angstrom"
        )
        negative_curvature = curvature is not None and float(curvature) < 0.0
        require_negative = bool(
            preflight_gate["require_negative_mode_curvature"]
        )
        force_limit = float(
            preflight_gate[
                "max_projected_force_norm_hartree_per_angstrom"
            ]
        )
        force_within_limit = (
            force_norm is not None and float(force_norm) <= force_limit
        )
        preflight_pass = (
            preflight_result_payload.get("status") == "success"
            and (negative_curvature or not require_negative)
            and force_within_limit
        )
        preflight_record: dict[str, Any] = {
            "status": "pending_settlement",
            "result": preflight_result_payload,
            "gate_evaluation": {
                "calculation_success": (
                    preflight_result_payload.get("status") == "success"
                ),
                "require_negative_mode_curvature": require_negative,
                "negative_mode_curvature": negative_curvature,
                "mode_curvature_eV_per_angstrom2": curvature,
                "projected_force_norm_hartree_per_angstrom": force_norm,
                "max_projected_force_norm_hartree_per_angstrom": force_limit,
                "projected_force_within_limit": force_within_limit,
                "preflight_gate_pass": preflight_pass,
            },
            "calculator_accounting": {
                "reserved_calls": preflight_token.reserved_calls,
                "consumed_calls": preflight_token.consumed_calls,
                "settled": False,
            },
        }
        preflight_artifact = args.artifact_dir / "preflight.json"
        _atomic_json(preflight_artifact, preflight_record)
        preflight_token.settle(
            recorded_calls=preflight_token.consumed_calls,
            persist_path=preflight_ledger_path,
        )
        preflight_accounting = preflight_ledger.to_dict()
        preflight_record["status"] = "pass" if preflight_pass else "fail"
        preflight_record["candidate_decision"] = (
            "accepted_for_dimer_search" if preflight_pass else "rejected"
        )
        preflight_record["ledger_sha256"] = sha256_file(
            preflight_ledger_path
        )
        preflight_record["calculator_accounting"] = {
            "committed_calculator_calls": preflight_accounting[
                "committed_calculator_calls"
            ],
            "pending_reservations": preflight_accounting["reservations"],
            "committed_tokens": preflight_accounting["committed_tokens"],
            "settled": True,
        }
        _atomic_json(preflight_artifact, preflight_record)
        report["preflight"] = preflight_record
        report["preflight_gate_status"] = "passed" if preflight_pass else "rejected"

        if not preflight_pass:
            report["status"] = "fail"
            report["execution_stage"] = "preflight"
            report["search_status"] = "not_executed"
            report["search_execution"] = {
                "status": "not_executed",
                "reason": "preflight_rejected",
                "initial_guess_eligible": False,
            }
            report["artifact_index"] = index_artifacts(args.artifact_dir)
            report["calculator_accounting"] = {
                "dimer_executed": False,
                "dimer_committed_calculator_calls": 0,
                "preflight_committed_calculator_calls": preflight_accounting[
                    "committed_calculator_calls"
                ],
            }
            report["gate_evaluation"] = {
                "preflight_gate_pass": False,
                "driver_gate_pass": False,
                "full_ts_validation": False,
            }
            report["claim_limits"].append(
                "The dimer search was not started because the separately metered preflight gate failed."
            )
            report["claim_limits"].append(
                "A preflight rejection is not evidence that a complete TS search failed, and this candidate is not an approved dimer initial guess."
            )
            _atomic_json(args.output, report)
            print(
                json.dumps(
                    {
                        "status": "fail",
                        "execution_stage": "preflight",
                        "negative_mode_curvature": negative_curvature,
                        "curvature_eV_per_angstrom2": curvature,
                        "projected_force_norm_hartree_per_angstrom": force_norm,
                        "calculator_calls": preflight_accounting[
                            "committed_calculator_calls"
                        ],
                    },
                    sort_keys=True,
                )
            )
            return 2

    budget = document["budget"]
    ledger = RunLedger(
        StageBudget(
            str(budget["phase"]),
            max_calculator_calls=int(budget["max_calculator_calls"]),
            max_concurrent_jobs=int(budget["max_concurrent_jobs"]),
            max_retries_per_job=int(budget["max_retries_per_job"]),
        )
    )
    token = ledger.issue_calculator_token(
        "gfn2-dimer-development-pilot",
        int(budget["max_calculator_calls"]),
        metadata={
            "protocol_id": protocol.protocol_id,
            "system_id": system.system_id,
            "driver_config_identity": search.identity,
        },
        persist_path=args.ledger,
    )
    result = run_ase_dimer_search(
        adapter,
        system,
        document["initial_mode"],
        budget_token=token,
        artifact_dir=args.artifact_dir,
        config=search,
        require_durable_token=True,
    )
    report["status"] = "pending_settlement"
    report["execution_stage"] = "dimer"
    report["search_status"] = "running"
    report["search_execution"] = {
        "status": "executed",
        "initial_guess_eligible": True,
        "eligibility_basis": "preflight_passed" if report["preflight"] is not None else "preflight_not_configured",
    }
    report["driver_result"] = asdict(result)
    report["artifact_index"] = index_artifacts(args.artifact_dir)
    report["calculator_accounting"] = {
        "reserved_calls": token.reserved_calls,
        "consumed_calls": token.consumed_calls,
        "settled": False,
    }
    _atomic_json(args.output, report)
    token.settle(
        recorded_calls=token.consumed_calls,
        persist_path=args.ledger,
    )

    gate = document["gate"]
    gradient = result.gradient_norm_hartree_per_angstrom
    curvature = result.curvature_eV_per_angstrom2
    gate_pass = (
        result.converged
        and curvature is not None
        and curvature < 0.0
        and gradient is not None
        and gradient <= float(gate["max_gradient_norm_hartree_per_angstrom"])
    )
    accounting = ledger.to_dict()
    preflight_calls = (
        0
        if report["preflight"] is None
        else int(
            report["preflight"]["calculator_accounting"][
                "committed_calculator_calls"
            ]
        )
    )
    dimer_calls = int(accounting["committed_calculator_calls"])
    report["status"] = "pass" if gate_pass else "fail"
    report["search_status"] = "completed_pass" if gate_pass else "completed_fail"
    report["ledger_sha256"] = sha256_file(args.ledger)
    report["calculator_accounting"] = {
        "committed_calculator_calls": dimer_calls,
        "preflight_committed_calculator_calls": preflight_calls,
        "total_committed_calculator_calls": preflight_calls + dimer_calls,
        "pending_reservations": accounting["reservations"],
        "committed_tokens": accounting["committed_tokens"],
        "settled": True,
    }
    report["gate_evaluation"] = {
        "preflight_gate_pass": (
            report["preflight"] is None
            or report["preflight"]["gate_evaluation"]["preflight_gate_pass"]
        ),
        "optimizer_converged": result.converged,
        "negative_dimer_curvature": curvature is not None and curvature < 0.0,
        "gradient_norm_hartree_per_angstrom": gradient,
        "max_gradient_norm_hartree_per_angstrom": float(
            gate["max_gradient_norm_hartree_per_angstrom"]
        ),
        "driver_gate_pass": gate_pass,
        "full_ts_validation": False,
    }
    report["claim_limits"].append(
        "Passing this driver gate remains insufficient for #19 until independent Hessian/mode and bidirectional endpoint evidence are attached."
    )
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "execution_stage": "dimer",
                "converged": result.converged,
                "curvature_eV_per_angstrom2": curvature,
                "gradient_norm_hartree_per_angstrom": gradient,
                "dimer_calculator_calls": dimer_calls,
                "total_calculator_calls": preflight_calls + dimer_calls,
            },
            sort_keys=True,
        )
    )
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
