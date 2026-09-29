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
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists() or args.ledger.exists():
        parser.error("refusing to overwrite an existing output or ledger")
    if args.artifact_dir.exists() and any(args.artifact_dir.iterdir()):
        parser.error("refusing to mix with a nonempty artifact directory")
    document = json.loads(args.config.read_text(encoding="utf-8"))
    if document.get("schema") != "xtbflow-gfn2-dimer-pilot/v1":
        parser.error("unsupported pilot config schema")

    protocol = _protocol(document["protocol"])
    system = _system(document["system"])
    search = _search_config(document["search"])
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
    adapter = XTBOracleAdapter(
        protocol=protocol,
        implementation="tblite",
        require_budget_token=True,
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
    artifacts = index_artifacts(args.artifact_dir)
    source_commit = _source_commit()
    report = {
        "schema": "xtbflow-gfn2-dimer-pilot-result/v1",
        "status": "pending_settlement",
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
            "initial_mode": document["initial_mode"],
            "search": document["search"],
            "gate": document["gate"],
        },
        "driver_result": asdict(result),
        "artifact_index": artifacts,
        "calculator_accounting": {
            "reserved_calls": token.reserved_calls,
            "consumed_calls": token.consumed_calls,
            "settled": False,
        },
        "claim_limits": list(document["claim_limits"]),
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
    report["status"] = "pass" if gate_pass else "fail"
    report["ledger_sha256"] = sha256_file(args.ledger)
    report["calculator_accounting"] = {
        "committed_calculator_calls": accounting["committed_calculator_calls"],
        "pending_reservations": accounting["reservations"],
        "committed_tokens": accounting["committed_tokens"],
        "settled": True,
    }
    report["gate_evaluation"] = {
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
                "converged": result.converged,
                "curvature_eV_per_angstrom2": curvature,
                "gradient_norm_hartree_per_angstrom": gradient,
                "calculator_calls": accounting["committed_calculator_calls"],
            },
            sort_keys=True,
        )
    )
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
