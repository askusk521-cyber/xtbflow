#!/usr/bin/env python3
"""Run a bounded same-geometry CP2K/GFN2 reference bridge pilot."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
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
    CalculatorProtocol,
    MolecularSystem,
    XTBOracleAdapter,
    build_cp2k_protocol,
    inspect_cp2k_runtime,
    load_cp2k_protocol_document,
    physical_protocol_identity,
    physical_protocol_record,
    protocol_document_sha256,
)
from xtbflow.runtime import RunLedger, StageBudget  # noqa: E402
from xtbflow.validation import bridge_case_record, summarize_bridge  # noqa: E402


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
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


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _system(case: Mapping[str, Any]) -> MolecularSystem:
    return MolecularSystem(
        tuple(str(symbol) for symbol in case["symbols"]),
        tuple(tuple(float(value) for value in row) for row in case["coordinates_angstrom"]),
        int(case["charge"]),
        int(case["multiplicity"]),
        {"periodic": False, "bridge_case": str(case["system_id"])},
        str(case["system_id"]),
    )


def _gfn2_protocol(config: Mapping[str, Any]) -> CalculatorProtocol:
    parameters = config.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("semiempirical.parameters must be a mapping")
    implementation = str(config.get("implementation", ""))
    if implementation not in {"tblite", "xtb"}:
        raise ValueError("semiempirical.implementation must be tblite or xtb")
    return CalculatorProtocol(
        protocol_id=str(config["protocol_id"]),
        calculator="xtb_oracle",
        method=str(config["method"]),
        backend="cpu",
        parameters={**dict(parameters), "implementation": implementation},
    )


def _failure(calculator: str, error: Exception, *, consumed_calls: int) -> dict[str, Any]:
    return {
        "calculator": calculator,
        "status": "failure",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "calculator_calls": consumed_calls,
    }


def _public_result(result: Any) -> dict[str, Any]:
    payload = asdict(result)
    metadata = dict(payload.get("metadata", {}))
    for name in ("artifact_directory", "implementation_identity_paths"):
        if metadata.pop(name, None) is not None:
            metadata[f"{name}_recorded_in_private_run"] = True
    payload["metadata"] = metadata
    return payload


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/experiments/cp2k_bridge.yaml"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--cp2k-executable", default=os.environ.get("XTBFlow_CP2K_EXECUTABLE"))
    parser.add_argument("--cp2k-data-dir", default=os.environ.get("XTBFlow_CP2K_DATA_DIR"))
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = _load_json(config_path, label="reference bridge config")
    cases = config.get("cases")
    reference_config = config.get("reference")
    semiempirical_config = config.get("semiempirical")
    budget_config = config.get("budget")
    if not isinstance(cases, list) or not cases:
        parser.error("config.cases must be a nonempty list")
    if not isinstance(reference_config, Mapping):
        parser.error("config.reference must be a mapping")
    if not isinstance(semiempirical_config, Mapping):
        parser.error("config.semiempirical must be a mapping")
    if not isinstance(budget_config, Mapping):
        parser.error("config.budget must be a mapping")
    cp2k_limit = int(budget_config["max_cp2k_calls"])
    gfn2_limit = int(budget_config["max_gfn2_calls"])
    if len(cases) > cp2k_limit or len(cases) > gfn2_limit:
        parser.error("case count exceeds the frozen calculator-call budget")
    if int(budget_config.get("max_concurrent_jobs", 1)) != 1:
        parser.error("the C2 pilot requires max_concurrent_jobs=1")
    if not args.cp2k_executable:
        parser.error("set --cp2k-executable or XTBFlow_CP2K_EXECUTABLE")

    protocol_path = Path(str(reference_config["protocol_file"]))
    if not protocol_path.is_absolute():
        protocol_path = ROOT / protocol_path
    protocol_path = protocol_path.resolve()
    protocol_document = load_cp2k_protocol_document(protocol_path)
    runtime = inspect_cp2k_runtime(
        args.cp2k_executable, data_dir=args.cp2k_data_dir
    )
    threads = int(reference_config.get("threads", 1))
    timeout_seconds = float(reference_config.get("timeout_seconds", 900))
    if threads < 1 or timeout_seconds <= 0:
        parser.error("reference threads and timeout_seconds must be positive")
    os.environ["OMP_NUM_THREADS"] = str(threads)

    artifact_root = args.artifact_dir
    if artifact_root is None:
        artifact_root = ROOT / str(reference_config["artifact_root"])
    artifact_root = artifact_root.expanduser().resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    args.ledger_dir.mkdir(parents=True, exist_ok=True)
    cp2k_ledger_path = args.ledger_dir / "cp2k_ledger.json"
    gfn2_ledger_path = args.ledger_dir / "gfn2_ledger.json"
    for path in (cp2k_ledger_path, gfn2_ledger_path):
        if path.exists():
            parser.error(f"refusing to reuse an existing pilot ledger: {path}")

    phase = str(budget_config.get("phase", "P2a-reference-bridge-pilot"))
    cp2k_ledger = RunLedger(
        StageBudget(
            f"{phase}-cp2k",
            max_calculator_calls=cp2k_limit,
            max_concurrent_jobs=1,
            max_retries_per_job=int(budget_config.get("max_retries_per_job", 0)),
        )
    )
    gfn2_ledger = RunLedger(
        StageBudget(
            f"{phase}-gfn2",
            max_calculator_calls=gfn2_limit,
            max_concurrent_jobs=1,
            max_retries_per_job=int(budget_config.get("max_retries_per_job", 0)),
        )
    )
    cp2k_token = cp2k_ledger.issue_calculator_token(
        "c2-reference-bridge-cp2k",
        cp2k_limit,
        metadata={"case_count": len(cases)},
        persist_path=cp2k_ledger_path,
    )
    gfn2_token = gfn2_ledger.issue_calculator_token(
        "c2-reference-bridge-gfn2",
        gfn2_limit,
        metadata={"case_count": len(cases)},
        persist_path=gfn2_ledger_path,
    )
    gfn2_protocol = _gfn2_protocol(semiempirical_config)
    gfn2_adapter = XTBOracleAdapter(
        protocol=gfn2_protocol,
        implementation=str(semiempirical_config["implementation"]),
        require_budget_token=True,
    )

    records: list[dict[str, Any]] = []
    report_base = {
        "schema": str(config.get("schema", "xtbflow-reference-bridge/v1")),
        "status": "running",
        "scientific_qualification": False,
        "source_commit": _source_commit(),
        "script_sha256": protocol_document_sha256(Path(__file__)),
        "config_sha256": protocol_document_sha256(config_path),
        "reference_protocol_document_sha256": protocol_document_sha256(protocol_path),
        "reference_runtime": runtime.public_record(),
        "semiempirical_protocol_id": gfn2_protocol.protocol_id,
        "semiempirical_protocol_identity": gfn2_protocol.identity,
        "case_count": len(cases),
        "records": records,
        "claim_limits": list(config.get("claim_limits", [])),
    }

    fatal_error: dict[str, str] | None = None
    try:
        for index, case in enumerate(cases):
            if not isinstance(case, Mapping):
                raise ValueError(f"case {index} must be a mapping")
            system = _system(case)
            selection_reason = str(case.get("selection_reason", "unspecified"))
            record: dict[str, Any] = {
                "index": index,
                "system_id": system.system_id,
                "input_hash": system.input_hash,
                "selection_reason": selection_reason,
                "status": "running",
            }

            cp2k_result = None
            cp2k_before = cp2k_token.consumed_calls
            started = time.perf_counter()
            try:
                cp2k_protocol = build_cp2k_protocol(
                    protocol_document,
                    runtime,
                    charge=system.charge,
                    multiplicity=system.multiplicity,
                )
                cp2k_adapter = CP2KAdapter(
                    cp2k_protocol,
                    executable=runtime.executable,
                    timeout_seconds=timeout_seconds,
                    require_budget_token=True,
                    artifact_dir=artifact_root / str(system.system_id),
                    require_artifacts=True,
                )
                cp2k_result = cp2k_adapter.evaluate(
                    system, operation="energy_forces", budget_token=cp2k_token
                )
                record["reference"] = _public_result(cp2k_result)
                record["reference_physical_protocol"] = physical_protocol_record(
                    protocol_document,
                    runtime,
                    charge=system.charge,
                    multiplicity=system.multiplicity,
                )
                record["reference_physical_protocol_identity"] = (
                    physical_protocol_identity(
                        protocol_document,
                        runtime,
                        charge=system.charge,
                        multiplicity=system.multiplicity,
                    )
                )
                record["reference_runtime_bound_protocol_identity"] = (
                    cp2k_protocol.identity
                )
            except Exception as exc:
                record["reference"] = _failure(
                    "cp2k",
                    exc,
                    consumed_calls=cp2k_token.consumed_calls - cp2k_before,
                )
            record["reference_wall_seconds"] = time.perf_counter() - started

            gfn2_result = None
            gfn2_before = gfn2_token.consumed_calls
            started = time.perf_counter()
            try:
                gfn2_result = gfn2_adapter.evaluate(
                    system, operation="energy_forces", budget_token=gfn2_token
                )
                record["comparison"] = _public_result(gfn2_result)
            except Exception as exc:
                record["comparison"] = _failure(
                    "xtb_oracle",
                    exc,
                    consumed_calls=gfn2_token.consumed_calls - gfn2_before,
                )
            record["comparison_wall_seconds"] = time.perf_counter() - started

            if (
                cp2k_result is not None
                and cp2k_result.status == "success"
                and gfn2_result is not None
                and gfn2_result.status == "success"
            ):
                bridge = bridge_case_record(
                    system,
                    cp2k_result,
                    gfn2_result,
                    selection_reason=selection_reason,
                )
                record["bridge"] = bridge
                record["status"] = "success"
            else:
                record["status"] = "failure"
            records.append(record)
            _atomic_json(
                args.output,
                {
                    **report_base,
                    "records": records,
                    "progress": {
                        "completed_cases": len(records),
                        "cp2k_calls_consumed": cp2k_token.consumed_calls,
                        "gfn2_calls_consumed": gfn2_token.consumed_calls,
                    },
                },
            )
    except Exception as exc:
        fatal_error = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    finally:
        cp2k_token.settle(
            recorded_calls=cp2k_token.consumed_calls,
            persist_path=cp2k_ledger_path,
        )
        gfn2_token.settle(
            recorded_calls=gfn2_token.consumed_calls,
            persist_path=gfn2_ledger_path,
        )

    successful = sum(record.get("status") == "success" for record in records)
    failed = len(records) - successful
    status = (
        "completed"
        if fatal_error is None and len(records) == len(cases) and failed == 0
        else "partial_failure"
    )
    final_report = {
        **report_base,
        "status": status,
        "records": records,
        "summary": summarize_bridge(records),
        "completed_cases": len(records),
        "successful_cases": successful,
        "failed_cases": failed,
        "fatal_error": fatal_error,
        "calculator_accounting": {
            "cp2k": {
                "committed_calculator_calls": cp2k_ledger.committed_calculator_calls,
                "pending_reservations": len(cp2k_ledger.reservations),
            },
            "gfn2": {
                "committed_calculator_calls": gfn2_ledger.committed_calculator_calls,
                "pending_reservations": len(gfn2_ledger.reservations),
            },
        },
        "artifact_policy": "raw CP2K inputs, outputs, stdout, and stderr persisted outside the public report",
    }
    _atomic_json(args.output, final_report)
    print(
        json.dumps(
            {
                "status": status,
                "successful_cases": successful,
                "failed_cases": failed,
                "cp2k_calls": cp2k_ledger.committed_calculator_calls,
                "gfn2_calls": gfn2_ledger.committed_calculator_calls,
            },
            sort_keys=True,
        )
    )
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
