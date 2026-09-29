#!/usr/bin/env python3
"""Run a bounded, auditable GFN2 preflight over normalized candidate rows.

The input JSONL is deliberately explicit: every row must contain symbols,
coordinates, charge, multiplicity and an initial mode. This runner performs
the three-call preflight only. It never turns a rejected preflight into a
complete TS-search failure and never runs a dimer search for a rejected row.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping


_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_sha256(record: Mapping[str, Any]) -> str:
    encoded = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_candidate_manifest(path: Path) -> list[dict[str, Any]]:
    """Load normalized candidates without supplying missing scientific fields."""

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid candidate JSON at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"candidate at line {line_number} must be an object")
            candidate_id = row.get("candidate_id")
            required = {"candidate_id", "symbols", "coordinates_angstrom", "charge", "multiplicity", "initial_mode"}
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(f"candidate {candidate_id or line_number} is missing {missing}")
            if not isinstance(candidate_id, str) or not candidate_id.strip():
                raise ValueError(f"candidate at line {line_number} has no candidate_id")
            if candidate_id in seen:
                raise ValueError(f"duplicate candidate_id: {candidate_id}")
            if type(row["charge"]) is not int or type(row["multiplicity"]) is not int or row["multiplicity"] < 1:
                raise ValueError(f"candidate {candidate_id} has invalid explicit electronic state")
            seen.add(candidate_id)
            rows.append(row)
    if not rows:
        raise ValueError("candidate manifest is empty")
    return rows


def _source_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _system(row: Mapping[str, Any]):
    from xtbflow.calculators import MolecularSystem

    return MolecularSystem(
        tuple(str(value) for value in row["symbols"]),
        tuple(tuple(float(value) for value in xyz) for xyz in row["coordinates_angstrom"]),
        int(row["charge"]),
        int(row["multiplicity"]),
        dict(row.get("environment", {"periodic": False})),
        str(row["candidate_id"]),
    )


def _protocol(payload: Mapping[str, Any]):
    from xtbflow.calculators import CalculatorProtocol

    return CalculatorProtocol(
        protocol_id=str(payload["protocol_id"]),
        calculator=str(payload["calculator"]),
        method=str(payload["method"]),
        backend=str(payload["backend"]),
        parameters=dict(payload["parameters"]),
    )


def _search_config(payload: Mapping[str, Any]):
    from xtbflow.validation.ase_dimer import ASEDimerConfig

    return ASEDimerConfig(
        fmax_eV_per_angstrom=float(payload["fmax_eV_per_angstrom"]),
        max_steps=int(payload["max_steps"]),
        dimer_separation_angstrom=float(payload["dimer_separation_angstrom"]),
        max_num_rot=int(payload["max_num_rot"]),
        maximum_translation_angstrom=float(payload["maximum_translation_angstrom"]),
        trial_translation_step_angstrom=float(payload["trial_translation_step_angstrom"]),
        random_seed=int(payload["random_seed"]),
        remove_rigid_body_modes=bool(payload["remove_rigid_body_modes"]),
        rigid_body_tolerance=float(payload["rigid_body_tolerance"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--source-locator", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing output")
    if args.ledger_dir.exists() and any(args.ledger_dir.iterdir()):
        parser.error("refusing to mix with a nonempty ledger directory")
    if args.artifact_dir.exists() and any(args.artifact_dir.iterdir()):
        parser.error("refusing to mix with a nonempty artifact directory")

    try:
        rows = load_candidate_manifest(args.manifest)
        config = json.loads(args.config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    if config.get("schema") != "xtbflow-kingfisher-gfn2-preflight/v1":
        parser.error("unsupported Kingfisher preflight config schema")

    root = Path(__file__).resolve().parents[1]
    from xtbflow.calculators import XTBOracleAdapter
    from xtbflow.runtime import RunLedger, StageBudget
    from xtbflow.validation.ase_dimer import run_dimer_preflight
    from xtbflow.validation.evidence import sanitize_public_value

    protocol = _protocol(config["protocol"])
    search_config = _search_config(config["search"])
    adapter = XTBOracleAdapter(protocol=protocol, implementation="tblite", require_budget_token=True)
    args.ledger_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    budget = config["preflight"]["budget"]
    gate = config["preflight"]["gate"]

    for index, row in enumerate(rows):
        base = {
            "index": index,
            "candidate_id": row["candidate_id"],
            "source_record_id": row.get("source_record_id"),
            "source_record_sha256": _record_sha256(row),
            "input": {
                "charge": row["charge"],
                "multiplicity": row["multiplicity"],
                "symbols": row["symbols"],
            },
            "preflight_status": "not_run",
            "search_status": "not_executed",
            "search_initial_guess_eligible": False,
            "status": "pending",
        }
        try:
            system = _system(row)
            base["input"].update({"input_hash": system.input_hash, "system_id": system.system_id})
            safe_system_id = _SAFE_ID.sub("_", system.system_id)
            ledger_path = args.ledger_dir / f"{safe_system_id}.json"
            artifact_path = args.artifact_dir / f"{safe_system_id}.json"
            ledger = RunLedger(
                StageBudget(
                    str(budget["phase"]),
                    max_calculator_calls=int(budget["max_calculator_calls"]),
                    max_concurrent_jobs=int(budget["max_concurrent_jobs"]),
                    max_retries_per_job=int(budget["max_retries_per_job"]),
                )
            )
            token = ledger.issue_calculator_token(
                f"kingfisher-preflight-{system.system_id}",
                int(budget["max_calculator_calls"]),
                metadata={
                    "candidate_id": system.system_id,
                    "input_hash": system.input_hash,
                    "protocol_id": protocol.protocol_id,
                },
                persist_path=ledger_path,
            )
            try:
                result = run_dimer_preflight(
                    adapter,
                    system,
                    row["initial_mode"],
                    budget_token=token,
                    step_angstrom=float(config["preflight"]["hvp_step_angstrom"]),
                    config=search_config,
                    require_durable_token=True,
                )
                result_payload = asdict(result)
            except Exception as exc:
                result_payload = {
                    "status": "failure",
                    "calculator_calls": token.consumed_calls,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            result_payload = sanitize_public_value(result_payload)
            token.settle(recorded_calls=token.consumed_calls, persist_path=ledger_path)
            ledger_payload = ledger.to_dict()
            curvature = result_payload.get("mode_curvature_eV_per_angstrom2")
            force_norm = result_payload.get("projected_force_norm_hartree_per_angstrom")
            passes = (
                result_payload.get("status") == "success"
                and (curvature is not None and float(curvature) < 0.0)
                and (force_norm is not None and float(force_norm) <= float(gate["max_projected_force_norm_hartree_per_angstrom"]))
            )
            base.update({
                "status": "completed",
                "runtime": {
                    "implementation": "tblite",
                    "version": adapter.version,
                    "build_hash": adapter.build_hash,
                },
                "protocol_id": protocol.protocol_id,
                "preflight_status": "passed" if passes else "rejected",
                "preflight_result": result_payload,
                "preflight_gate": dict(gate),
                "search_status": "not_executed",
                "search_initial_guess_eligible": passes,
                "search_execution_reason": "preflight_only_batch_runner",
                "ledger": {
                    "relative_path": ledger_path.relative_to(args.ledger_dir.parent).as_posix(),
                    "sha256": _sha256(ledger_path),
                    "committed_calculator_calls": ledger_payload["committed_calculator_calls"],
                    "pending_reservations": ledger_payload["reservations"],
                    "settled": True,
                },
            })
            artifact_path.write_text(json.dumps(base, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        except Exception as exc:
            base.update({
                "status": "input_rejected",
                "preflight_status": "not_run",
                "search_status": "not_executed",
                "search_execution_reason": "invalid_candidate_input",
                "error": f"{type(exc).__name__}: {exc}",
            })
            base = sanitize_public_value(base)
        records.append(base)

    summary = {
        "total": len(records),
        "preflight_pass": sum(row.get("preflight_status") == "passed" for row in records),
        "preflight_rejected": sum(row.get("preflight_status") == "rejected" for row in records),
        "input_rejected": sum(row.get("status") == "input_rejected" for row in records),
        "search_executed": 0,
    }
    report = sanitize_public_value({
        "schema": "xtbflow-kingfisher-gfn2-preflight/v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "scientific_qualification": False,
        "source_commit": _source_commit(root),
        "source_asset": {"locator": args.source_locator, "sha256": _sha256(args.manifest), "record_count": len(rows)},
        "protocol": {"protocol_id": protocol.protocol_id, "identity": protocol.identity, "method": protocol.method, "backend": protocol.backend, "parameters": dict(protocol.parameters)},
        "runtime": {"version": adapter.version, "build_hash": adapter.build_hash},
        "search_config": asdict(search_config),
        "summary": summary,
        "records": records,
        "claim_limits": [
            "This batch performs only a three-call GFN2 preflight per valid candidate.",
            "A preflight rejection is not a complete TS-search failure.",
            "Rejected candidates are not approved as dimer initial guesses.",
            "A preflight pass records eligibility only; no dimer search is executed by this command.",
        ],
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"schema": report["schema"], "summary": summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
