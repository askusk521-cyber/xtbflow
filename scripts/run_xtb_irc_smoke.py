#!/usr/bin/env python3
"""Run one bounded xTB Hessian/dual-path certification smoke.

The input file is deliberately external to the repository fixture.  It must
contain only the candidate geometry that a caller is authorized to validate;
the runner never reads a product, reference TS, or sealed label.  Raw input
and endpoint coordinates stay in the caller's private artifact directory; the
public manifest records hashes and structured status only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xtbflow.calculators import CalculatorProtocol, MolecularSystem, XTBOracleAdapter  # noqa: E402
from xtbflow.runtime import RunLedger, StageBudget  # noqa: E402
from xtbflow.validation.xtb_irc import XTBIRCConfig, XTBIRCResult, run_xtb_irc_certification  # noqa: E402


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _git(*args: str) -> str | None:
    result = subprocess.run(["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _load_input(path: Path) -> tuple[dict, MolecularSystem, str]:
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("input must be a JSON object")
    required = {"symbols", "coordinates_angstrom", "charge", "multiplicity", "environment", "system_id", "candidate_id"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"input is missing explicit fields: {missing}")
    system = MolecularSystem(
        tuple(str(item) for item in payload["symbols"]),
        tuple(tuple(float(value) for value in row) for row in payload["coordinates_angstrom"]),
        int(payload["charge"]),
        int(payload["multiplicity"]),
        dict(payload["environment"]),
        str(payload["system_id"]),
    )
    return payload, system, raw


def _config(payload: dict) -> XTBIRCConfig:
    values = dict(payload.get("certification", {}))
    return XTBIRCConfig(**values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--implementation", choices=("xtb", "tblite"), default="xtb")
    parser.add_argument("--max-calculator-calls", type=int, default=256)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("refusing to overwrite a nonempty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    input_payload, system, input_raw = _load_input(args.input)
    config_values = dict(input_payload.get("certification", {}))
    config_values["max_calls"] = args.max_calculator_calls
    config = XTBIRCConfig(**config_values)
    protocol = CalculatorProtocol(
        protocol_id=str(input_payload.get("protocol_id", "xtbflow-xtb-irc-smoke-v1")),
        calculator="xtb_oracle",
        method="GFN2-xTB",
        backend="cpu",
        parameters={
            "accuracy": float(input_payload.get("accuracy", 1.0)),
            "max_iterations": int(input_payload.get("max_iterations", 100)),
            "electronic_temperature": int(input_payload.get("electronic_temperature", 300)),
            "solvent": input_payload.get("solvent"),
        },
    )
    manifest: dict[str, object] = {
        "schema": "xtbflow-xtb-irc-smoke/v1",
        "status": "preflight",
        "source": {"commit": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain"))},
        "execution_host": platform.node(),
        "python": platform.python_version(),
        "input": {"candidate_id": input_payload["candidate_id"], "system_id": system.system_id, "input_hash": system.input_hash},
        "input_sha256": _sha256_text(input_raw),
        "protocol": {"protocol_id": protocol.protocol_id, "identity": protocol.identity, "parameters": dict(protocol.parameters)},
        "config": config.to_mapping(),
        "config_sha256": _sha256_text(_canonical(config.to_mapping())),
        "claim_levels": {"workflow": "software_demo", "certification": "strict_validation"},
        "claim_limit": "A successful record validates this frozen software protocol only; it is not a discovery, barrier, mechanism, or chemical-accuracy claim.",
    }
    ledger_path = args.output_dir / "ledger.json"
    result: XTBIRCResult | None = None
    ledger: RunLedger | None = None
    try:
        if importlib.util.find_spec(args.implementation) is None:
            manifest.update({"status": "blocked_backend", "reason_code": "BLOCKED_BACKEND", "backend": args.implementation})
        else:
            ledger = RunLedger(StageBudget("xtb-irc-smoke", max_calculator_calls=args.max_calculator_calls, max_cpu_core_hours=2.0, max_wall_minutes_per_job=30.0))
            token = ledger.issue_calculator_token("xtb-irc-smoke", args.max_calculator_calls, metadata={"candidate_id": input_payload["candidate_id"], "protocol_id": protocol.protocol_id}, persist_path=ledger_path)
            backend = XTBOracleAdapter(protocol=protocol, implementation=args.implementation, require_budget_token=True)
            try:
                result = run_xtb_irc_certification(backend, system, token=token, candidate_id=str(input_payload["candidate_id"]), config=config)
            finally:
                token.settle(recorded_calls=token.consumed_calls, persist_path=ledger_path)
            manifest.update({"status": result.status, "reason_code": result.reason_code, "result": result.to_mapping(), "ledger": ledger.summary(), "calculator": asdict(backend.capabilities)})
    except Exception as exc:
        manifest.update({"status": "execution_failed", "reason_code": "RUNNER_EXECUTION", "error": f"{type(exc).__name__}: {exc}"})
        if ledger is not None:
            manifest["ledger"] = ledger.summary()
    manifest["cost"] = {
        "wall_seconds": time.perf_counter() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
        "cpu_hours": (time.process_time() - started_cpu) / 3600.0,
        "gpu_hours": 0.0,
    }
    _write(args.output_dir / "manifest.json", manifest)
    (args.output_dir / "summary.md").write_text(
        "# xTB TS/IRC smoke\n\n"
        f"status: `{manifest['status']}`\n\n"
        f"reason_code: `{manifest.get('reason_code', 'none')}`\n\n"
        f"source_commit: `{manifest['source']['commit']}` source_dirty: `{manifest['source']['dirty']}`\n\n"
        "This artifact is a bounded software/backend smoke and carries no discovery or chemical-accuracy claim.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": manifest["status"], "reason_code": manifest.get("reason_code"), "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0 if manifest["status"] in {"success", "attempted_unresolved", "blocked_backend", "budget_exhausted"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
