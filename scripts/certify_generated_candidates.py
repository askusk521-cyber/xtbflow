#!/usr/bin/env python3
"""Consume a complete hashed generation stage through the shared reference service."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xtbflow.governance.generation_bridge import validate_journal  # noqa: E402
from xtbflow.governance.research_replay import git  # noqa: E402
from xtbflow.runtime import RunLedger, StageBudget  # noqa: E402
from xtbflow.validation.certification import (  # noqa: E402
    CertificationCandidate, CertificationService, file_hash, identity, write_json,
)
from xtbflow.validation.reference_backend import ReferenceBackend  # noqa: E402


def load_generation(folder: Path, generation_config: Path) -> tuple[dict, list[tuple[dict, CertificationCandidate]]]:
    report = json.loads((folder / "report.json").read_text())
    cfg = json.loads(generation_config.read_text())
    if report["status"] != "generation_completed" or report["settlement_status"] != "settled":
        raise ValueError("generation must complete and settle before certification")
    if report["config_sha256"] != file_hash(generation_config):
        raise ValueError("generation configuration identity mismatch")
    for artifact in report["artifacts"]:
        path = (folder / artifact["path"]).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file():
            raise ValueError("invalid generation artifact")
        if path.stat().st_size != artifact["bytes"] or file_hash(path) != artifact["sha256"]:
            raise ValueError("generation artifact identity mismatch")
    plan_path, journal_path = folder / "raw/certification-plan.json", folder / "raw/calls.json"
    if file_hash(plan_path) != report["certification_plan_sha256"]:
        raise ValueError("generation plan identity mismatch")
    plan = json.loads(plan_path.read_text())
    if plan["schema"] != "xtbflow-generated-certification-plan/v1" or plan["source_commit"] != report["source_commit"]:
        raise ValueError("generation source identity mismatch")
    if plan["generator_journal_sha256"] != file_hash(journal_path):
        raise ValueError("generation journal identity mismatch")
    calls = validate_journal(json.loads(journal_path.read_text()), cap=cfg["generation"]["max_forward_evaluations"])
    if calls != report["call_summary"] or calls["unfinished_calls"]:
        raise ValueError("generation call accounting mismatch")
    expected = [row for row in report["candidates"] if row["status"] == "generated"]
    if len(expected) != len(plan["candidates"]) or len(expected) != report["certification_candidates"]:
        raise ValueError("candidate selection mismatch")
    candidates = []
    for summary, item in zip(expected, plan["candidates"]):
        candidate = CertificationCandidate(**item["candidate"])
        if (identity(candidate.content()) != summary["candidate_content_sha256"]
                or candidate.candidate_id != summary["candidate_id"]
                or item["method"] != summary["method"]
                or item["generation_forward_evaluations"] != summary["forward_evaluations"]
                or candidate.provenance["generation_protocol_sha256"] != identity(plan["generation_protocol"])):
            raise ValueError("generated candidate identity mismatch")
        candidates.append((item, candidate))
    return report, candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--generation-config", type=Path, default=ROOT / "configs/research/generation_bridge_20261107.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/validation/generated_candidates_20261107.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != "node2" or not os.environ.get("SLURM_JOB_ID"):
        parser.error("use ssh n2 and a finite Slurm GPU allocation")
    if git(ROOT, "status", "--porcelain"):
        parser.error("clean committed source required")
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text())
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema=cfg["schema"], status="preflight", host=socket.gethostname(),
                  source_commit=git(ROOT, "rev-parse", "HEAD"), python=platform.python_version(),
                  started_at=datetime.now(timezone.utc).isoformat(), slurm_job_id=os.environ["SLURM_JOB_ID"],
                  config_sha256=file_hash(config_path), budget=cfg["stage_budget"],
                  claim_limits=cfg["claim_limits"], candidates=[])
    write_json(args.out / "report.json", report)
    started = time.monotonic()
    service = None
    try:
        generation, candidates = load_generation(args.generation, args.generation_config)
        if len(candidates) > cfg["max_candidates"]:
            raise ValueError("candidate cap exceeded")
        report["generation"] = dict(report_sha256=file_hash(args.generation / "report.json"),
            source_commit=generation["source_commit"], plan_sha256=generation["certification_plan_sha256"],
            forward_evaluations=generation["call_summary"]["forward_evaluations"],
            wall_seconds=generation["wall_seconds"], cpu_seconds=generation["cpu_seconds"],
            candidates=generation["candidates"])
        probe = subprocess.check_output([sys.executable, "-c",
            "import cupy as c,json; r=c.cuda.runtime; p=r.getDeviceProperties(0); "
            "a=c.zeros(1); c.cuda.get_current_stream().synchronize(); "
            "print(json.dumps(dict(gpu=p['name'].decode(),driver=r.driverGetVersion(),runtime=r.runtimeGetVersion())))"],
            text=True, timeout=30)
        backend = ReferenceBackend(device="gpu", execution_build=json.loads(probe))
        report["protocol"] = backend.identity
        report["protocol_sha256"] = identity(report["protocol"])
        if report["protocol"]["version"] != cfg["protocol_version"]:
            raise ValueError("reference protocol version mismatch")
        service = CertificationService(args.out / "service", StageBudget(**cfg["stage_budget"]))
        report["status"] = "executing"
        write_json(args.out / "report.json", report)
        for item, candidate in candidates:
            result = service.run(candidate, backend, **cfg["per_candidate"])
            record = result["record"]
            cached = service.run(candidate, backend, **cfg["per_candidate"])
            report["candidates"].append(dict(candidate_id=candidate.candidate_id, method=item["method"],
                candidate_sha256=identity(candidate.content()), cache_key=record["cache_key"],
                execution_status=record["execution_status"], feedback={k: record["feedback"][k] for k in (
                    "evidence_status", "stationary_point", "first_order_saddle", "bidirectional_minima",
                    "planned_event_realized", "planned_event_id", "observed_event_id", "raw_status")},
                charged_calls=record["charged_calls"], resources=record["resources"],
                result_content_sha256=record["content_sha256"], artifact_bytes=sum(a["bytes"] for a in record["artifacts"]),
                cache_replay=dict(cache_hit=cached["cache_hit"], new_calculator_calls=cached["new_calculator_calls"],
                    new_channel=cached["new_channel"], identical_record=cached["record"] == record)))
            if not cached["cache_hit"] or cached["new_calculator_calls"] or cached["new_channel"] or cached["record"] != record:
                raise ValueError("certification cache replay mismatch")
            report["ledger"] = RunLedger.load(service.ledger_path).summary()
            write_json(args.out / "report.json", report)
        report["status"] = "pipeline_completed" if candidates else "no_generated_candidates"
    except Exception as exc:
        report.update(status="execution_failed", failure_type=type(exc).__name__)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    if service and service.ledger_path.exists():
        report["ledger"] = RunLedger.load(service.ledger_path).summary()
    channels = args.out / "service/channels.json"
    report["unique_certified_channels"] = len(json.loads(channels.read_text())) if channels.exists() else 0
    report["wall_seconds"] = time.monotonic() - started
    write_json(args.out / "report.json", report)
    print(json.dumps({"status": report["status"], "unique_certified_channels": report["unique_certified_channels"]}))
    return 0 if report["status"] in {"pipeline_completed", "no_generated_candidates"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
