#!/usr/bin/env python3
"""Bounded n2 GPU reference replay. Frozen source plan; no training side effects."""
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
from xtbflow.runtime import RunLedger, StageBudget  # noqa: E402
from xtbflow.validation.certification import (  # noqa: E402
    CertificationCandidate, CertificationService, file_hash, identity, write_json,
)
from xtbflow.validation.reference_backend import ReferenceBackend  # noqa: E402


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def mapped_candidate(item: dict, plan_hash: str) -> CertificationCandidate:
    """Whitelist generated state only; old verdicts never enter the backend."""
    if item["kind"] != "candidate" or item["coordinate_unit"] != "Angstrom":
        raise ValueError("unsupported archived candidate role or units")
    return CertificationCandidate(
        candidate_id=item["candidate_id"], parent_id=item["parent_id"],
        atomic_numbers=item["z"], coordinates_angstrom=item["x_start"],
        atom_ids=tuple(f"{item['query_id']}:source-index:{i}" for i in range(len(item["z"]))),
        reactant_be=item["b_r"], planned_product_be=item["b_predicted"],
        reactant_automorphisms=item["perms"], charge=item["charge"], multiplicity=item["multiplicity"],
        environment={"kind": "gas", "periodic": False},
        provenance={"input_sha256": plan_hash,
                    "atom_mapping_evidence": "PR119 O2 mapped fixed-pool query; z, BE and coordinates share original indices",
                    "electronic_state_evidence": "explicit O2 plan charge and multiplicity; not inferred from elements",
                    "source_item_id": item["item_id"], "role": "numerical_regression_only"})


def compare(raw: dict, archived: dict, tolerance: dict) -> dict:
    """Numeric comparison happens after certification; it cannot select inputs."""
    import numpy as np
    if not raw.get("ts", {}).get("x_ts") or len(raw.get("endpoints", [])) != 2:
        return {"status": "incomplete_chain", "passed": False}
    try:
        delta_e = abs(raw["ts"]["energy_hartree"] - archived["ts"]["energy_hartree"])
        delta_freq = float(np.max(np.abs(np.asarray(raw["ts"]["frequencies_cm1"]) - archived["ts"]["frequencies_cm1"])))
        x = np.asarray(raw["ts"]["x_ts"])
        y = np.asarray(archived["ts"]["x_ts"])
        x, y = x - x.mean(0), y - y.mean(0)
        u, _, vt = np.linalg.svd(x.T @ y)
        rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
        rmsd = float(np.sqrt(np.mean(np.sum((x @ rotation - y)**2, axis=1))))
        # IRC direction sign is arbitrary; compare each endpoint by reactant/product role.
        old_r, new_r = archived["identity"]["reactant_end"], raw["identity"]["reactant_end"]
        end_delta = max(abs(raw["endpoints"][i]["energy_hartree"] - archived["endpoints"][j]["energy_hartree"])
                        for i, j in ((new_r, old_r), (1 - new_r, 1 - old_r)))
        event_equal = raw["identity"]["actual_event_id"] == archived["identity"]["actual_event_id"]
        passed = (delta_e <= tolerance["ts_energy_hartree"] and delta_freq <= tolerance["frequencies_cm1"]
                  and rmsd <= tolerance["aligned_ts_rmsd_angstrom"] and end_delta <= tolerance["endpoint_energy_hartree"]
                  and event_equal)
        return dict(status="compared", passed=bool(passed), ts_energy_difference_hartree=delta_e,
                    max_frequency_difference_cm1=delta_freq, aligned_ts_rmsd_angstrom=rmsd,
                    max_endpoint_energy_difference_hartree=end_delta, actual_event_identical=event_equal)
    except (KeyError, ValueError, TypeError):
        return {"status": "unresolved_comparison", "passed": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/validation/certification_replay_20261107.json")
    args = parser.parse_args()
    if socket.gethostname() != "node2" or not os.environ.get("SLURM_JOB_ID"):
        parser.error("run through ssh n2 within a finite Slurm GPU allocation")
    if git("status", "--porcelain"):
        parser.error("source checkout must be clean")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema="xtbflow-certification-replay/v1", status="preflight", host=socket.gethostname(),
                  source_commit=git("rev-parse", "HEAD"), source_branch=git("branch", "--show-current"),
                  config_sha256=file_hash(config_path), input_plan_sha256=file_hash(args.plan),
                  started_at=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                  slurm_job_id=os.environ["SLURM_JOB_ID"], stage_budget=config["stage_budget"],
                  claim_limits=config["claim_limits"], retry_context=config.get("retry_context"), cases=[])
    write_json(args.out / "report.json", report)
    started = time.monotonic()
    try:
        if report["input_plan_sha256"] != config["input_plan_sha256"]:
            raise ValueError("source_plan_hash_mismatch")
        plan = json.loads(args.plan.read_text())
        items = {i["item_id"]: i for i in plan["items"] if i["kind"] == "candidate"}
        # Verify all inputs before allocating any primitive call budget.
        panel = []
        for case in config["cases"]:
            path = args.archive / ("candidate_" + case["item_id"]) / "verdict.json"
            if file_hash(path) != case["archived_verdict_sha256"]:
                raise ValueError("archived_verdict_hash_mismatch")
            panel.append((case, mapped_candidate(items[case["item_id"]], report["input_plan_sha256"]), path))
        # CUDA probe is a separate process, so the parent never owns a GPU context.
        probe_started = time.monotonic()
        probe = subprocess.check_output([sys.executable, "-c",
            "import cupy as c,json; r=c.cuda.runtime; p=r.getDeviceProperties(0); "
            "a=c.zeros(1); c.cuda.get_current_stream().synchronize(); "
            "print(json.dumps(dict(gpu=p['name'].decode(),driver=r.driverGetVersion(),runtime=r.runtimeGetVersion())))"],
            text=True, timeout=30)
        report["gpu"] = json.loads(probe)
        report["probe_wall_seconds"] = time.monotonic() - probe_started
        backend = ReferenceBackend(device="gpu", execution_build=report["gpu"])
        report["protocol"] = backend.identity
        report["protocol_sha256"] = identity(report["protocol"])
        if report["protocol"]["version"] != config["protocol_version"]:
            raise ValueError("protocol_version_mismatch")
        service = CertificationService(args.out / "service", StageBudget(**config["stage_budget"]))
        report["status"] = "executing"
        write_json(args.out / "report.json", report)
        for case, candidate, path in panel:
            result = service.run(candidate, backend, **config["per_candidate"])
            cached = service.run(candidate, backend, **config["per_candidate"])
            record = result["record"]
            row = dict(item_id=case["item_id"], archived_verdict_sha256=case["archived_verdict_sha256"],
                       candidate_sha256=identity(candidate.content()), cache_key=record["cache_key"],
                       execution_status=record["execution_status"], evidence_status=record["feedback"]["evidence_status"],
                       observed_event_id=record["feedback"]["observed_event_id"],
                       planned_event_id=record["feedback"]["planned_event_id"],
                       resources=record["resources"], charged_calls=record["charged_calls"],
                       artifact_count=len(record["artifacts"]), artifact_bytes=sum(x["bytes"] for x in record["artifacts"]),
                       result_content_sha256=record["content_sha256"],
                       numerical_comparison=compare(record["evidence"], json.loads(path.read_text()), config["numerical_tolerances"]),
                       cache_replay=dict(cache_hit=cached["cache_hit"], new_calculator_calls=cached["new_calculator_calls"],
                                         new_channel=cached["new_channel"], identical_record=cached["record"] == record))
            row["passed"] = (row["evidence_status"] == case["expected_evidence_status"] and row["numerical_comparison"]["passed"]
                             and cached["cache_hit"] and cached["new_calculator_calls"] == 0 and not cached["new_channel"])
            report["cases"].append(row)
            report["ledger"] = RunLedger.load(service.ledger_path).summary()
            write_json(args.out / "report.json", report)
        report["status"] = "replay_passed" if all(c["passed"] for c in report["cases"]) else "replay_failed"
        report["unique_channels"] = len(json.loads((service.root / "channels.json").read_text())) if (service.root / "channels.json").exists() else 0
    except Exception as exc:
        # Exception messages and private raw paths remain in n2 logs.
        report.update(status="execution_failed", failure_type=type(exc).__name__)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    report["wall_seconds"] = time.monotonic() - started
    write_json(args.out / "report.json", report)
    print(json.dumps({"status": report["status"], "cases": len(report["cases"])}))
    return 0 if report["status"] == "replay_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
