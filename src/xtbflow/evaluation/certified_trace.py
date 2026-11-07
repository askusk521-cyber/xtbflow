"""Reconstruct a reference-phase discovery attempt from durable service evidence."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from xtbflow.validation.certification import CertificationService, feedback, file_hash, identity
from xtbflow.runtime import RunLedger
from .discovery import Certificate, COST_AXES, DiscoveryAttempt


def read_reference_attempt(work: Path) -> tuple[DiscoveryAttempt, dict]:
    """Verify all artifacts before exposing a scientific observation or its cost.

    This is one stored reference attempt, not a whole pipeline cost. Generator,
    training/tuning, shared setup, scheduler overhead and cache-lookup work must
    be supplied separately; absent stages must never be imputed as free.
    """
    work = Path(work)
    record = CertificationService._load_record(work, work.parent.name)
    settlement = json.loads((work / "settlement.json").read_text())
    if settlement["status"] != "settled" or settlement["result_sha256"] != file_hash(work / "result.json"):
        raise ValueError("reference result is not durably settled")
    request = json.loads((work / "request.json").read_text())
    ledger = RunLedger.load(work.parent.parent / "ledger.json")
    events = [e for e in ledger.events if e.event_id == record["token_id"]]
    if len(events) != 1 or asdict(events[0]) != record["ledger_event"]:
        raise ValueError("reference event does not reconcile with the shared ledger")
    if record["charged_calls"] and ledger.committed_tokens.get(record["token_id"], {}).get("consumed_calls") != record["charged_calls"]:
        raise ValueError("reference calls have no matching committed token")
    if request["protocol"] != record["protocol"]:
        raise ValueError("request and result protocols differ")
    observed = feedback(record["evidence"])
    if record["execution_status"] != "completed":
        observed["evidence_status"] = "attempted_unresolved"
    if observed != record["feedback"]:
        raise ValueError("recorded feedback does not reproduce from retained evidence")
    path = work / "raw/calls.json"
    journal = json.loads(path.read_text()) if path.exists() else {"consumed_calls": 0, "attempts": []}
    calls = journal["attempts"]
    if len(calls) != journal["consumed_calls"] or len(calls) != record["charged_calls"]:
        raise ValueError("durable primitive accounting mismatch")
    costs = {k: 0 for k in COST_AXES}
    mapping = {"scf": "energy_calls", "gradient": "force_calls", "hessian": "hessian_calls"}
    for i, call in enumerate(calls, 1):
        if call["sequence"] != i or call["kind"] not in mapping or call["status"] not in {"started", "completed"}:
            raise ValueError("invalid reference primitive journal")
        costs[mapping[call["kind"]]] += 1
    costs.update(reference_primitive_calls=len(calls),
                 reference_attempts=int(not record["execution_status"].startswith("not_started")),
                 retries=int(record["attempt"] > 1), wall_seconds=record["resources"]["wall_seconds"],
                 cpu_seconds=record["resources"]["cpu_seconds"],
                 gpu_seconds=record["resources"]["wall_seconds"] if record["protocol"]["device"] == "gpu" else 0,
                 storage_bytes=sum(a["bytes"] for a in record["artifacts"]))
    certificate = Certificate(result_sha256=file_hash(work / "result.json"),
        protocol_sha256=identity(record["protocol"]), evidence_status=record["feedback"]["evidence_status"],
        observed_event_id=record["feedback"]["observed_event_id"])
    status = record["execution_status"]
    attempt = DiscoveryAttempt(candidate_id=record["candidate_id"], status=status, cost=costs, certificate=certificate)
    return attempt, dict(parent_id=request["candidate"]["parent_id"],
        input_sha256=request["candidate"]["provenance"]["input_sha256"],
        original_execution_status=record["execution_status"], resource_basis=record["resources"]["basis"],
        result_sha256=certificate.result_sha256, request_sha256=record["request_sha256"],
        unfinished_primitives=sum(c["status"] == "started" for c in calls),
        cost_scope="reference_attempt_only",
        excluded_cost_stages=["generation", "training", "tuning", "scheduler_and_shared_setup", "cache_lookup"])
