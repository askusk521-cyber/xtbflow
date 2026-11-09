#!/usr/bin/env python3
"""Read-only replay of a completed live-reference artifact and its settled ledger."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import subprocess
import sys

# Run from the exact execution checkout; the driver may be copied outside it.
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from xtbflow.governance.live_reference import load_live_feedback  # noqa: E402
from xtbflow.runtime import RunLedger  # noqa: E402
from xtbflow.validation import certification  # noqa: E402
from xtbflow.validation.certification import CertificationService, feedback, file_hash, identity  # noqa: E402
from xtbflow.validation.reference_backend import ReferenceBackend  # noqa: E402


def audit(root: Path, config_path: Path) -> dict:
    """No calculator or service.run call is possible in this read-only path."""
    config = json.loads(config_path.read_text())
    report_path = root / "reference/report.json"
    report = json.loads(report_path.read_text())
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    if dirty or source != report["source_commit"] or report["source_dirty"] is not False:
        raise ValueError("clean exact execution source required")
    if report["status"] != "pipeline_completed" or report["config_sha256"] != file_hash(config_path):
        raise ValueError("completed frozen reference report required")
    upstream, candidates = load_live_feedback(
        root / "live_input", root / "live_config.json", root / "reactant_context.json", config["live_source"]
    )
    if upstream != report["generation"] or len(report["candidates"]) != len(candidates):
        raise ValueError("upstream or candidate denominator mismatch")
    protocol = ReferenceBackend(device="gpu", execution_build=report["protocol"]["execution_build"]).identity
    if protocol != report["protocol"] or identity(protocol) != report["protocol_sha256"]:
        raise ValueError("reference build/protocol identity mismatch")
    folder = root / "reference/service"
    ledger = RunLedger.load(folder / "ledger.json")
    if ledger.reservations or ledger.summary() != report["ledger"]:
        raise ValueError("unsettled or mismatched reference ledger")
    checks, channel_ids = [], set()
    for summary, (item, candidate) in zip(report["candidates"], candidates):
        key = identity(dict(candidate=candidate.content(), protocol=protocol,
                            service_source_sha256=file_hash(Path(certification.__file__)),
                            ledger_source_sha256=file_hash(ROOT / "src/xtbflow/runtime/ledger.py")))
        if key != summary["cache_key"] or item["method"] != summary["method"]:
            raise ValueError("frozen candidate cache identity mismatch")
        work = folder / key / f"attempt-{summary['attempt']:04d}"
        record = CertificationService._load_record(work, key)
        replayed = feedback(record["evidence"])
        if record["execution_status"] != "completed":
            replayed["evidence_status"] = "attempted_unresolved"
        if replayed != record["feedback"] or record["content_sha256"] != summary["result_content_sha256"]:
            raise ValueError("replayed verdict or immutable result hash mismatch")
        if any(summary["feedback"][k] != record["feedback"][k] for k in summary["feedback"]):
            raise ValueError("public feedback differs from immutable result")
        expected_fields = dict(candidate_sha256=identity(candidate.content()),
                               execution_status=record["execution_status"], attempt=record["attempt"],
                               wall_seconds=record["resources"]["wall_seconds"],
                               cpu_hours=record["resources"]["cpu_seconds"] / 3600,
                               gpu_hours=record["ledger_event"]["gpu_seconds"] / 3600,
                               reason_code=record["feedback"]["reason"] or record["feedback"]["raw_status"])
        if any(summary[k] != value for k, value in expected_fields.items()):
            raise ValueError("public identity or cost differs from immutable result")
        if summary["cache_replay"] != dict(cache_hit=True, new_calculator_calls=0, new_channel=False, identical_record=True):
            raise ValueError("original cache replay did not pass")
        token = ledger.committed_tokens.get(record["token_id"])
        if record["charged_calls"] != summary["charged_calls"] or (token or {}).get("consumed_calls", 0) != record["charged_calls"]:
            raise ValueError("primitive charge mismatch")
        if [asdict(e) for e in ledger.events if e.event_id == record["token_id"]] != [record["ledger_event"]]:
            raise ValueError("resource event mismatch")
        settlement = json.loads((work / "settlement.json").read_text())
        if settlement != dict(status="settled", result_sha256=file_hash(work / "result.json")):
            raise ValueError("settlement mismatch")
        calls_path = work / "raw/calls.json"
        calls = json.loads(calls_path.read_text()) if calls_path.exists() else {"attempts": [], "consumed_calls": 0}
        attempts = calls["attempts"]
        if calls["consumed_calls"] != record["charged_calls"] or [c["sequence"] for c in attempts] != list(range(1, record["charged_calls"] + 1)):
            raise ValueError("durable primitive journal mismatch")
        raw = record["evidence"]
        if record["channel_id"]:
            channel_ids.add(record["channel_id"])
        checks.append(dict(
            arm=item["method"], cache_key=key, status=record["feedback"]["evidence_status"],
            execution_status=record["execution_status"], reason_code=summary["reason_code"],
            attempt=record["attempt"], wall_seconds=summary["wall_seconds"],
            cpu_hours=summary["cpu_hours"], gpu_hours=summary["gpu_hours"],
            charged_calls=record["charged_calls"], primitive_counts_by_stage=dict(Counter(c["stage"] for c in attempts)),
            unfinished_primitive_count=sum(c["status"] != "completed" for c in attempts),
            ts_status=raw.get("ts", {}).get("status"), irc_status=raw.get("irc", {}).get("status"),
            endpoint_statuses=[e.get("status") for e in raw.get("endpoints", [])],
            channel_id=record["channel_id"], observed_event_id=record["feedback"]["observed_event_id"],
            artifact_count=len(record["artifacts"]), artifact_bytes=sum(a["bytes"] for a in record["artifacts"]),
            request_sha256=record["request_sha256"], result_sha256=file_hash(work / "result.json"),
            result_content_sha256=record["content_sha256"], primitive_journal_sha256=file_hash(calls_path) if calls_path.exists() else None,
            replayed_feedback_identical=True, original_cache_replay=summary["cache_replay"],
        ))
    channels = json.loads((folder / "channels.json").read_text()) if (folder / "channels.json").exists() else {}
    if set(channels) != channel_ids or len(channel_ids) != report["unique_certified_channels"]:
        raise ValueError("channel index mismatch")
    if sum(c["charged_calls"] for c in checks) != ledger.summary()["totals"]["calculator_calls"]:
        raise ValueError("reference total cost mismatch")
    return dict(source_commit=source, source_dirty=dirty, config_sha256=file_hash(config_path),
                protocol_sha256=report["protocol_sha256"], report_sha256=file_hash(report_path),
                reference_ledger_sha256=file_hash(folder / "ledger.json"),
                parent_denominator=upstream["parent_denominator"], candidate_denominator=len(checks),
                unique_certified_channels=len(channel_ids), arms=checks, reference_totals=ledger.summary()["totals"],
                new_calculator_calls=0, replay_kind="read_only_raw_hash_verdict_and_ledger_reconstruction")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = dict(schema="xtbflow-live-reference-cache-audit/v1", host=socket.gethostname(),
                   audit_driver_sha256=file_hash(Path(__file__)), audited_at=datetime.now(timezone.utc).isoformat(),
                   status="failed", reason_code=None, attempt=1, wall_seconds=None, cpu_hours=None, gpu_hours=0.0,
                   new_calculator_calls=0,
                   evidence_level="software_demo", claim_limit="Cache/raw-file replay is not a new independent physical calculation.")
    try:
        receipt.update(audit(args.run_root, args.config), status="passed")
    except Exception as exc:
        receipt.update(reason_code="REFERENCE_CACHE_AUDIT_FAILED", error_type=type(exc).__name__)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    try:
        with args.out.open("x") as stream:
            json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    except OSError as exc:
        receipt.update(status="report_write_failed", reason_code="AUDIT_REPORT_WRITE_FAILED", error_type=type(exc).__name__)
        print(json.dumps(receipt, sort_keys=True, allow_nan=False))
        return 2
    print(json.dumps({k: receipt.get(k) for k in ("status", "reason_code", "new_calculator_calls", "unique_certified_channels")}))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
