#!/usr/bin/env python3
"""Bounded n2 CPU replay of a single decisioner over original reference evidence."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xtbflow.runtime.feedback_loop import (  # noqa: E402
    ACTIONS, SUPPORTED, CachedReferenceTools, FeedbackLoop, candidate_view,
)
from xtbflow.validation.certification import CertificationCandidate, file_hash, identity, write_json  # noqa: E402


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def inventory(root: Path) -> list[dict]:
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise ValueError("symlink in replay artifacts")
        if p.is_file() and not p.name.endswith(".lock"):
            rows.append(dict(path=str(p.relative_to(root)), bytes=p.stat().st_size, sha256=file_hash(p)))
    return rows


def expired(signum, frame):
    raise TimeoutError("feedback replay wall budget exhausted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != "node2" or not os.environ.get("SLURM_JOB_ID"):
        parser.error("requires ssh n2 and finite CPU Slurm allocation")
    if git("status", "--porcelain"):
        parser.error("source checkout must be clean")
    cfg_path = ROOT / "configs/agent/feedback_policy_20261107.json"
    replay_path = ROOT / "configs/agent/feedback_replay_20261107.json"
    cfg, replay = json.loads(cfg_path.read_text()), json.loads(replay_path.read_text())
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema="xtbflow-feedback-replay-report/v1", status="preflight", host=socket.gethostname(),
        source_commit=git("rev-parse", "HEAD"), source_branch=git("branch", "--show-current"),
        started_at=datetime.now(timezone.utc).isoformat(), slurm_job_id=os.environ["SLURM_JOB_ID"],
        python=platform.python_version(), packages={k: importlib.metadata.version(k) for k in ("numpy", "torch", "rdkit")},
        policy_sha256=file_hash(cfg_path), replay_config_sha256=file_hash(replay_path),
        limits=replay, new_calculator_calls=0, new_gpu_seconds=0, episodes=[],
        unavailable_actions=sorted(set(ACTIONS) - SUPPORTED),
        claim_limits=["Retrospective replay of deliberately selected numerical-regression cases, not blind discovery or an independent policy comparison.",
            "Historical reference costs are attributed once per unique evidence result in each policy episode; they are not new calculator spending.",
            "Sampler changes are configuration records only; this increment does not execute a subsequent generator, probe, refinement or network expansion.",
            "Repeated cache reads do not establish repeated physical convergence. The repeated-channel response has fixture tests only.",
            "Full pipeline, training/tuning and shared overhead costs remain unavailable. W09 two real adaptive loops and W10 matched-budget effectiveness remain unfinished."])
    write_json(args.out / "report.json", report)
    started, cpu_started = time.monotonic(), time.process_time()
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(replay["max_wall_seconds"])
    resource.setrlimit(resource.RLIMIT_FSIZE, (replay["max_output_bytes"], replay["max_output_bytes"]))
    try:
        panel = args.reference_panel.resolve()
        if (panel / "report.json").stat().st_size > replay["max_input_bytes"] or file_hash(panel / "report.json") != replay["original_panel_report_sha256"]:
            raise ValueError("original panel report identity changed")
        original_report = json.loads((panel / "report.json").read_text())
        if original_report["source_commit"] != replay["original_source_commit"]:
            raise ValueError("original execution source mismatch")
        inputs = {"report.json": dict(bytes=(panel / "report.json").stat().st_size,
                                       sha256=replay["original_panel_report_sha256"])}
        jobs = []
        for row in replay["attempts"]:
            relative = Path(row["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("invalid input locator")
            work = panel / relative
            if not work.resolve().is_relative_to(panel):
                raise ValueError("input escaped declared panel")
            if any((work / name).stat().st_size > replay["max_input_bytes"] for name in ("request.json", "result.json")):
                raise ValueError("input byte budget exceeded")
            if file_hash(work / "result.json") != row["result_sha256"] or file_hash(work / "request.json") != row["request_sha256"]:
                raise ValueError("frozen reference input changed")
            for source in [*(p for p in work.rglob("*") if p.is_file()), work.parent.parent / "ledger.json"]:
                if source.is_symlink():
                    raise ValueError("input symlink")
                name, size = str(source.relative_to(panel)), source.stat().st_size
                if name not in inputs and sum(r["bytes"] for r in inputs.values()) + size > replay["max_input_bytes"]:
                    raise ValueError("input byte budget exceeded")
                inputs[name] = dict(bytes=size, sha256=file_hash(source))
            request = json.loads((work / "request.json").read_text())
            c = CertificationCandidate(**request["candidate"])
            view = candidate_view(c, identity(request["protocol"]))
            jobs.append((row, work, c.candidate_id, view))
        report["inputs"] = inputs
        report["status"] = "executing"
        write_json(args.out / "report.json", report)
        for number, (row, work, candidate_id, view) in enumerate(jobs):
            for policy in replay["policies"]:
                active = deepcopy(cfg)
                active["policy"] = policy
                out = args.out / "raw" / f"episode-{number:02d}-{policy}"
                tools = CachedReferenceTools({candidate_id: work}, {candidate_id: row["result_sha256"]})
                loop = FeedbackLoop(out, [view], active, tools)
                state = loop.run()
                before = inventory(out)
                resumed = loop.run()
                if state != resumed or before != inventory(out):
                    raise ValueError("decision or durable artifact replay changed")
                actions = [json.loads(p.read_text()) for p in sorted(out.glob("step-*/action.json"))]
                report["episodes"].append(dict(episode=number, policy=policy, initial_candidate=view,
                    configuration_sha256=identity(active), state_sha256=identity(state),
                    replay_identical=True, artifacts=before, unique_channels=len(state["channels"]),
                    duplicate_channels=state["duplicate_channels"], reused_evidence=state["reused_evidence"],
                    historical_attributed_reference_cost=state["attributed_reference_cost"],
                    observed_statuses=[e["feedback"]["evidence_status"] for e in state["evidence"].values()],
                    execution_statuses=[e["execution_status"] for e in state["evidence"].values()],
                    observed_event_ids=[e["feedback"]["observed_event_id"] for e in state["evidence"].values()],
                    sampler=state["sampler"], sampler_changes=state["sampler_changes"], actions=actions,
                    stop_reason=state["stop_reason"]))
                if sum(r["bytes"] for r in inventory(args.out)) > replay["max_output_bytes"]:
                    raise ValueError("output byte budget exceeded")
                write_json(args.out / "report.json", report)
        if any(file_hash(panel / p) != r["sha256"] for p, r in inputs.items()):
            raise ValueError("original reference evidence changed")
        report.update(status="replay_passed", original_evidence_unchanged=True,
                      unique_original_results=len(jobs), original_reference_calls_once=sum(
                          e["historical_attributed_reference_cost"]["reference_primitive_calls"]
                          for e in report["episodes"] if e["policy"] == "fixed"))
    except Exception as exc:
        report.update(status="execution_failed", error_type=type(exc).__name__)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        signal.alarm(0)
    report.update(wall_seconds=time.monotonic() - started, cpu_seconds=time.process_time() - cpu_started)
    write_json(args.out / "report.json", report)
    print(json.dumps({k: report[k] for k in ("status", "new_calculator_calls", "wall_seconds")}))
    return 0 if report["status"] == "replay_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
