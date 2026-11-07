#!/usr/bin/env python3
"""Consume a completed real feedback batch and execute one bounded next proposal."""
from __future__ import annotations

import argparse
import fcntl
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from certify_generated_candidates import load_generation  # noqa: E402
from xtbflow.governance.research_replay import git  # noqa: E402
from xtbflow.runtime import RunLedger, StageBudget  # noqa: E402
from xtbflow.runtime.feedback_loop import FeedbackLoop, candidate_view, zero_cost  # noqa: E402
from xtbflow.runtime.generation_actions import GenerationActions  # noqa: E402
from xtbflow.runtime.live_feedback_tools import LiveGenerationTools  # noqa: E402
from xtbflow.validation.certification import CertificationCandidate, file_hash, identity, write_json  # noqa: E402


def prepare_inputs(generation: Path, reference: Path, config: dict) -> tuple:
    if file_hash(generation / "report.json") != config["generation_report_sha256"]:
        raise ValueError("original generation report changed")
    previous, candidates = load_generation(generation, ROOT / "configs/research/generation_bridge_20261107.json")
    if previous["source_commit"] != config["input_source_commit"] or previous["certification_plan_sha256"] != config["generation_plan_sha256"]:
        raise ValueError("original source or generation plan changed")
    packet_path = generation / "raw/reactants.json"
    if file_hash(packet_path) != config["reactants_sha256"]:
        raise ValueError("original reactant packet changed")
    queries = json.loads(packet_path.read_text())["queries"]
    if len(queries) != 1:
        raise ValueError("one fixed reactant pool required")
    report = json.loads((reference / "report.json").read_text())
    if report["status"] != "pipeline_completed" or report["source_commit"] != config["input_source_commit"] or report["protocol_sha256"] != config["reference_protocol_sha256"]:
        raise ValueError("reference batch must be complete under the frozen source/protocol")
    if report["generation"]["report_sha256"] != config["generation_report_sha256"] or report["generation"]["plan_sha256"] != config["generation_plan_sha256"]:
        raise ValueError("reference input differs from the frozen generated batch")
    expected = {c.candidate_id: c for _, c in candidates}
    if len(report["candidates"]) != len(expected) or {r["candidate_id"] for r in report["candidates"]} != set(expected):
        raise ValueError("reference batch omitted a candidate")
    views, attempts, pins = [], {}, {}
    for row in report["candidates"]:
        work = reference / "service" / row["cache_key"] / "attempt-0001"
        if not work.resolve().is_relative_to((reference / "service").resolve()):
            raise ValueError("reference locator escaped service root")
        request = json.loads((work / "request.json").read_text())
        c = CertificationCandidate(**request["candidate"])
        if identity(c.content()) != identity(expected[row["candidate_id"]].content()) or identity(request["protocol"]) != config["reference_protocol_sha256"]:
            raise ValueError("reference request and generation candidate differ")
        record = json.loads((work / "result.json").read_text())
        if record["content_sha256"] != row["result_content_sha256"]:
            raise ValueError("reference result differs from the completed batch")
        views.append(candidate_view(c, config["reference_protocol_sha256"]))
        attempts[c.candidate_id], pins[c.candidate_id] = work, file_hash(work / "result.json")
    costs = zero_cost()
    costs.update(raw_proposals=len(previous["candidates"]), wall_seconds=previous["wall_seconds"],
        cpu_seconds=previous["cpu_seconds"], gpu_seconds=previous["wall_seconds"],
        storage_bytes=sum(r["bytes"] for r in previous["artifacts"]))
    for role in ("joint", "event", "geometry"):
        costs[role + "_forwards"] = previous["call_summary"]["by_role"].get(role, 0)
    return queries[0], views, attempts, pins, costs, file_hash(reference / "report.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--research-source", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Reverify/reconcile the same run; never silently retry an interrupted generation")
    args = parser.parse_args()
    if socket.gethostname() != "node2" or not os.environ.get("SLURM_JOB_ID"):
        parser.error("requires ssh n2 and a finite Slurm GPU allocation")
    if git(ROOT, "status", "--porcelain"):
        parser.error("source checkout must be clean")
    run_path, policy_path = (ROOT / f"configs/agent/live_generation_{name}_20261107.json" for name in ("run", "policy"))
    cfg, policy = json.loads(run_path.read_text()), json.loads(policy_path.read_text())
    args.out = args.out.resolve()
    if not args.resume:
        args.out.mkdir(parents=True, exist_ok=False)
    run_lock = (args.out / "run.lock").open("a")
    fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    binding = dict(source_commit=git(ROOT, "rev-parse", "HEAD"), run_config_sha256=file_hash(run_path),
        policy_template_sha256=file_hash(policy_path), generation=str(args.generation.resolve()),
        reference=str(args.reference.resolve()), research_source=str(args.research_source.resolve()), model_root=str(args.model_root.resolve()))
    manifest_path = args.out / "run-manifest.json"
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        if manifest["binding"] != binding:
            parser.error("resume source, config or input locations changed")
        if (args.out / "report.json").exists():
            history = args.out / "reports"
            history.mkdir(exist_ok=True)
            snapshot = history / f"report-{len(list(history.glob('report-*.json'))):04d}.json"
            snapshot.write_bytes((args.out / "report.json").read_bytes())
    else:
        manifest = dict(schema="xtbflow-live-feedback-run/v1", binding=binding, inputs=None)
        write_json(manifest_path, manifest)
    started, cpu = time.monotonic(), time.process_time()
    report = dict(schema="xtbflow-live-feedback-generation/v1", status="preflight", source_commit=git(ROOT, "rev-parse", "HEAD"),
        host=socket.gethostname(), python=platform.python_version(), packages={p: importlib.metadata.version(p) for p in ("torch", "numpy", "rdkit")},
        slurm_job_id=os.environ["SLURM_JOB_ID"], started_at=datetime.now(timezone.utc).isoformat(),
        run_config_sha256=file_hash(run_path), policy_template_sha256=file_hash(policy_path),
        config=cfg, claim_limits=cfg["claim_limits"], new_calculator_calls=0, resumed=args.resume,
        prior_controller_overhead="separate_report_snapshots_or_unknown_if_interrupted" if args.resume else None)
    write_json(args.out / "report.json", report)
    try:
        query, views, attempts, pins, prior, ref_hash = prepare_inputs(args.generation, args.reference, cfg)
        inputs = dict(reference_report_sha256=ref_hash, reference_results=pins, query_sha256=identity(query))
        if manifest["inputs"] is not None and manifest["inputs"] != inputs:
            raise ValueError("resume scientific inputs changed")
        manifest["inputs"] = inputs
        write_json(manifest_path, manifest)
        policy["prior_generation_cost"] = prior
        write_json(args.out / "resolved-policy.json", policy)
        report.update(reference_report_sha256=ref_hash, reference_result_sha256=pins,
                      resolved_policy_sha256=file_hash(args.out / "resolved-policy.json"), input_sha256=identity(query),
                      fixed_pool_sha256=views[0]["pool_sha256"], status="executing")
        write_json(args.out / "report.json", report)
        template = json.loads((ROOT / "configs/research/generation_bridge_20261107.json").read_text())
        generator = GenerationActions(args.out / "generation-actions", StageBudget(**cfg["stage_budget"]),
            source=ROOT, research_source=args.research_source, model_root=args.model_root, template=template)
        tools = LiveGenerationTools(attempts, pins, query=query, reference_protocol_sha256=cfg["reference_protocol_sha256"], generator=generator)
        loop = FeedbackLoop(args.out / "decisions", views, policy, tools)
        state = loop.run()
        # A resumed loop must reverify stored results, not invoke another worker.
        before = file_hash(generator.ledger_path) if generator.ledger_path.exists() else None
        if state != loop.run() or (before and file_hash(generator.ledger_path) != before):
            raise ValueError("live action replay or cost ledger changed")
        report.update(state_sha256=identity(state), replay_identical=True, stop_reason=state["stop_reason"],
            historical_reference_cost=state["attributed_reference_cost"], prior_generation_cost=prior,
            new_generation_cost=state["generation_spent"], native_candidate_count=len(state["pending_certification"]),
            reference_channels_before_new_generation=len(state["channels"]),
            actions=[json.loads(p.read_text()) for p in sorted((args.out / "decisions").glob("step-*/action.json"))],
            generation=[{k: v for k, v in row.items() if k != "candidates"} for row in state["generation"]])
        if generator.ledger_path.exists():
            report["generation_ledger"] = RunLedger.load(generator.ledger_path).summary()
        handoffs = []
        for row in state["generation"]:
            if row["status"] == "generation_completed" and row["candidates"]:
                handoffs.append(dict(generation=f"generation-actions/{row['service_directory']}/generation",
                    generation_config=f"generation-actions/{row['service_directory']}/config.json",
                    generation_report_sha256=row["generation_report_sha256"],
                    certification_plan_sha256=row["certification_plan_sha256"]))
        write_json(args.out / "certification-handoffs.json", {"schema": "xtbflow-agent-certification-handoffs/v1", "handoffs": handoffs})
        report["handoff_sha256"] = file_hash(args.out / "certification-handoffs.json")
        report["status"] = "feedback_generation_completed_pending_certification" if handoffs else "no_new_certifiable_candidates"
    except Exception as exc:
        report.update(status="execution_failed", failure_type=type(exc).__name__)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    report.update(wall_seconds=time.monotonic() - started, decision_process_cpu_seconds=time.process_time() - cpu,
                  retained_bytes=sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()))
    if report["retained_bytes"] > cfg["max_output_bytes"] or report["wall_seconds"] > cfg["max_wall_seconds"]:
        report["status"] = "run_budget_exhausted"
    write_json(args.out / "report.json", report)
    print(json.dumps({k: report[k] for k in ("status", "wall_seconds", "new_calculator_calls")}))
    run_lock.close()
    return 0 if report["status"] == "feedback_generation_completed_pending_certification" else 1


if __name__ == "__main__":
    raise SystemExit(main())
