#!/usr/bin/env python3
"""Rebuild real-source task counts and test hidden-label invariance on n2."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import importlib.metadata
import json
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

from xtbflow.data.paired_loader import load_paired_pilot  # noqa: E402
from xtbflow.data.records import canonical_hash  # noqa: E402
from xtbflow.data.task_manifest import TaskLoader, build_task_manifest, digest  # noqa: E402


def timeout_handler(signum, frame):
    raise TimeoutError("task_view_audit_wall_limit")


def replacement_check(sample, seed: int) -> dict:
    import torch
    from xtbflow.evaluation import BlindRolloutConfig, blind_rollout, make_reactant_input
    from xtbflow.models import JointEventGeometryFlow, total_electron_projector
    changed = replace(sample, product_smiles="counterfactual-hidden-product",
                      product_coordinates=sample.product_coordinates + 7,
                      product_bonds=sample.product_bonds + 2,
                      ts_coordinates=sample.ts_coordinates - 3,
                      record=replace(sample.record,
                        event_label={**sample.record.event_label, "sha256": "9" * 64},
                        ts_geometry={**sample.record.ts_geometry, "sha256": "8" * 64},
                        product_label={**sample.record.product_label, "graph_sha256": "7" * 64}))
    first = next(iter(TaskLoader([sample], task="paired_joint", split="train", purpose="training")))
    other = next(iter(TaskLoader([changed], task="paired_joint", split="train", purpose="training")))
    inputs_equal = all(torch.equal(v, other.reactant_input[k]) for k, v in first.reactant_input.items())
    targets_changed = not torch.equal(first.targets["ts_coordinates"], other.targets["ts_coordinates"])
    n = len(sample.atomic_numbers)
    left, right = make_reactant_input(sample, n), make_reactant_input(changed, n)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = JointEventGeometryFlow(total_electron_projector(left.symbols), node_feature_dim=8,
                                       condition_feature_dim=4, hidden_dim=8, radial_features=4)
    config = BlindRolloutConfig(nfe=2, candidate_cap=2)
    a = blind_rollout(model, left, arm_id="joint_bidirectional", seed=seed, config=config)
    b = blind_rollout(model, right, arm_id="joint_bidirectional", seed=seed, config=config)
    return dict(record_id=sample.record.record_id, parent_id=sample.record.parent_reaction_id,
                input_identical=inputs_equal, hidden_targets_changed=targets_changed,
                attempts_identical=a.attempts == b.attempts,
                generated_output_sha256=canonical_hash(a.to_dict()),
                flow_evaluations=a.actual_flow_evaluations + b.actual_flow_evaluations,
                candidate_attempts=a.actual_candidate_attempts + b.actual_candidate_attempts,
                passed=inputs_equal and targets_changed and a.attempts == b.attempts,
                scope="Randomly initialized fixed model; software input-isolation diagnostic only")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != "node2":
        parser.error("real source reconstruction must run through ssh n2")
    git = lambda *a: subprocess.check_output(["git", "-C", str(ROOT), *a], text=True).strip()
    if git("status", "--porcelain"):
        parser.error("clean committed source required")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    resource.setrlimit(resource.RLIMIT_CPU, (600, 601))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2, 16 * 1024**2))
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(600)
    report = dict(schema="xtbflow-task-view-audit/v1", source_commit=git("rev-parse", "HEAD"),
                  audit_protocol_sha256=digest(Path(__file__)),
                  host=socket.gethostname(), started_at=datetime.now(timezone.utc).isoformat(),
                  environment={"python": platform.python_version(), **{p: importlib.metadata.version(p) for p in ("numpy", "torch", "rdkit")}},
                  status="preflight", cpu_limit_seconds=600, wall_limit_seconds=600, new_calculator_calls=0,
                  new_gpu_seconds=0, training_steps=0, invariance_cases=[],
                  claim_limits=["Reuses existing development admission and splits; no confirmation or new source admission.",
                                "No energy/force labels are provided by this source adapter.",
                                "Task counts overlap; event/geometry/joint rows are not independent datasets.",
                                "A fixed randomly initialized model tests software input isolation, not predictive accuracy.",
                                "Declared partial conditioning, multi-fidelity ingestion and certified feedback promotion remain open."])
    try:
        manifest_path = ROOT / "data/manifests/paired_pilot_v1/manifest.jsonl"
        report["source_manifest_sha256"] = digest(manifest_path)
        source = load_paired_pilot(manifest_path, args.cache)
        index = build_task_manifest([s.record for s in source.samples], version="dft-da-task-views-20261107-v1",
                                    source_manifest_sha256=report["source_manifest_sha256"])
        report["counts"] = index["counts"]
        report["split_audit"] = index["split_audit"]
        report["source_assets"] = sorted({s.record.source_asset_sha256 for s in source.samples})
        report["task_batch_shapes"] = {}
        for task in ("event_only", "geometry_only", "paired_joint"):
            batch = next(iter(TaskLoader(source.samples, task=task, split="train", purpose="training")))
            views = batch.training_views()
            report["task_batch_shapes"][task] = {section: {k: list(v.shape) for k, v in views[section].items()}
                                                for section in ("reactant_input", "training_supervision")}
        parents = set()
        for sample in source.samples:
            if sample.record.admission != "development_train" or sample.record.parent_reaction_id in parents:
                continue
            parents.add(sample.record.parent_reaction_id)
            report["invariance_cases"].append(replacement_check(sample, 132))
            if len(parents) == 3:
                break
        if not report["invariance_cases"] or not all(c["passed"] for c in report["invariance_cases"]):
            raise ValueError("hidden_label_invariance_failed")
        index_path = args.out / "task_manifest.json"
        with index_path.open("x") as stream:
            json.dump(index, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        report["task_manifest_sha256"] = digest(index_path)
        report["status"] = "task_views_verified"
    except Exception as exc:
        report.update(status="audit_failed", failure_type=type(exc).__name__)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        signal.alarm(0)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report["wall_seconds"] = time.monotonic() - started
        report["cpu_seconds"] = usage.ru_utime + usage.ru_stime
        with (args.out / "report.json").open("x") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"status": report["status"], "cases": len(report["invariance_cases"])}))
    return 0 if report["status"] == "task_views_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
