#!/usr/bin/env python3
"""Verify pinned O2 inputs on n2, replay analysis, and index candidate evidence.

No training or new calculator calls. Required: exact clean PR119 source checkout,
the historical runs root, and a new output directory. The frozen 600 s / 2 GiB
input / 64 MiB output caps apply. Existing outputs are never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xtbflow.governance.research_replay import (  # noqa: E402
    ReplayError, audit_inputs, candidate_links, check_source, digest, git,
)


def analysis_limits() -> None:
    """The pinned analyzer writes only results.json and its captured log."""
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CPU, (600, 600))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-source", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != "node2":
        parser.error("Real archival-data replay must run through ssh n2 on node2")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    config_path = ROOT / "configs/research/assets_20261107.json"
    config = json.loads(config_path.read_text())
    replay = config["replay"]
    asset = next(a for a in config["assets"] if a["pr"] == replay["asset_pr"])
    budget = replay["budget"]
    started = time.monotonic()
    deadline = started + budget["wall_seconds"]
    report = dict(schema="xtbflow-research-replay/v1", host=socket.gethostname(),
                  started_at=datetime.now(timezone.utc).isoformat(),
                  source_commit=git(ROOT, "rev-parse", "HEAD"),
                  source_dirty=bool(git(ROOT, "status", "--porcelain")),
                  registry_sha256=digest(config_path), budget=budget,
                  environment={"python": platform.python_version(), **{
                      name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch", "rdkit")}},
                  status="not_run", stage="preflight", input_audit=[],
                  new_calculator_calls=0, new_gpu_seconds=0,
                  claim_limits=["Replays existing exploratory records; no new quantum chemistry or independent confirmation.",
                                "Byte identity verifies analysis reproducibility, not physical correctness.",
                                "Candidate linkage begins at archived O2 inputs; fresh reactant-only generation replay remains required.",
                                "Historical GPU ledger excludes disclosed diagnostic checks; not a full generation-to-certification budget.",
                                "Archived strict statuses are preserved; no new certificate or data admission is granted."])
    try:
        if report["source_dirty"]:
            raise ReplayError("replay_source_dirty")
        report["research_source"] = check_source(args.research_source, asset)
        manifest = json.loads((args.research_source / replay["manifest"]).read_text())
        report["stage"] = "input_audit"
        audit_inputs(manifest, args.runs_root, report["input_audit"],
                     max_bytes=budget["max_input_bytes"], deadline=deadline)
        report["stage"] = "analysis"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
                   PYTHONPATH=str(args.research_source / "src") + ":" + str(args.research_source / "vendor/mechai_reusable"))
        output = args.out / "results.json"
        command = [sys.executable, replay["script"], "analyze", "--run-root",
                   str(args.runs_root / replay["run_relative"]), "--parents",
                   str(args.runs_root / replay["parents_relative"]), "--out", str(output)]
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ReplayError("analysis_budget_exhausted")
        with (args.out / "analysis.log").open("xb") as log:
            subprocess.run(command, cwd=args.research_source, env=env, stdout=log, stderr=log,
                           timeout=remaining, check=True, preexec_fn=analysis_limits)
        report["stage"] = "result_comparison"
        if sum(p.stat().st_size for p in args.out.iterdir() if p.is_file()) > budget["max_output_bytes"]:
            raise ReplayError("output_budget_exhausted")
        report["result_sha256"] = digest(output)
        report["expected_result_sha256"] = digest(args.research_source / replay["expected_result"])
        if report["result_sha256"] != report["expected_result_sha256"]:
            raise ReplayError("replay_result_mismatch")
        result = json.loads(output.read_text())
        report["stage"] = "candidate_linkage"
        report["candidates"] = candidate_links(args.runs_root / replay["run_relative"], result)
        report["certification"] = result["certification"]
        report["historical_gpu_ledger"] = result["gpu"]
        check_source(args.research_source, asset)
        report.update(status="replayed_byte_identical", stage="complete")
    except subprocess.TimeoutExpired:
        report.update(status="budget_exhausted", error="analysis_timeout")
    except subprocess.CalledProcessError as exc:
        report.update(status="execution_failed", error="subprocess_failed", returncode=exc.returncode)
    except (ReplayError, OSError, KeyError, ValueError, TypeError) as exc:
        # Raw exception strings can contain private paths. Only fixed gate codes
        # from ReplayError are public; diagnostics stay in the private run log.
        code = str(exc) if isinstance(exc, ReplayError) else type(exc).__name__
        report.update(status="budget_exhausted" if "budget_exhausted" in code else "verification_failed", error=code)
    report["wall_seconds"] = time.monotonic() - started
    usage = resource.getrusage(resource.RUSAGE_SELF)
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    report["cpu_seconds"] = usage.ru_utime + usage.ru_stime + child.ru_utime + child.ru_stime
    report["input_bytes_hashed"] = sum(r.get("bytes", 0) for r in report["input_audit"])
    # Exclusive output directory preserves failures as well as successful runs.
    with (args.out / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "stage": report["stage"]}))
    return 0 if report["status"] == "replayed_byte_identical" else 1


if __name__ == "__main__":
    raise SystemExit(main())
