#!/usr/bin/env python3
"""Bounded n2 replay of historical comparison assets plus verified reference costs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
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
from xtbflow.evaluation.certified_trace import read_reference_attempt  # noqa: E402
from xtbflow.evaluation.comparison_matrix import functional_dispatch, validate_matrix  # noqa: E402
from xtbflow.governance.research_replay import audit_inputs, check_source, git, logical_path  # noqa: E402
from xtbflow.validation.certification import file_hash, write_json  # noqa: E402


def merge_manifest_inputs(*sources: list[dict]) -> list[dict]:
    """Deduplicate identical identities; conflicting expectations are errors."""
    by = {}
    for rows in sources:
        for row in rows:
            key = logical_path(row["path"]).as_posix()
            if key in by and by[key]["sha256"] != row["sha256"]:
                raise ValueError("conflicting raw input identities")
            by[key] = row
    return [by[k] for k in sorted(by)]


def worker_limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (600, 601))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16*1024**2,)*2)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enum-source", type=Path, required=True)
    parser.add_argument("--thermo-source", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--reference-panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != "node2":
        parser.error("real archived data parsing must execute through ssh n2")
    if git(ROOT, "status", "--porcelain"):
        parser.error("clean committed integration source required")
    # The archived analyzers contain this absolute locator; do not silently
    # rewrite or monkey-patch their source to make a different input look equal.
    if args.runs_root.resolve() != Path.home() / "xtbflow-runs":
        parser.error("archived analyzer locator requires the original n2 runs root")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    raw = args.out / "raw"
    raw.mkdir()
    config = ROOT / "configs/evaluation/comparison_replay_20261107.json"
    cfg = json.loads(config.read_text())
    registry = json.loads((ROOT / "configs/research/assets_20261107.json").read_text())
    start = time.monotonic()
    deadline = start + cfg["budget"]["max_wall_seconds"]
    before = resource.getrusage(resource.RUSAGE_SELF)
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    report = dict(schema=cfg["schema"], status="preflight", host=socket.gethostname(),
        source_commit=git(ROOT, "rev-parse", "HEAD"), python=platform.python_version(),
        packages={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "torch")},
        config_sha256=file_hash(config), started_at=datetime.now(timezone.utc).isoformat(),
        budget=cfg["budget"], claim_limits=cfg["claim_limits"], new_calculator_calls=0, new_gpu_seconds=0,
        inputs=[], replays=[], reference_attempts=[])
    write_json(args.out / "report.json", report)
    try:
        matrix_path = ROOT / "configs/evaluation/comparison_matrix_20261107.json"
        matrix = json.loads(matrix_path.read_text())
        profile = json.loads((ROOT / "configs/research/generation_bridge_20261107.json").read_text())
        report["matrix_sha256"] = file_hash(matrix_path)
        report["matrix_readiness"] = validate_matrix(matrix, ROOT)
        report["functional_dispatch"] = functional_dispatch(matrix, ROOT, profile, profile["methods"])
        assets = {pr: next(a for a in registry["assets"] if a["pr"] == pr) for pr in cfg["assets"]}
        sources = {113: args.enum_source.resolve(), 120: args.thermo_source.resolve()}
        report["source_audits"] = {str(pr): check_source(source, assets[pr]) for pr, source in sources.items()}
        enum_manifest = json.loads((sources[113] / "docs/evidence/review_enum/manifest.json").read_text())
        acquisition_path = args.runs_root / cfg["thermo_directory"] / "manifest.json"
        archived_acquisition = sources[120] / "docs/evidence/explore_enum_thermo/acquisition/screen.manifest.json"
        if file_hash(acquisition_path) != file_hash(archived_acquisition):
            raise ValueError("physical acquisition manifest differs from archived source")
        acquisition = json.loads(acquisition_path.read_text())
        if not acquisition["complete"]:
            raise ValueError("historical acquisition incomplete")
        frozen = [{"path": str(args.runs_root / name), "sha256": sha} for name, sha in cfg["inputs"].items()]
        physical = [{"path": str(acquisition_path.parent / (r["parent_id"] + ".jsonl")),
                     "sha256": r["output_sha256"]} for r in acquisition["parent_results"]]
        original_inputs = [{"path": name, "sha256": sha} for name, sha in acquisition["input_sha256"].items()]
        enum_inputs = [{"path": name, "sha256": sha} for name, sha in acquisition["enum_input_sha256"].items()]
        inputs = merge_manifest_inputs(enum_manifest["files"], frozen, physical, original_inputs, enum_inputs)
        audit_inputs({"files": inputs}, args.runs_root, report["inputs"],
                     max_bytes=cfg["budget"]["max_input_bytes"], deadline=deadline)
        report["status"] = "replaying"
        write_json(args.out / "report.json", report)
        root = args.runs_root / "v1a-20261007"
        enumdir = args.runs_root / cfg["enum_directory"]
        jobs = [(113, "scripts/review_enum_analyze.py", "docs/evidence/review_enum/results.json",
                 ["--root", str(root), "--enumdir", str(enumdir), "--devdir", str(args.runs_root / cfg["enum_development_directory"])]),
                (120, "scripts/followup_thermo_analyze.py", "docs/evidence/explore_enum_thermo/results.json",
                 ["--root", str(root), "--enumdir", str(enumdir), "--physical", str(acquisition_path.parent)])]
        for pr, script, expected, flags in jobs:
            source, output = sources[pr], raw / f"pr{pr}-results.json"
            env = dict(os.environ, PYTHONPATH=str(source / "src"), OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("comparison replay stage budget exhausted")
            with (raw / f"pr{pr}-analysis.log").open("wb") as log:
                result = subprocess.run([sys.executable, str(source / script), *flags, "--out", str(output)],
                    cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=remaining,
                    preexec_fn=worker_limits, check=False)
            row = dict(pr=pr, source_commit=assets[pr]["commit"], script_sha256=file_hash(source / script),
                returncode=result.returncode, expected_sha256=file_hash(source / expected),
                actual_sha256=file_hash(output) if output.exists() else None)
            row["byte_identical"] = row["actual_sha256"] == row["expected_sha256"]
            if output.exists():
                values = json.loads(output.read_text())
                row.update(parents=values["n_parents"], primary=values.get("primary", "S_N-B0"),
                           primary_proxy_comparison=values["comparisons"][values.get("primary", "S_N-B0")])
            report["replays"].append(row)
            write_json(args.out / "report.json", report)
            if result.returncode:
                raise ValueError("archived analyzer execution failed")
        panel_report = args.reference_panel / "report.json"
        if file_hash(panel_report) != "0365ceceaae80a5f2bc724dc38b30ff080c17902814c6212a381c9086f5dac9b":
            raise ValueError("reference audit requires the preserved job-4096 report")
        reference = json.loads(panel_report.read_text())
        for row in reference["cases"]:
            work = args.reference_panel / "service" / row["cache_key"] / "attempt-0001"
            attempt, provenance = read_reference_attempt(work)
            report["reference_attempts"].append(dict(attempt=asdict(attempt), provenance=provenance))
        streams = args.runs_root / "v1a-r-20261008/extract/streams.jsonl.gz"
        inventory = {}
        with gzip.open(streams, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                by = inventory.setdefault(row["arm"], dict(parent_seed_cells=0, raw_proposals=0, legacy_proxy_units=0.))
                by["parent_seed_cells"] += 1
                by["raw_proposals"] += row["n_proposals"]
                by["legacy_proxy_units"] += row["spent_units"]
        report["historical_cost_inventory"] = dict(generator_streams=inventory,
            thermo_screen_measured_cpu_seconds=acquisition["cpu_s"],
            missing=["complete per-method training/tuning cost", "complete generator/enum runtime resources",
                     "all scheduler/setup/retry/cache overhead", "common reference certification"],
            full_cost_comparison_available=False, proxy_units_are_not_seconds_or_calculator_calls=True)
        for pr, source in sources.items():
            check_source(source, assets[pr])
        report["status"] = "replay_passed" if all(r["byte_identical"] for r in report["replays"]) else "replay_mismatch"
    except subprocess.TimeoutExpired:
        report.update(status="execution_timeout", failure_type="analyzer_wall_limit")
    except Exception as exc:
        report.update(status="execution_failed", failure_type=type(exc).__name__)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    after, child = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    report.update(wall_seconds=time.monotonic()-start,
        cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime+
                    child.ru_utime+child.ru_stime-child_before.ru_utime-child_before.ru_stime,
        artifacts=[dict(path=str(p.relative_to(args.out)), bytes=p.stat().st_size, sha256=file_hash(p))
                   for p in sorted(raw.iterdir()) if p.is_file()])
    if report["wall_seconds"] > cfg["budget"]["max_wall_seconds"]:
        report["status"] = "time_budget_exhausted"
    if sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()) > cfg["budget"]["max_output_bytes"]:
        report["status"] = "storage_budget_exhausted"
    write_json(args.out / "report.json", report)
    print(json.dumps({"status": report["status"], "verified_inputs": len(report["inputs"]), "replays": len(report["replays"])}))
    return 0 if report["status"] == "replay_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
