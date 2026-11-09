#!/usr/bin/env python3
"""Audit a historical live-loop manifest without replaying calculator work.

Inputs are read-only.  The output is an erratum linked by byte hashes; it does
not rewrite the original verdict, source commit, configuration, or costs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def source_identity() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def audit(manifest_path: Path, config_path: Path, evidence_path: Path) -> dict[str, Any]:
    manifest, manifest_sha = read_json(manifest_path)
    _, config_sha = read_json(config_path)
    published, published_sha = read_json(evidence_path)
    if manifest.get("schema") != "xtbflow-real-feedback-loop/v1":
        raise ValueError("expected a live-loop/v1 raw manifest")
    findings: list[dict[str, Any]] = []
    if manifest.get("config_sha256") != config_sha:
        findings.append({"reason_code": "RAW_CONFIG_HASH_MISMATCH"})
    if published.get("config_sha256") != manifest.get("config_sha256"):
        findings.append({
            "reason_code": "PUBLIC_CONFIG_HASH_MISMATCH",
            "published_config_sha256": published.get("config_sha256"),
            "raw_manifest_config_sha256": manifest.get("config_sha256"),
        })
    if published.get("source") != manifest.get("source"):
        findings.append({"reason_code": "SOURCE_IDENTITY_MISMATCH"})
    rows = []
    for arm in manifest["arms"]:
        certification = arm["certification"]
        result = certification.get("backend_result") or {}
        evidence = result.get("evidence") or {}
        endpoints = evidence.get("endpoints", [])
        gaps = []
        if evidence.get("irc_method") == "bounded_cartesian_gradient_following_v1":
            gaps.append("CARTESIAN_DESCENT_IS_NOT_INDEPENDENT_IRC")
        if len(endpoints) != 2 or any(row.get("hessian_eigenvalues") is None for row in endpoints):
            gaps.append("ENDPOINT_HESSIAN_EVIDENCE_MISSING")
        if evidence.get("reactant_endpoint_identity") is None:
            gaps.append("REACTANT_ENDPOINT_IDENTITY_MISSING")
        if gaps:
            findings.append({"reason_code": "UNSUPPORTED_STRICT_CERTIFICATION", "arm": arm["arm"], "missing_checks": gaps})
        cost = certification.get("cost") or {}
        rows.append({
            "arm": arm["arm"],
            "status": "attempted_unresolved" if gaps else "requires_independent_review",
            "reason_code": "INCOMPLETE_IRC_CERTIFICATION" if gaps else "NOT_A_CERTIFICATION_VERIFIER",
            "attempt": 1,
            "original_reported_status": certification.get("status"),
            "original_reported_reason_code": certification.get("reason_code"),
            "calculator_calls": result.get("calculator_calls"),
            "wall_seconds": cost.get("wall_seconds"),
            "cpu_hours": cost.get("cpu_hours"),
            "gpu_hours": cost.get("gpu_hours"),
            "path_frame_counts": [len(row.get("path", [])) for row in endpoints],
            "endpoint_hessians_present": [row.get("hessian_eigenvalues") is not None for row in endpoints],
            "missing_checks": gaps,
        })
    return {
        "schema": "xtbflow-live-certification-evidence-audit/v1",
        "status": "invalidated" if findings else "requires_independent_review",
        "reason_code": "EVIDENCE_INSUFFICIENT" if findings else "NOT_A_CERTIFICATION_VERIFIER",
        "evidence_level": "software_demo",
        "auditor_source": source_identity(),
        "auditor_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "audited_run_source": manifest.get("source"),
        "execution_host": platform.node(),
        "python": platform.python_version(),
        "input_hash": manifest.get("input_hash"),
        "asset_hashes": {"raw_manifest": manifest_sha, "configuration": config_sha, "published_evidence": published_sha},
        "findings": findings,
        "arms": rows,
        "counts": {"attempts_in_denominator": len(rows), "strict_certified_channels_supported": 0},
        "audit_calculator_calls": 0,
        "claim_limit": "Read-only evidence audit. Historical bytes and charged costs are retained. No strict certification or chemical negative is inferred.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--published-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("audit output already exists; choose a new path")
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        report = audit(args.manifest, args.config, args.published_evidence)
        code = 1 if report["status"] == "invalidated" else 0
    except Exception as exc:
        report = {"schema": "xtbflow-live-certification-evidence-audit/v1", "status": "execution_failed", "reason_code": "AUDIT_INPUT_FAILURE", "error": f"{type(exc).__name__}: {exc}", "evidence_level": "software_demo"}
        code = 2
    report["attempt"] = 1
    report["wall_seconds"] = time.perf_counter() - wall
    report["cpu_hours"] = (time.process_time() - cpu) / 3600.0
    report["gpu_hours"] = 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "reason_code": report["reason_code"]}, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
