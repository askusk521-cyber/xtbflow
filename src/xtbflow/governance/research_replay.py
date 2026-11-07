"""Bounded, read-only replay of pinned research evidence, without QC imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any


class ReplayError(ValueError):
    """A failed provenance or replay gate; never a chemical failure."""


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def content_id(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *args],
                                   text=True, stderr=subprocess.PIPE, timeout=30).strip()


def check_source(source: Path, asset: dict) -> dict:
    """Require the exact clean revision; do not change an existing checkout."""
    sha = git(source, "rev-parse", "HEAD")
    if sha != asset["commit"]:
        raise ReplayError("research_source_revision_mismatch")
    if git(source, "status", "--porcelain", "--untracked-files=normal"):
        raise ReplayError("research_source_dirty")
    for row in asset["files"]:
        path = source / row["path"]
        if path.is_symlink() or digest(path) != row["sha256"]:
            raise ReplayError("research_source_file_mismatch")
    return {"commit": sha, "dirty": False, "pinned_files": len(asset["files"])}


def logical_path(archived: str) -> Path:
    """Relocate historical n2 paths under an explicitly supplied runs root."""
    parts = Path(archived).parts
    if parts.count("xtbflow-runs") != 1 or ".." in parts:
        raise ReplayError("unrecognized_archive_root")
    relative = Path(*parts[parts.index("xtbflow-runs") + 1:])
    if not relative.parts:
        raise ReplayError("empty_archive_path")
    return relative


def contained(root: Path, relative: Path) -> Path:
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ReplayError("missing_or_outside_run_root")
    return path


def audit_inputs(manifest: dict, runs_root: Path, rows: list[dict], *,
                 max_bytes: int, deadline: float) -> None:
    """Keep every completed check in rows, including failures and spent I/O."""
    used = 0
    seen: dict[str, str] = {}
    for source in manifest["files"]:
        relative = logical_path(source["path"])
        key = relative.as_posix()
        if key in seen:
            raise ReplayError("duplicate_manifest_path")
        seen[key] = source["sha256"]
        row = {"asset": key, "expected_sha256": source["sha256"], "status": "not_run"}
        rows.append(row)
        try:
            path = contained(runs_root, relative)
        except ReplayError:
            row["status"] = "unavailable"
            raise
        size = path.stat().st_size
        if used + size > max_bytes or time.monotonic() >= deadline:
            row["status"] = "budget_exhausted"
            raise ReplayError("input_audit_budget_exhausted")
        row.update(bytes=size, actual_sha256=digest(path))
        used += size
        row["status"] = "verified" if row["actual_sha256"] == source["sha256"] else "hash_mismatch"
        if row["status"] != "verified":
            raise ReplayError("raw_input_hash_mismatch")


def candidate_links(run_root: Path, result: dict) -> list[dict]:
    """Index archival geometry/event/cost evidence; do not issue new certificates.

    Endpoint event IDs are copied from observed endpoint records, never from
    proposed events. Physical quantities retain their original protocol units.
    """
    plan = json.loads((run_root / "o2-plan.json").read_text())
    costs = {r["item_id"]: r for r in result["items"]}
    if len(costs) != len(result["items"]) or len({i["item_id"] for i in plan["items"]}) != len(plan["items"]):
        raise ReplayError("duplicate_item_identity")
    if set(costs) != {i["item_id"] for i in plan["items"]}:
        raise ReplayError("result_plan_identity_mismatch")
    links = []
    for item in plan["items"]:
        if item["kind"] != "candidate":
            continue
        work = run_root / "o2-run" / f"candidate_{item['item_id']}"
        manifest = json.loads((work / "manifest.json").read_text())
        if manifest["item"] != item:
            raise ReplayError("candidate_manifest_identity_mismatch")
        verdict_path = work / "verdict.json"
        verdict = json.loads(verdict_path.read_text()) if verdict_path.exists() else {}
        identity, ts = verdict.get("identity", {}), verdict.get("ts", {})
        cost = costs[item["item_id"]]
        if cost["parent_id"] != item["parent_id"] or cost["kind"] != "candidate":
            raise ReplayError("candidate_cost_identity_mismatch")
        if manifest["slurm_job_id"] != cost["slurm_job_id"]:
            raise ReplayError("candidate_job_identity_mismatch")
        links.append({
            "candidate_id": item["candidate_id"], "item_id": item["item_id"],
            "parent_id": item["parent_id"], "query_id": item["query_id"],
            "input_sha256": content_id({k: item[k] for k in
                ("z", "x_start", "charge", "multiplicity", "coordinate_unit", "b_r", "b_predicted", "perms")}),
            "charge": item["charge"], "multiplicity": item["multiplicity"],
            "coordinate_unit": item["coordinate_unit"], "energy_unit": "Hartree",
            "frequency_unit": "cm^-1", "method": verdict.get("method"),
            "protocol_version": manifest["protocol_version"],
            "execution_commit": manifest["source_commit"],
            "planned_product_be_sha256": content_id(item["b_predicted"]),
            "planned_event_id": identity.get("predicted_event_id"),
            "observed_event_id": identity.get("actual_event_id"),
            "ts_geometry_sha256": content_id(ts["x_ts"]) if "x_ts" in ts else None,
            "ts_status": ts.get("status"), "irc_status": verdict.get("irc", {}).get("status"),
            "endpoint_geometry_sha256": [content_id(e["x"]) for e in verdict.get("endpoints", []) if "x" in e],
            "endpoint_statuses": [e["status"] for e in verdict.get("endpoints", [])],
            "archived_status": cost["status"],
            "historical_cost": {k: cost[k] for k in ("calls", "slurm_job_id", "slurm_runtime_s", "wall_s")},
            "evidence_kind": "archival_linkage_only",
        })
    return links
