#!/usr/bin/env python3
"""Validate the W13 claim/comparison/evidence manifest without scientific backends."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping


SCHEMA = "xtbflow-claim-evidence-manifest/v1"
EVIDENCE_LEVELS = {
    "software_available",
    "real_closed_loop",
    "development_comparison",
    "independent_confirmation",
}
SCIENTIFIC_CLAIM_TYPES = {"joint_coupling", "physics_feedback", "adaptive_decision"}
PRIMARY_METRIC = "unique_strict_certified_channels_per_parent"
STATUSES = {"planned", "observed", "blocked", "unresolved", "not_supported"}
ACCOUNTING_STATUSES = {"complete", "partial", "unknown", "not_started"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _text(value: Any, name: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{name} must be a non-empty string")
    return value.strip()


def _hash(value: Any, name: str) -> str | None:
    if value is None:
        return None
    value = _text(value, name)
    _require(len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value), f"{name} must be a SHA-256 digest")
    return value.lower()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be an object")
    _require(all(isinstance(key, str) for key in value), f"{name} keys must be strings")
    return value


def _nonnegative(value: Any, name: str, *, integer: bool = False) -> None:
    if value is None:
        return
    if integer:
        _require(type(value) is int and value >= 0, f"{name} must be a nonnegative integer or null")
    else:
        _require(isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0, f"{name} must be nonnegative or null")


def _commit(value: Any, name: str) -> str | None:
    if value is None:
        return None
    value = _text(value, name)
    _require(bool(re.fullmatch(r"[0-9a-fA-F]{40}", value)), f"{name} must be a full 40-character commit SHA")
    return value.lower()


def _validate_source(
    value: Any,
    location: str,
    *,
    require_complete: bool = False,
    check_reachable: bool = False,
    root: Path | None = None,
) -> None:
    source = _mapping(value, location)
    required = {"code_commit", "config_sha256", "model_sha256", "protocol_sha256", "data_manifest_sha256"}
    _require(set(source) == required, f"{location} must contain exactly {sorted(required)}")
    commit = _commit(source["code_commit"], f"{location}.code_commit")
    if require_complete:
        _require(commit is not None, f"{location}.code_commit is required for observed claims")
    for key in sorted(required - {"code_commit"}):
        digest = _hash(source[key], f"{location}.{key}")
        if require_complete:
            _require(digest is not None, f"{location}.{key} is required for observed claims")
    if check_reachable and commit is not None and root is not None:
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        _require(result.returncode == 0, f"{location}.code_commit is not reachable from this repository")


def _validate_budget(value: Any, location: str, *, require_complete: bool = False) -> str:
    budget = _mapping(value, location)
    required = {"model_forwards", "calculator_calls", "cpu_core_hours", "gpu_hours", "wall_seconds", "accounting_status"}
    _require(set(budget) == required, f"{location} must contain exactly {sorted(required)}")
    _nonnegative(budget["model_forwards"], f"{location}.model_forwards", integer=True)
    _nonnegative(budget["calculator_calls"], f"{location}.calculator_calls", integer=True)
    for key in ("cpu_core_hours", "gpu_hours", "wall_seconds"):
        _nonnegative(budget[key], f"{location}.{key}")
    accounting = _text(budget["accounting_status"], f"{location}.accounting_status")
    _require(accounting in ACCOUNTING_STATUSES, f"unsupported {location}.accounting_status: {accounting}")
    if require_complete:
        for key in ("model_forwards", "calculator_calls", "cpu_core_hours", "gpu_hours", "wall_seconds"):
            _require(budget[key] is not None, f"{location}.{key} is required for observed claims")
    return accounting


def _validate_refs(
    value: Any,
    location: str,
    *,
    check_paths: bool,
    root: Path | None,
    require_hash: bool = False,
) -> int:
    _require(isinstance(value, list), f"{location} must be an array")
    for index, item in enumerate(value):
        ref = _mapping(item, f"{location}[{index}]")
        _require(set(ref) == {"kind", "path", "sha256"}, f"{location}[{index}] has unexpected fields")
        _text(ref["kind"], f"{location}[{index}].kind")
        path_text = _text(ref["path"], f"{location}[{index}].path")
        path = Path(path_text)
        _require(not path.is_absolute() and ".." not in path.parts, f"{location}[{index}].path must stay repository-relative")
        digest = _hash(ref["sha256"], f"{location}[{index}].sha256")
        if require_hash:
            _require(digest is not None, f"{location}[{index}].sha256 is required for observed claims")
        if check_paths:
            _require(root is not None, "repository root is required for path checks")
            target = root / path
            _require(target.is_file(), f"evidence path does not exist: {path_text}")
            root_resolved = root.resolve()
            target_resolved = target.resolve()
            _require(
                target_resolved == root_resolved or root_resolved in target_resolved.parents,
                f"evidence path escapes repository root: {path_text}",
            )
            if digest is not None:
                observed_digest = hashlib.sha256(target.read_bytes()).hexdigest()
                _require(observed_digest == digest.lower(), f"evidence hash mismatch: {path_text}")
    return len(value)


def repository_root(path: Path) -> Path:
    """Find the checkout root for a manifest path without depending on Git."""

    resolved = path.resolve()
    candidates = (resolved.parent, *resolved.parents)
    for candidate in candidates:
        if (candidate / "AGENTS.md").is_file() or (candidate / ".git").exists():
            return candidate
    return resolved.parent


def _validate_claim(value: Any, index: int, *, check_paths: bool, root: Path | None) -> dict[str, Any]:
    location = f"claims[{index}]"
    claim = _mapping(value, location)
    required = {
        "claim_id", "claim_type", "statement", "evidence_level", "comparison_id",
        "primary_metric", "status", "arms", "input_contract", "source", "budget",
        "evidence_refs", "claim_limit",
    }
    _require(set(claim) == required, f"{location} must contain exactly {sorted(required)}")
    claim_id = _text(claim["claim_id"], f"{location}.claim_id")
    claim_type = _text(claim["claim_type"], f"{location}.claim_type")
    _require(claim_type in SCIENTIFIC_CLAIM_TYPES | {"software_contract", "other"}, f"unsupported {location}.claim_type: {claim_type}")
    _text(claim["statement"], f"{location}.statement")
    evidence_level = _text(claim["evidence_level"], f"{location}.evidence_level")
    _require(evidence_level in EVIDENCE_LEVELS, f"unsupported {location}.evidence_level: {evidence_level}")
    comparison_id = claim["comparison_id"]
    if comparison_id is not None:
        _text(comparison_id, f"{location}.comparison_id")
    primary_metric = _text(claim["primary_metric"], f"{location}.primary_metric")
    if claim_type in SCIENTIFIC_CLAIM_TYPES:
        _require(primary_metric == PRIMARY_METRIC, f"{location}.primary_metric must be {PRIMARY_METRIC!r}")
    status = _text(claim["status"], f"{location}.status")
    _require(status in STATUSES, f"unsupported {location}.status: {status}")

    arms = claim["arms"]
    _require(isinstance(arms, list) and arms, f"{location}.arms must be a non-empty array")
    arm_ids: set[str] = set()
    for arm_index, arm_value in enumerate(arms):
        arm = _mapping(arm_value, f"{location}.arms[{arm_index}]")
        _require(set(arm) == {"arm_id", "method", "role"}, f"{location}.arms[{arm_index}] has unexpected fields")
        arm_id = _text(arm["arm_id"], f"{location}.arms[{arm_index}].arm_id")
        _require(arm_id not in arm_ids, f"duplicate arm_id in {location}: {arm_id}")
        arm_ids.add(arm_id)
        _text(arm["method"], f"{location}.arms[{arm_index}].method")
        _text(arm["role"], f"{location}.arms[{arm_index}].role")

    input_contract = _mapping(claim["input_contract"], f"{location}.input_contract")
    required_input = {"schema", "split", "group_key", "parent_count", "family_holdout"}
    _require(set(input_contract) == required_input, f"{location}.input_contract has unexpected fields")
    for key in ("schema", "split", "group_key"):
        _text(input_contract[key], f"{location}.input_contract.{key}")
    _nonnegative(input_contract["parent_count"], f"{location}.input_contract.parent_count", integer=True)
    _require(input_contract["family_holdout"] is None or type(input_contract["family_holdout"]) is bool, f"{location}.input_contract.family_holdout must be boolean or null")

    observed = status == "observed"
    _validate_source(
        claim["source"],
        f"{location}.source",
        require_complete=observed,
        check_reachable=check_paths and observed,
        root=root,
    )
    accounting = _validate_budget(claim["budget"], f"{location}.budget", require_complete=observed)
    ref_count = _validate_refs(
        claim["evidence_refs"],
        f"{location}.evidence_refs",
        check_paths=check_paths,
        root=root,
        require_hash=observed,
    )
    _text(claim["claim_limit"], f"{location}.claim_limit")

    if status == "observed":
        _require(ref_count > 0, f"{location} observed claims require evidence_refs")
        _require(accounting == "complete", f"{location} observed claims require complete budget accounting")
        _require(evidence_level != "software_available" or claim_type == "software_contract", f"{location} cannot use software_available for a scientific observed claim")
    if status in {"blocked", "unresolved", "not_supported"}:
        _require("not" in claim["claim_limit"].lower() or "blocked" in claim["claim_limit"].lower() or "unresolved" in claim["claim_limit"].lower() or "unknown" in claim["claim_limit"].lower(), f"{location} status {status} needs an explicit limiting claim_limit")
    return {"claim_id": claim_id, "comparison_id": comparison_id, "status": status, "evidence_level": evidence_level}


def validate_manifest(value: Any, *, check_paths: bool = False, root: Path | None = None) -> dict[str, Any]:
    manifest = _mapping(value, "manifest")
    required = {"schema", "manifest_id", "status", "created_utc", "claims", "claim_limits"}
    allowed = required | {"source_commit", "panel_manifest_sha256"}
    _require(set(manifest).issubset(allowed), f"manifest has unexpected fields: {sorted(set(manifest) - allowed)}")
    _require(manifest.get("schema") == SCHEMA, f"unsupported manifest schema: {manifest.get('schema')!r}")
    manifest_id = _text(manifest.get("manifest_id"), "manifest.manifest_id")
    status = _text(manifest.get("status"), "manifest.status")
    _require(status in {"draft", "ready", "blocked"}, f"unsupported manifest.status: {status}")
    _text(manifest.get("created_utc"), "manifest.created_utc")
    if "source_commit" in manifest and manifest["source_commit"] is not None:
        _text(manifest["source_commit"], "manifest.source_commit")
    if "panel_manifest_sha256" in manifest:
        _hash(manifest["panel_manifest_sha256"], "manifest.panel_manifest_sha256")
    limits = manifest.get("claim_limits")
    _require(isinstance(limits, list) and limits and all(isinstance(item, str) and item.strip() for item in limits), "manifest.claim_limits must be a non-empty string array")

    claims = manifest.get("claims")
    _require(isinstance(claims, list) and claims, "manifest.claims must be a non-empty array")
    records = [_validate_claim(item, index, check_paths=check_paths, root=root) for index, item in enumerate(claims)]
    claim_ids = [item["claim_id"] for item in records]
    _require(len(set(claim_ids)) == len(claim_ids), "claim_id values must be unique")
    comparison_ids = [item["comparison_id"] for item in records if item["comparison_id"] is not None]
    _require(len(set(comparison_ids)) == len(comparison_ids), "comparison_id values must be unique")
    if status == "ready":
        _require(all(item["status"] not in {"blocked", "unresolved", "not_supported"} for item in records), "ready manifests cannot contain blocked, unresolved, or not_supported claims")
    return {
        "schema": SCHEMA,
        "manifest_id": manifest_id,
        "status": status,
        "claim_count": len(records),
        "status_counts": {item: sum(row["status"] == item for row in records) for item in sorted(STATUSES)},
        "evidence_level_counts": {item: sum(row["evidence_level"] == item for row in records) for item in sorted(EVIDENCE_LEVELS)},
        "claim_ids": claim_ids,
    }


def load_manifest(path: Path, *, check_paths: bool = False) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return validate_manifest(value, check_paths=check_paths, root=repository_root(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--check-paths", action="store_true", help="verify repository-relative evidence paths and hashes")
    args = parser.parse_args()
    try:
        summary = load_manifest(args.manifest, check_paths=args.check_paths)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
