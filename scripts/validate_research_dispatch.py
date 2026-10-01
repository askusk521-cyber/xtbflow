#!/usr/bin/env python3
"""Validate the machine-readable research execution dispatch contract.

The dispatch is a coordination contract, not a scientific result.  This
validator is deliberately standard-library only so that it can run in an
offline checkout.  In particular, ``--offline`` never queries GitHub or a
remote git server.  Remote pull-request fields in the manifest are therefore
treated as dated snapshots and cannot be upgraded to a current verification.

The validator checks four boundaries that are easy to lose when several
agents work in parallel:

* every artifact has one producer, a schema, a path/type and an explicit
  evidence/status boundary;
* dependencies reference artifacts (rather than whole issue numbers) and
  form a DAG;
* an artifact name cannot silently acquire two different hashes; and
* ``unknown``/``blocked`` evidence cannot be represented as a passed or ready
  scientific claim.

It does not require planned output paths to exist.  Planned and blocked
artifacts are expected in this dispatch; the returned report separates
contract validity from scientific readiness.
"""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/runs/research_dispatch_20261001.json"

SCHEMA_ID = "xtbflow-research-dispatch/v1"
WORK_PACKAGE_IDS = tuple(f"W{index:02d}" for index in range(12))
ISSUE_BY_WORK_PACKAGE = {f"W{index:02d}": 75 + index for index in range(12)}
STAGES = (
    "interface_specified",
    "implemented",
    "software_tested",
    "real_development_run",
    "chemically_validated",
    "confirmatory",
)
HANDOFF_FIELDS = (
    "work_package",
    "run_id",
    "evidence_level",
    "code_commit",
    "config_hash",
    "data_hash",
    "split_hash",
    "input_view_hash",
    "checkpoint_hash",
    "physical_protocol_hash",
    "ledger_id",
    "output_index",
    "status",
    "failure_reason",
)

# Artifact ``status`` is intentionally not called ``passed``.  A future
# artifact may be implemented or software-tested, but a scientific result is
# never silently implied by those states.
ARTIFACT_STATUSES = {
    "planned",
    "implemented",
    "software_tested",
    "blocked",
    "unknown",
    "not_run",
    "failed",
    "observed",
}
REQUIREMENT_STATUSES = ARTIFACT_STATUSES | {
    "real_development_run",
    "chemically_validated",
    "confirmatory",
}
CELL_STATES = {"observed", "planned", "blocked", "unknown", "not_run", "failed"}
EVIDENCE_LEVELS = {
    "interface_specified",
    "implemented",
    "software_tested",
    "real_development_run",
    "chemically_validated",
    "confirmatory",
    "unknown",
}
PSEUDO_PASS_VALUES = {"pass", "passed", "complete", "completed", "success", "successful"}
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

KIND_SUFFIXES = {
    "config": {".json", ".yaml", ".yml"},
    "contract": {".json", ".yaml", ".yml", ".md"},
    "doc": {".md", ".json", ".yaml", ".yml"},
    "manifest": {".json", ".jsonl", ".yaml", ".yml"},
    "report": {".json", ".jsonl", ".md"},
    "script": {".py", ".sh"},
    "test": {".py"},
    "ledger": {".json", ".jsonl", ".yaml", ".yml"},
}


class DispatchValidationError(ValueError):
    """Raised when a dispatch payload violates the coordination contract."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(str(error) for error in errors)
        super().__init__("; ".join(self.errors))


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _require_string(value: Any, label: str, errors: list[str], *, allow_unknown: bool = False) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label} must be a non-empty string")
    elif not allow_unknown and value == "unknown":
        errors.append(f"{label} must not be unknown")


def _validate_date(value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not ISO_DATE_RE.fullmatch(value):
        errors.append(f"{label} must be an ISO date (YYYY-MM-DD)")
        return
    try:
        date.fromisoformat(value)
    except ValueError:
        errors.append(f"{label} is not a calendar date: {value!r}")


def _validate_commit(value: Any, label: str, errors: list[str], *, allow_unknown: bool = False) -> None:
    if allow_unknown and value == "unknown":
        return
    if not isinstance(value, str) or not COMMIT_RE.fullmatch(value):
        errors.append(f"{label} must be a 7-40 character hexadecimal commit or unknown")


def _validate_hash(value: Any, label: str, errors: list[str], *, allow_unknown: bool = True) -> None:
    if allow_unknown and value == "unknown":
        return
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        errors.append(f"{label} must be a SHA-256 hex digest or unknown")


def _validate_relative_path(value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label} must be a non-empty repository-relative path")
        return
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        errors.append(f"{label} must stay inside the repository: {value!r}")


def _check_forbidden_pass_claims(value: Any, location: str, errors: list[str]) -> None:
    """Reject status spellings that turn unknown evidence into a pass claim."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            if key_text in {"passed", "pass", "is_passed", "scientific_pass"}:
                if child is True or (isinstance(child, str) and child.lower() in PSEUDO_PASS_VALUES):
                    errors.append(f"{location}.{key} contains a pseudo-passed claim")
            if key_text in {"status", "state", "result", "claim_status"} and isinstance(child, str):
                if child.lower() in PSEUDO_PASS_VALUES:
                    errors.append(f"{location}.{key}={child!r} is not a dispatch status")
            _check_forbidden_pass_claims(child, f"{location}.{key}", errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_forbidden_pass_claims(child, f"{location}[{index}]", errors)


def _validate_stage(stage: Any, label: str, base_commit: str, errors: list[str]) -> None:
    if not _is_mapping(stage):
        errors.append(f"{label} must be an object")
        return
    required = {"state", "evidence_level", "evidence_ref", "code_commit", "observed_on", "reason"}
    missing = sorted(required - set(stage))
    if missing:
        errors.append(f"{label} is missing fields: {missing}")
        return
    state = stage.get("state")
    if state not in CELL_STATES:
        errors.append(f"{label}.state must be one of {sorted(CELL_STATES)}")
    evidence_level = stage.get("evidence_level")
    if evidence_level not in EVIDENCE_LEVELS:
        errors.append(f"{label}.evidence_level must be one of {sorted(EVIDENCE_LEVELS)}")
    _validate_string_or_unknown(stage.get("evidence_ref"), f"{label}.evidence_ref", errors)
    _validate_commit(stage.get("code_commit"), f"{label}.code_commit", errors, allow_unknown=True)
    _validate_date(stage.get("observed_on"), f"{label}.observed_on", errors)
    _require_string(stage.get("reason"), f"{label}.reason", errors)
    if state == "observed":
        if evidence_level == "unknown":
            errors.append(f"{label} cannot be observed with unknown evidence_level")
        if stage.get("evidence_ref") in (None, "", "unknown"):
            errors.append(f"{label} observed state requires an evidence_ref")
        if stage.get("code_commit") in (None, "", "unknown"):
            errors.append(f"{label} observed state requires code_commit")
    if state in {"blocked", "unknown", "not_run", "failed"} and evidence_level == "confirmatory":
        errors.append(f"{label} cannot claim confirmatory evidence while {state}")
    if stage.get("code_commit") not in ("unknown", base_commit) and isinstance(stage.get("code_commit"), str):
        # A dispatch may be refreshed from another commit, but it must still be
        # a real commit-shaped identity.  This branch is intentionally only a
        # shape check; we do not contact a remote or rewrite history.
        _validate_commit(stage.get("code_commit"), f"{label}.code_commit", errors, allow_unknown=False)


def _validate_string_or_unknown(value: Any, label: str, errors: list[str]) -> None:
    if value is None:
        errors.append(f"{label} must be a string or explicit unknown")
    elif not isinstance(value, str) or not value.strip():
        errors.append(f"{label} must be a non-empty string or explicit unknown")


def _validate_handoff(work_package: Mapping[str, Any], label: str, errors: list[str]) -> None:
    handoff = work_package.get("handoff")
    if not _is_mapping(handoff):
        errors.append(f"{label}.handoff must be an object")
        return
    required_fields = handoff.get("required_fields")
    if tuple(required_fields or ()) != HANDOFF_FIELDS:
        errors.append(f"{label}.handoff.required_fields must equal the frozen handoff fields")
    template = handoff.get("minimum_payload")
    if not _is_mapping(template):
        errors.append(f"{label}.handoff.minimum_payload must be an object")
    else:
        missing = [field for field in HANDOFF_FIELDS if field not in template]
        if missing:
            errors.append(f"{label}.handoff.minimum_payload is missing fields: {missing}")
        if template.get("work_package") != work_package.get("work_package"):
            errors.append(f"{label}.handoff.minimum_payload.work_package does not match its work package")
        if template.get("status") not in ARTIFACT_STATUSES:
            errors.append(f"{label}.handoff.minimum_payload.status is not a valid explicit status")
        if not isinstance(template.get("output_index"), list):
            errors.append(f"{label}.handoff.minimum_payload.output_index must be a list")
        if template.get("failure_reason") in (None, ""):
            errors.append(f"{label}.handoff.minimum_payload.failure_reason must preserve unknown when unset")
    if not isinstance(handoff.get("next_handoff"), list):
        errors.append(f"{label}.handoff.next_handoff must be a list")
    elif handoff.get("next_handoff") != work_package.get("next_handoff"):
        errors.append(f"{label}.handoff.next_handoff must match the work-package next_handoff")


def _artifact_rank(status: str) -> int:
    return {
        "planned": 0,
        "blocked": 0,
        "unknown": 0,
        "not_run": 0,
        "failed": 0,
        "implemented": 1,
        "software_tested": 2,
        "observed": 2,
        "real_development_run": 3,
        "chemically_validated": 4,
        "confirmatory": 5,
    }.get(status, -1)


def _validate_artifacts(payload: Mapping[str, Any], errors: list[str]) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        errors.append("artifacts must be a non-empty list")
        return {}, {}
    artifacts: dict[str, Mapping[str, Any]] = {}
    names: dict[str, Mapping[str, Any]] = {}
    for index, artifact in enumerate(raw_artifacts):
        label = f"artifacts[{index}]"
        if not _is_mapping(artifact):
            errors.append(f"{label} must be an object")
            continue
        required = {
            "artifact_id",
            "name",
            "schema",
            "kind",
            "path",
            "producer",
            "consumers",
            "status",
            "evidence_level",
            "evidence_ref",
            "artifact_hash",
            "code_commit",
            "ready_software",
            "blocked_science",
            "failure_reason",
        }
        missing = sorted(required - set(artifact))
        if missing:
            errors.append(f"{label} is missing fields: {missing}")
            continue
        artifact_id = artifact.get("artifact_id")
        name = artifact.get("name")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            errors.append(f"{label}.artifact_id must be a non-empty string")
        elif artifact_id in artifacts:
            errors.append(f"duplicate artifact_id {artifact_id!r}")
        else:
            artifacts[artifact_id] = artifact
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{label}.name must be a non-empty string")
        elif name in names:
            previous = names[name]
            if artifact.get("artifact_hash") != previous.get("artifact_hash"):
                errors.append(
                    f"artifact name {name!r} has conflicting hashes: "
                    f"{previous.get('artifact_hash')!r} versus {artifact.get('artifact_hash')!r}"
                )
            else:
                errors.append(f"duplicate artifact name {name!r}; use one canonical artifact")
        else:
            names[name] = artifact
        _require_string(artifact.get("schema"), f"{label}.schema", errors)
        kind = artifact.get("kind")
        if kind not in KIND_SUFFIXES:
            errors.append(f"{label}.kind must be one of {sorted(KIND_SUFFIXES)}")
        _validate_relative_path(artifact.get("path"), f"{label}.path", errors)
        if isinstance(artifact.get("path"), str) and kind in KIND_SUFFIXES:
            suffix = Path(artifact["path"]).suffix.lower()
            if suffix not in KIND_SUFFIXES[kind]:
                errors.append(f"{label}.path suffix {suffix!r} is incompatible with kind {kind!r}")
        producer = artifact.get("producer")
        if producer not in WORK_PACKAGE_IDS:
            errors.append(f"{label}.producer must be a known work package")
        consumers = artifact.get("consumers")
        if not isinstance(consumers, list) or any(item not in WORK_PACKAGE_IDS for item in consumers):
            errors.append(f"{label}.consumers must contain only known work package IDs")
        status = artifact.get("status")
        if status not in ARTIFACT_STATUSES:
            errors.append(f"{label}.status must be one of {sorted(ARTIFACT_STATUSES)}")
        evidence_level = artifact.get("evidence_level")
        if evidence_level not in EVIDENCE_LEVELS:
            errors.append(f"{label}.evidence_level must be one of {sorted(EVIDENCE_LEVELS)}")
        _validate_string_or_unknown(artifact.get("evidence_ref"), f"{label}.evidence_ref", errors)
        _validate_hash(artifact.get("artifact_hash"), f"{label}.artifact_hash", errors)
        _validate_commit(artifact.get("code_commit"), f"{label}.code_commit", errors, allow_unknown=True)
        if type(artifact.get("ready_software")) is not bool:
            errors.append(f"{label}.ready_software must be boolean")
        if type(artifact.get("blocked_science")) is not bool:
            errors.append(f"{label}.blocked_science must be boolean")
        if artifact.get("failure_reason") in (None, ""):
            errors.append(f"{label}.failure_reason must use an explicit string, including unknown")
        if status == "unknown" and artifact.get("ready_software") is True:
            errors.append(f"{label} cannot be ready_software while status is unknown")
        if status in {"blocked", "unknown", "not_run", "failed"} and artifact.get("blocked_science") is not True:
            errors.append(f"{label} must keep blocked_science=true for status {status!r}")
        if status in {"implemented", "software_tested", "observed"}:
            if evidence_level == "unknown":
                errors.append(f"{label} cannot claim {status} with unknown evidence_level")
            if artifact.get("evidence_ref") in (None, "", "unknown"):
                errors.append(f"{label} {status} status requires evidence_ref")
            if artifact.get("code_commit") in (None, "", "unknown"):
                errors.append(f"{label} {status} status requires code_commit")
        if artifact.get("ready_science") is True and status not in {"real_development_run", "chemically_validated", "confirmatory"}:
            errors.append(f"{label} cannot claim ready_science from status {status!r}")
    return artifacts, names


def _validate_work_packages(payload: Mapping[str, Any], artifacts: Mapping[str, Mapping[str, Any]], errors: list[str]) -> dict[str, Mapping[str, Any]]:
    raw_work_packages = payload.get("work_packages")
    if not isinstance(raw_work_packages, list) or len(raw_work_packages) != len(WORK_PACKAGE_IDS):
        errors.append(f"work_packages must contain exactly {len(WORK_PACKAGE_IDS)} entries")
        return {}
    work_packages: dict[str, Mapping[str, Any]] = {}
    write_owners: dict[str, str] = {}
    for index, work_package in enumerate(raw_work_packages):
        label = f"work_packages[{index}]"
        if not _is_mapping(work_package):
            errors.append(f"{label} must be an object")
            continue
        required = {
            "work_package",
            "issue",
            "title",
            "owner",
            "claim_status",
            "write_scope",
            "forbidden_scope",
            "ready_now",
            "blocked_execution",
            "produces",
            "consumes",
            "status_matrix",
            "handoff",
            "next_handoff",
        }
        missing = sorted(required - set(work_package))
        if missing:
            errors.append(f"{label} is missing fields: {missing}")
            continue
        work_package_id = work_package.get("work_package")
        if work_package_id not in WORK_PACKAGE_IDS:
            errors.append(f"{label}.work_package must be one of {WORK_PACKAGE_IDS}")
            continue
        if work_package_id in work_packages:
            errors.append(f"duplicate work package {work_package_id}")
        else:
            work_packages[work_package_id] = work_package
        if work_package.get("issue") != ISSUE_BY_WORK_PACKAGE[work_package_id]:
            errors.append(f"{label}.issue does not match {work_package_id}")
        _require_string(work_package.get("title"), f"{label}.title", errors)
        _require_string(work_package.get("owner"), f"{label}.owner", errors)
        _require_string(work_package.get("claim_status"), f"{label}.claim_status", errors)
        write_scope = work_package.get("write_scope")
        if not isinstance(write_scope, list) or not write_scope or any(not isinstance(item, str) for item in write_scope):
            errors.append(f"{label}.write_scope must be a non-empty list of paths")
        else:
            for path in write_scope:
                _validate_relative_path(path, f"{label}.write_scope[{path!r}]", errors)
                previous_owner = write_owners.get(path)
                if previous_owner is not None and previous_owner != work_package_id:
                    errors.append(f"write path {path!r} has multiple owners: {previous_owner} and {work_package_id}")
                write_owners[path] = work_package_id
        forbidden_scope = work_package.get("forbidden_scope")
        if not isinstance(forbidden_scope, list) or any(not isinstance(item, str) for item in forbidden_scope):
            errors.append(f"{label}.forbidden_scope must be a list of strings")
        if type(work_package.get("ready_now")) is not bool:
            errors.append(f"{label}.ready_now must be boolean")
        _require_string(work_package.get("blocked_execution"), f"{label}.blocked_execution", errors)
        for key in ("produces", "consumes", "next_handoff"):
            value = work_package.get(key)
            if not isinstance(value, list):
                errors.append(f"{label}.{key} must be a list")
        for artifact_id in work_package.get("produces", []):
            if artifact_id not in artifacts:
                errors.append(f"{label}.produces references unknown artifact {artifact_id!r}")
            elif artifacts[artifact_id].get("producer") != work_package_id:
                errors.append(f"{label}.produces artifact {artifact_id!r} has a different producer")
        for artifact_id in work_package.get("consumes", []):
            if artifact_id not in artifacts:
                errors.append(f"{label}.consumes references unknown artifact {artifact_id!r}")
        for next_id in work_package.get("next_handoff", []):
            if next_id not in WORK_PACKAGE_IDS:
                errors.append(f"{label}.next_handoff references unknown work package {next_id!r}")
        matrix = work_package.get("status_matrix")
        if not _is_mapping(matrix):
            errors.append(f"{label}.status_matrix must be an object")
        else:
            for stage in STAGES:
                if stage not in matrix:
                    errors.append(f"{label}.status_matrix missing stage {stage!r}")
                else:
                    _validate_stage(matrix[stage], f"{label}.status_matrix.{stage}", payload.get("source", {}).get("base_commit", "unknown"), errors)
        _validate_handoff(work_package, label, errors)
    if set(work_packages) != set(WORK_PACKAGE_IDS):
        errors.append(f"work package IDs must be exactly {WORK_PACKAGE_IDS}")
    return work_packages


def _validate_dependencies(payload: Mapping[str, Any], artifacts: Mapping[str, Mapping[str, Any]], errors: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    raw_dependencies = payload.get("dependencies")
    if not isinstance(raw_dependencies, list):
        errors.append("dependencies must be a list")
        return [], []
    dependency_ids: set[str] = set()
    graph: dict[str, list[str]] = {artifact_id: [] for artifact_id in artifacts}
    unmet: list[str] = []
    for index, dependency in enumerate(raw_dependencies):
        label = f"dependencies[{index}]"
        if not _is_mapping(dependency):
            errors.append(f"{label} must be an object")
            continue
        required = {"dependency_id", "from_artifact", "to_artifact", "requirement", "required_status"}
        missing = sorted(required - set(dependency))
        if missing:
            errors.append(f"{label} is missing fields: {missing}")
            continue
        dependency_id = dependency.get("dependency_id")
        if dependency_id in dependency_ids:
            errors.append(f"duplicate dependency_id {dependency_id!r}")
        dependency_ids.add(dependency_id)
        source_id = dependency.get("from_artifact")
        target_id = dependency.get("to_artifact")
        if source_id not in artifacts:
            errors.append(f"{label}.from_artifact references unknown artifact {source_id!r}")
            continue
        if target_id not in artifacts:
            errors.append(f"{label}.to_artifact references unknown artifact {target_id!r}")
            continue
        if source_id == target_id:
            errors.append(f"{label} cannot depend on itself")
        graph[source_id].append(target_id)
        source = artifacts[source_id]
        target = artifacts[target_id]
        source_consumers = source.get("consumers", [])
        if target.get("producer") not in source_consumers:
            errors.append(
                f"{label} target producer {target.get('producer')!r} is not listed in "
                f"{source_id}.consumers"
            )
        required_status = dependency.get("required_status")
        if required_status not in REQUIREMENT_STATUSES:
            errors.append(f"{label}.required_status is not a valid artifact/evidence status")
        elif _artifact_rank(source.get("status")) < _artifact_rank(required_status):
            unmet.append(dependency_id)
        if dependency.get("artifact_hash") not in (None, "unknown"):
            if dependency.get("artifact_hash") != source.get("artifact_hash"):
                errors.append(f"{label}.artifact_hash does not match its source artifact")
    cycle = _find_cycle(graph)
    if cycle:
        errors.append("dependency graph contains a cycle: " + " -> ".join(cycle))
    return [dict(item) for item in raw_dependencies if _is_mapping(item)], unmet


def _find_cycle(graph: Mapping[str, Sequence[str]]) -> list[str] | None:
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        if node in visiting:
            try:
                start = stack.index(node)
            except ValueError:
                start = 0
            return stack[start:] + [node]
        if node in visited:
            return None
        visiting.add(node)
        stack.append(node)
        for child in graph.get(node, ()):
            cycle = visit(child)
            if cycle:
                return cycle
        stack.pop()
        visiting.remove(node)
        visited.add(node)
        return None

    for node in graph:
        cycle = visit(node)
        if cycle:
            return cycle
    return None


def _validate_source_and_verification(payload: Mapping[str, Any], errors: list[str], offline: bool) -> None:
    source = payload.get("source")
    if not _is_mapping(source):
        errors.append("source must be an object")
    else:
        for key in ("dispatch_issue", "coordination_issue", "base_commit", "review_baseline_commit", "observed_on"):
            if key not in source:
                errors.append(f"source is missing {key!r}")
        if source.get("dispatch_issue") != 74 or source.get("coordination_issue") != 75:
            errors.append("source must identify issues 74 and 75")
        _validate_commit(source.get("base_commit"), "source.base_commit", errors, allow_unknown=False)
        _validate_commit(source.get("review_baseline_commit"), "source.review_baseline_commit", errors, allow_unknown=False)
        _validate_date(source.get("observed_on"), "source.observed_on", errors)
    verification = payload.get("verification")
    if not _is_mapping(verification):
        errors.append("verification must be an object")
        return
    if verification.get("mode") not in {"offline_snapshot", "live_remote_check"}:
        errors.append("verification.mode must be offline_snapshot or live_remote_check")
    if type(verification.get("remote_checked")) is not bool:
        errors.append("verification.remote_checked must be boolean")
    if offline:
        if verification.get("mode") != "offline_snapshot":
            errors.append("--offline requires verification.mode=offline_snapshot")
        if verification.get("remote_checked") is True:
            errors.append("--offline cannot claim that remote PRs were checked live")
    snapshots = verification.get("pull_requests")
    if not isinstance(snapshots, list):
        errors.append("verification.pull_requests must be a list")
        return
    seen: set[int] = set()
    for index, snapshot in enumerate(snapshots):
        label = f"verification.pull_requests[{index}]"
        if not _is_mapping(snapshot):
            errors.append(f"{label} must be an object")
            continue
        for key in ("number", "state", "head_commit", "base_branch", "merge_commit", "merged_at", "observed_on", "claim"):
            if key not in snapshot:
                errors.append(f"{label} is missing {key!r}")
        number = snapshot.get("number")
        if not isinstance(number, int) or number <= 0:
            errors.append(f"{label}.number must be a positive integer")
        elif number in seen:
            errors.append(f"duplicate pull-request snapshot {number}")
        else:
            seen.add(number)
        if snapshot.get("state") not in {"OPEN", "CLOSED", "MERGED"}:
            errors.append(f"{label}.state must be OPEN, CLOSED or MERGED")
        _validate_commit(snapshot.get("head_commit"), f"{label}.head_commit", errors, allow_unknown=False)
        _require_string(snapshot.get("base_branch"), f"{label}.base_branch", errors)
        merge_commit = snapshot.get("merge_commit")
        if snapshot.get("state") == "MERGED":
            _validate_commit(merge_commit, f"{label}.merge_commit", errors, allow_unknown=False)
            _require_string(snapshot.get("merged_at"), f"{label}.merged_at", errors)
        elif merge_commit is not None or snapshot.get("merged_at") is not None:
            errors.append(f"{label} non-merged PR must not carry merge fields")
        _validate_date(snapshot.get("observed_on"), f"{label}.observed_on", errors)
        if snapshot.get("claim") != "snapshot_only":
            errors.append(f"{label}.claim must be snapshot_only; offline data is not current verification")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load a dispatch JSON object without resolving or touching output paths."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DispatchValidationError(["dispatch config must contain a JSON object"])
    return payload


def validate_dispatch(payload: Mapping[str, Any], *, offline: bool = False) -> dict[str, Any]:
    """Validate ``payload`` and return a non-scientific readiness report.

    ``DispatchValidationError`` is raised for contract violations.  A valid
    report may still contain blocked or not-yet-run scientific artifacts; that
    is represented by ``science_ready=False`` and explicit artifact IDs.
    """

    errors: list[str] = []
    if payload.get("schema") != SCHEMA_ID:
        errors.append(f"schema must be {SCHEMA_ID!r}")
    if payload.get("dispatch_id") != "research-dispatch-20261001":
        errors.append("dispatch_id must be research-dispatch-20261001")
    if payload.get("status") != "interface_contract_frozen_science_incomplete":
        errors.append("dispatch status must preserve interface_contract_frozen_science_incomplete")
    _check_forbidden_pass_claims(payload, "config", errors)
    _validate_source_and_verification(payload, errors, offline)
    handoff_fields = payload.get("handoff_fields")
    if tuple(handoff_fields or ()) != HANDOFF_FIELDS:
        errors.append("handoff_fields must equal the frozen minimal handoff field list")
    status_matrix = payload.get("status_matrix")
    if not _is_mapping(status_matrix) or tuple(status_matrix.get("stages") or ()) != STAGES:
        errors.append("status_matrix.stages must enumerate the six frozen evidence stages")
    budget = payload.get("shared_budget")
    if not _is_mapping(budget):
        errors.append("shared_budget must be an object")
    else:
        if budget.get("ledger_id") != "shared-stage-ledger":
            errors.append("dispatch must reference the existing shared-stage-ledger")
        if budget.get("max_concurrent_jobs") != 1:
            errors.append("dispatch must preserve max_concurrent_jobs=1")
        if budget.get("new_dft_calls") != 0:
            errors.append("dispatch contract cannot grant new DFT calls")
    artifacts, _ = _validate_artifacts(payload, errors)
    work_packages = _validate_work_packages(payload, artifacts, errors)
    dependencies, unmet = _validate_dependencies(payload, artifacts, errors)
    if errors:
        raise DispatchValidationError(errors)
    blocked = sorted(
        artifact_id
        for artifact_id, artifact in artifacts.items()
        if artifact.get("status") in {"blocked", "unknown", "not_run", "failed"}
    )
    planned = sorted(artifact_id for artifact_id, artifact in artifacts.items() if artifact.get("status") == "planned")
    ready_software = sorted(artifact_id for artifact_id, artifact in artifacts.items() if artifact.get("ready_software") is True)
    return {
        "schema": "xtbflow-research-dispatch-audit/v1",
        "valid": True,
        "science_ready": False,
        "offline": offline,
        "remote_currentity": "unknown_offline" if offline else "snapshot_only",
        "work_package_count": len(work_packages),
        "artifact_count": len(artifacts),
        "dependency_count": len(dependencies),
        "unmet_dependency_ids": sorted(unmet),
        "blocked_artifacts": blocked,
        "planned_artifacts": planned,
        "ready_software_artifacts": ready_software,
        "claim_limits": [
            "A valid dispatch proves contract shape and DAG integrity only.",
            "Offline PR fields are dated snapshots and are not current remote verification.",
            "Unknown, blocked, failed and budget-truncated artifacts remain outside scientific claims.",
        ],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--offline", action="store_true", help="Do not query or claim current remote verification.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        payload = load_config(args.config)
        report = validate_dispatch(payload, offline=args.offline)
        report["config_path"] = str(args.config)
        report["config_sha256"] = _sha256(args.config)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0
    except (OSError, json.JSONDecodeError, DispatchValidationError) as exc:
        errors = list(exc.errors) if isinstance(exc, DispatchValidationError) else [str(exc)]
        report = {
            "schema": "xtbflow-research-dispatch-audit/v1",
            "valid": False,
            "science_ready": False,
            "offline": args.offline,
            "errors": errors,
        }
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
