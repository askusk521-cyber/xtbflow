"""Validate the machine-readable handoff/freeze contract.

The validator is deliberately local and conservative.  It checks the shape of
the handoff, verifies hashes for repository-local files, and reports readiness
separately from contract validity.  It never contacts GitHub and never turns a
missing or quarantined source into a passing freeze.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


SCHEMA_ID = "xtbflow-handoff-freeze/v1"
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ARTIFACT_NAMES = (
    "code",
    "config",
    "data",
    "split",
    "input_view",
    "protocol",
    "checkpoint",
    "ledger",
)
ARTIFACT_STATUSES = {"verified", "declared", "unknown", "missing", "not_applicable"}
SOURCE_STATUSES = {"registered", "external_only", "quarantine", "unknown", "unavailable"}
EVIDENCE_LEVELS = {"interface", "software_test", "development", "validation", "confirmation"}
EVIDENCE_STATUSES = {"unknown", "pass", "blocked", "fail"}
TOP_STATUSES = {"draft", "ready", "blocked", "failed", "completed"}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_hash(value: Any, pattern: re.Pattern[str], label: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        errors.append(f"{label} must be a lowercase hexadecimal digest")


def _safe_relative_path(locator: str) -> Path | None:
    """Return a safe repository-relative path, or ``None`` for an external locator."""

    if not locator or locator.startswith(("file:", "ssh:", "https:", "http:")):
        return None
    candidate = Path(locator)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    return candidate


def _artifact_blockers(name: str, artifact: Mapping[str, Any], root: Path, blockers: list[str], errors: list[str]) -> None:
    required = {"kind", "locator", "sha256", "status", "source_status"}
    missing = sorted(required - set(artifact))
    if missing:
        errors.append(f"artifacts.{name} missing required fields: {', '.join(missing)}")
        return

    kind = artifact.get("kind")
    if kind not in {"file", "directory", "external", "none"}:
        errors.append(f"artifacts.{name}.kind is invalid")
    locator = artifact.get("locator")
    if not _is_nonempty_string(locator):
        errors.append(f"artifacts.{name}.locator must be a non-empty string")
    status = artifact.get("status")
    if status not in ARTIFACT_STATUSES:
        errors.append(f"artifacts.{name}.status is invalid")
    source_status = artifact.get("source_status")
    if source_status not in SOURCE_STATUSES:
        errors.append(f"artifacts.{name}.source_status is invalid")

    digest = artifact.get("sha256")
    if digest is not None:
        _validate_hash(digest, SHA256_RE, f"artifacts.{name}.sha256", errors)

    if source_status != "registered":
        blockers.append(
            f"{name}: source_status={source_status!r} is not registered; freeze remains blocked"
        )
    if status != "verified" or digest is None:
        blockers.append(f"{name}: complete verified SHA-256 identity is required for a freeze")

    if not isinstance(locator, str) or kind != "file":
        return
    relative = _safe_relative_path(locator)
    if relative is None:
        if status == "verified" and source_status == "registered":
            blockers.append(f"{name}: external or unsafe locator cannot be locally verified")
        return
    target = root / relative
    if not target.is_file() or target.is_symlink():
        blockers.append(f"{name}: local artifact is missing or not a regular file ({locator})")
        return
    actual = sha256_file(target)
    if digest is None or actual != digest:
        blockers.append(f"{name}: declared SHA-256 does not match {locator}")


def validate_manifest(payload: Mapping[str, Any], *, root: Path, offline: bool = False) -> dict[str, Any]:
    """Validate a handoff payload and return a JSON-serializable audit.

    ``contract_valid`` means that the record is structurally coherent.  A
    structurally valid draft or blocked record is useful evidence and returns
    ``contract_valid=True`` with ``ready=False``.  ``--require-ready`` in the
    command-line wrapper turns any blocker into a non-zero exit status.
    """

    errors: list[str] = []
    blockers: list[str] = []
    required_top = {"schema", "manifest_id", "status", "work_package", "github", "artifacts", "evidence", "execution"}
    if not isinstance(payload, Mapping):
        return {
            "schema": SCHEMA_ID,
            "contract_valid": False,
            "ready": False,
            "errors": ["handoff manifest must be a JSON object"],
            "blocking_reasons": [],
        }
    missing_top = sorted(required_top - set(payload))
    if missing_top:
        errors.append(f"missing top-level fields: {', '.join(missing_top)}")
    if payload.get("schema") != SCHEMA_ID:
        errors.append(f"schema must be {SCHEMA_ID}")
    if not _is_nonempty_string(payload.get("manifest_id")):
        errors.append("manifest_id must be a non-empty string")
    top_status = payload.get("status")
    if top_status not in TOP_STATUSES:
        errors.append("status is invalid")

    work_package = payload.get("work_package")
    if not isinstance(work_package, Mapping):
        errors.append("work_package must be an object")
    else:
        for key in ("id", "title", "owner"):
            if not _is_nonempty_string(work_package.get(key)):
                errors.append(f"work_package.{key} must be a non-empty string")
        scope = work_package.get("write_scope")
        if not isinstance(scope, list) or not scope or any(not _is_nonempty_string(item) for item in scope):
            errors.append("work_package.write_scope must be a non-empty string list")

    github = payload.get("github")
    if not isinstance(github, Mapping):
        errors.append("github must be an object")
    else:
        for key in ("repository", "base_ref", "head_ref"):
            if not _is_nonempty_string(github.get(key)):
                errors.append(f"github.{key} must be a non-empty string")
        for key in ("base_sha", "head_sha"):
            value = github.get(key)
            if not isinstance(value, str) or not SHA1_RE.fullmatch(value):
                errors.append(f"github.{key} must be a full 40-character SHA-1")
        remote = github.get("remote_verification")
        if remote not in {"verified", "offline", "unknown"}:
            errors.append("github.remote_verification is invalid")
        if offline and remote == "verified":
            blockers.append("offline validation cannot assert that GitHub base/head refs are remotely verified")
        if remote != "verified":
            blockers.append(f"GitHub ref verification is {remote!r}; a freeze requires verified refs")

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        errors.append("artifacts must be an object")
    else:
        for name in ARTIFACT_NAMES:
            artifact = artifacts.get(name)
            if not isinstance(artifact, Mapping):
                errors.append(f"artifacts.{name} must be an object")
            else:
                _artifact_blockers(name, artifact, root, blockers, errors)
        extras = sorted(set(artifacts) - set(ARTIFACT_NAMES))
        if extras:
            errors.append(f"artifacts has unknown entries: {', '.join(extras)}")

    evidence = payload.get("evidence")
    evidence_level = None
    failure_reasons: list[str] = []
    if not isinstance(evidence, Mapping):
        errors.append("evidence must be an object")
    else:
        evidence_level = evidence.get("level")
        if evidence_level not in EVIDENCE_LEVELS:
            errors.append("evidence.level is invalid")
        evidence_status = evidence.get("status")
        if evidence_status not in EVIDENCE_STATUSES:
            errors.append("evidence.status is invalid")
        raw_reasons = evidence.get("failure_reasons")
        if not isinstance(raw_reasons, list) or any(not _is_nonempty_string(item) for item in raw_reasons):
            errors.append("evidence.failure_reasons must be a list of non-empty strings")
            raw_reasons = []
        failure_reasons = list(raw_reasons)
        if evidence_status in {"blocked", "fail"} and not failure_reasons:
            errors.append("blocked/fail evidence must include failure_reasons")
        if evidence_status == "pass" and failure_reasons:
            errors.append("pass evidence cannot include failure_reasons")
        if evidence_status != "pass":
            blockers.append(f"evidence status is {evidence_status!r}; scientific freeze is not passed")

    execution = payload.get("execution")
    if not isinstance(execution, Mapping):
        errors.append("execution must be an object")
    else:
        for key in ("command", "ledger_id", "output_index"):
            if not _is_nonempty_string(execution.get(key)):
                errors.append(f"execution.{key} must be a non-empty string")

    if top_status in {"ready", "completed"} and blockers:
        errors.append(f"status={top_status!r} is incompatible with {len(blockers)} blocking reason(s)")
    if top_status in {"blocked", "failed"} and not failure_reasons:
        errors.append(f"status={top_status!r} requires evidence.failure_reasons")
    if top_status == "draft" and evidence_level == "confirmation":
        errors.append("draft manifests cannot claim confirmation evidence")

    contract_valid = not errors
    ready = contract_valid and not blockers and top_status in {"ready", "completed"}
    return {
        "schema": "xtbflow-handoff-freeze-audit/v1",
        "manifest_id": payload.get("manifest_id"),
        "contract_valid": contract_valid,
        "ready": ready,
        "evidence_level": evidence_level,
        "failure_reasons": failure_reasons,
        "errors": errors,
        "blocking_reasons": blockers,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity": {
            "github_base_sha": (github or {}).get("base_sha") if isinstance(github, Mapping) else None,
            "github_head_sha": (github or {}).get("head_sha") if isinstance(github, Mapping) else None,
            "artifacts": {
                name: {
                    "sha256": (artifacts.get(name) or {}).get("sha256") if isinstance(artifacts, Mapping) and isinstance(artifacts.get(name), Mapping) else None,
                    "locator": (artifacts.get(name) or {}).get("locator") if isinstance(artifacts, Mapping) and isinstance(artifacts.get(name), Mapping) else None,
                    "source_status": (artifacts.get(name) or {}).get("source_status") if isinstance(artifacts, Mapping) and isinstance(artifacts.get(name), Mapping) else None,
                }
                for name in ARTIFACT_NAMES
            },
            "ledger_id": (execution or {}).get("ledger_id") if isinstance(execution, Mapping) else None,
            "output_index": (execution or {}).get("output_index") if isinstance(execution, Mapping) else None,
        },
    }


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("manifest must contain a JSON object")
    return value
