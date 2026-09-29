"""Path-safe evidence summaries for bounded C2 calculator runs."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any, Mapping


_PRIVATE_PATH = re.compile(
    r"(?<![:A-Za-z0-9_])(?:/(?:[^\s\"'`,;)}\]]+/)+[^\s\"'`,;)}\]]+|[A-Za-z]:[\\/])[^\s\"'`,;)}\]]*"
)


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitize_error_message(value: Any) -> str | None:
    """Remove host-local absolute paths from a public error string."""

    if value is None:
        return None
    return _PRIVATE_PATH.sub("[private-path]", str(value))


def sanitize_public_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): sanitize_public_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_public_value(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_public_value(item) for item in value]
    if isinstance(value, str):
        return sanitize_error_message(value)
    return value


def index_artifacts(root: str | Path) -> list[dict[str, Any]]:
    """Hash regular files below ``root`` without exposing the absolute root."""

    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise ValueError(f"artifact root is not a directory: {base}")
    records: list[dict[str, Any]] = []
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact index refuses symbolic links: {path.name}")
        if not path.is_file():
            continue
        records.append(
            {
                "relative_path": path.relative_to(base).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not records:
        raise ValueError("artifact root contains no regular files")
    return records


def _sanitized_case(case: Mapping[str, Any]) -> dict[str, Any]:
    result = sanitize_public_value(dict(case))
    metadata = dict(result.get("metadata", {}))
    if metadata.pop("artifact_directory", None) is not None:
        metadata["artifact_directory_recorded_in_private_run"] = True
    result["metadata"] = metadata
    return result


def sanitize_cp2k_calibration_report(
    payload: Mapping[str, Any],
    *,
    execution_source_commit: str,
    publication_source_commit: str,
    private_report_sha256: str,
    calibration_script_sha256: str,
    adapter_sha256: str,
    artifacts: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a public calibration record with hashes but no host paths."""

    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("calibration report must contain nonempty cases")
    if not execution_source_commit.strip():
        raise ValueError("execution_source_commit is required")
    if not publication_source_commit.strip():
        raise ValueError("publication_source_commit is required")
    public = {
        key: payload[key]
        for key in (
            "schema",
            "status",
            "calculator",
            "cp2k_version",
            "source_revision",
            "build_hash",
            "protocol_family",
            "cutoff_ry",
            "runtime_threads",
            "scope",
        )
        if key in payload
    }
    public.update(
        {
            "evidence_schema": "xtbflow-cp2k-calibration-public/v1",
            "execution_source_commit": execution_source_commit,
            "publication_source_commit": publication_source_commit,
            "private_report_sha256": private_report_sha256,
            "calibration_script_sha256": calibration_script_sha256,
            "cp2k_adapter_sha256": adapter_sha256,
            "cases": [_sanitized_case(case) for case in cases],
            "artifact_index": [dict(item) for item in artifacts],
            "scientific_qualification": False,
            "claim_limits": [
                "This is a real installation/state/element E/F replay with persistent artifacts.",
                "It does not establish cutoff or basis convergence, broad chemical accuracy, transition-state validity, or endpoint connectivity.",
                "The full private artifact root is intentionally omitted; relative filenames, sizes, and SHA-256 values are retained.",
            ],
        }
    )
    return sanitize_public_value(public)


def sanitize_cp2k_convergence_report(
    payload: Mapping[str, Any],
    *,
    execution_source_commit: str,
    publication_source_commit: str,
    private_report_sha256: str,
    ledger_sha256: str,
    convergence_script_sha256: str,
    adapter_sha256: str,
    artifacts: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a path-safe public record for one metered cutoff ladder."""

    runs = payload.get("runs")
    comparisons = payload.get("comparisons_to_highest_cutoff")
    if not isinstance(runs, list) or len(runs) < 2:
        raise ValueError("convergence report must contain at least two runs")
    if not isinstance(comparisons, list):
        raise ValueError("convergence comparisons must be a list")
    if not isinstance(execution_source_commit, str) or not execution_source_commit.strip():
        raise ValueError("execution_source_commit is required")
    if not isinstance(publication_source_commit, str) or not publication_source_commit.strip():
        raise ValueError("publication_source_commit is required")
    public = {
        key: payload[key]
        for key in (
            "schema",
            "status",
            "scientific_qualification",
            "runtime",
            "system",
            "runtime_threads",
            "energy_tolerance_hartree",
            "force_tolerance_hartree_per_angstrom",
            "cutoffs_ry",
            "calculator_calls",
            "scope",
            "claim_limits",
        )
        if key in payload
    }
    public.update(
        {
            "evidence_schema": "xtbflow-cp2k-convergence-public/v1",
            "execution_source_commit": execution_source_commit,
            "publication_source_commit": publication_source_commit,
            "private_report_sha256": private_report_sha256,
            "ledger_sha256": ledger_sha256,
            "convergence_script_sha256": convergence_script_sha256,
            "cp2k_adapter_sha256": adapter_sha256,
            "runs": [_sanitized_case(run) for run in runs],
            "comparisons_to_highest_cutoff": [
                dict(item) for item in comparisons
            ],
            "artifact_index": [dict(item) for item in artifacts],
        }
    )
    public.pop("artifact_root", None)
    public["scientific_qualification"] = False
    limits = list(public.get("claim_limits", []))
    limits.append(
        "The private artifact root is omitted; relative paths, sizes, hashes, and the metered ledger hash are retained."
    )
    public["claim_limits"] = limits
    return sanitize_public_value(public)
