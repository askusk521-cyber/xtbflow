"""Source-level audit helpers for public datasets.

The repository currently has source identities and limits, but no imported raw
records.  These helpers make that gap explicit instead of treating a URL or a
historical relative path as an available file.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SourceAudit:
    source_dataset: str
    source_url: str
    source_revision: str
    source_manifest: str
    source_bytes_status: str
    source_bytes_sha256: str
    record_count_claim: int | None
    task_status: str
    claim_limit: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def hash_file(path: str | Path) -> str:
    """Hash an actually present file; callers must not pass a guessed path."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_local_asset(path: str | Path | None, expected_sha256: str | None = None) -> dict[str, str]:
    """Return observed byte status without claiming remote availability."""

    if path is None:
        return {"status": "unavailable", "sha256": "unknown"}
    candidate = Path(path)
    if not candidate.is_file():
        return {"status": "unavailable", "sha256": "unknown"}
    observed = hash_file(candidate)
    if expected_sha256 is not None and observed != expected_sha256:
        return {"status": "hash_mismatch", "sha256": observed}
    return {"status": "verified", "sha256": observed}
