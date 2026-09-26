"""Deterministic group splitting and leakage audits for public records."""
from __future__ import annotations

from collections import defaultdict
import hashlib
from typing import Iterable, Mapping

from .records import PublicRecord, QUARANTINED, UNKNOWN


class SplitError(ValueError):
    """Raised when records cannot be assigned without leakage or guessing."""


def _bucket(group: str, seed: str) -> float:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_group_splits(records: Iterable[PublicRecord], *, seed: str = "xtbflow-v1", ratios: Mapping[str, float] | None = None) -> dict[str, str]:
    """Assign complete parent groups; unknown groups remain quarantined."""

    ratios = dict(ratios or {"train": 0.8, "validation": 0.1, "test": 0.1})
    if set(ratios) != {"train", "validation", "test"} or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("ratios must contain train/validation/test and sum to one")
    boundaries = (ratios["train"], ratios["train"] + ratios["validation"])
    assignments: dict[str, str] = {}
    for record in records:
        if record.parent_reaction_id in {UNKNOWN, "unavailable", "ambiguous"} or record.split_group in {UNKNOWN, "unavailable", "ambiguous"}:
            assignments[record.record_id] = QUARANTINED
            continue
        value = _bucket(record.split_group, seed)
        assignments[record.record_id] = "train" if value < boundaries[0] else "validation" if value < boundaries[1] else "test"
    return assignments


def audit_no_group_leakage(records: Iterable[PublicRecord], assignments: Mapping[str, str]) -> None:
    """Reject a group appearing in more than one non-quarantine split."""

    groups: dict[str, set[str]] = defaultdict(set)
    for record in records:
        split = assignments.get(record.record_id, QUARANTINED)
        if split != QUARANTINED:
            groups[record.split_group].add(split)
    leaking = {group: sorted(splits) for group, splits in groups.items() if len(splits) > 1}
    if leaking:
        raise SplitError(f"split-group leakage: {leaking}")


def admitted_records(records: Iterable[PublicRecord], assignments: Mapping[str, str]) -> list[PublicRecord]:
    """Return only records that passed contracts and a usable group split."""

    output: list[PublicRecord] = []
    for record in records:
        if record.admission in {"train", "validation", "test"} and assignments.get(record.record_id) == record.admission:
            output.append(record)
    return output
