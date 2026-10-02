"""Task-level availability audits for provenance-bearing public records.

The audit reports what a manifest declares and what has formal training
admission separately.  It only uses explicit availability bits and adapter
attestations; it never infers a reaction/geometry pairing from two labels that
happen to appear in the same row.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping

from .records import QUARANTINED, PublicRecord, UNKNOWN, canonical_hash


TASK_EVENT = "event_only"
TASK_GEOMETRY = "geometry_only"
TASK_ENERGY_FORCE = "energy_force"
TASK_PAIRED_JOINT = "paired_joint"
TASKS = (TASK_EVENT, TASK_GEOMETRY, TASK_ENERGY_FORCE, TASK_PAIRED_JOINT)

_UNRESOLVED = frozenset({"", UNKNOWN, "unavailable", "ambiguous", "missing", "not_provided", "unverified"})


def _explicit_bool(mask: Mapping[str, object], *keys: str) -> bool:
    """Return true only for a literal boolean true in one of ``keys``."""

    return any(type(mask.get(key)) is bool and mask[key] for key in keys)


def _parent_key(record: PublicRecord) -> str | None:
    value = record.parent_reaction_id
    if not isinstance(value, str) or value.strip().lower() in _UNRESOLVED:
        return None
    return value


def _task_flags(record: PublicRecord) -> dict[str, bool]:
    mask = record.available_label_mask
    event = _explicit_bool(mask, "event", "event_label", "event_features", "pair_edits")
    geometry = _explicit_bool(mask, "geometry", "geometry_label", "ts_geometry", "geometry_coordinates")
    energy = _explicit_bool(mask, "energy", "energy_label")
    forces = _explicit_bool(mask, "forces", "force", "forces_label")
    same_geometry = _explicit_bool(mask, "same_geometry_e_f", "energy_force_same_geometry")
    paired_identity = _explicit_bool(mask, "paired_identity", "event_geometry_pair", "paired_event_geometry")
    return {
        TASK_EVENT: event,
        TASK_GEOMETRY: geometry,
        TASK_ENERGY_FORCE: energy and forces and same_geometry,
        TASK_PAIRED_JOINT: event and geometry and paired_identity,
    }


@dataclass(frozen=True)
class TaskAvailabilityStat:
    """Counts for one task view, retaining declared and admitted evidence."""

    declared_records: int = 0
    admitted_records: int = 0
    declared_parent_reactions: int = 0
    admitted_parent_reactions: int = 0

    def __post_init__(self) -> None:
        values = (
            self.declared_records,
            self.admitted_records,
            self.declared_parent_reactions,
            self.admitted_parent_reactions,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("availability counts must be nonnegative integers")
        if self.admitted_records > self.declared_records:
            raise ValueError("admitted records cannot exceed declared records")
        if self.admitted_parent_reactions > self.declared_parent_reactions:
            raise ValueError("admitted parent reactions cannot exceed declared parents")

    def to_dict(self) -> dict[str, int]:
        return {
            "declared_records": self.declared_records,
            "admitted_records": self.admitted_records,
            "declared_parent_reactions": self.declared_parent_reactions,
            "admitted_parent_reactions": self.admitted_parent_reactions,
        }


@dataclass(frozen=True)
class TaskAvailabilityAudit:
    """Deterministic manifest audit suitable for a report or handoff artifact."""

    record_count: int
    unique_record_count: int
    parent_reaction_count: int
    identified_parent_reaction_count: int
    admitted_record_count: int
    quarantined_record_count: int
    task_stats: Mapping[str, TaskAvailabilityStat]
    admission_counts: Mapping[str, int]
    quarantine_reason_counts: Mapping[str, int]
    policy_note: str = (
        "Counts use explicit availability bits. energy_force requires an explicit same-geometry attestation; "
        "paired_joint requires an explicit event/geometry identity attestation."
    )

    def __post_init__(self) -> None:
        scalar_values = (
            self.record_count,
            self.unique_record_count,
            self.parent_reaction_count,
            self.identified_parent_reaction_count,
            self.admitted_record_count,
            self.quarantined_record_count,
        )
        if any(type(value) is not int or value < 0 for value in scalar_values):
            raise ValueError("audit counts must be nonnegative integers")
        if self.unique_record_count > self.record_count:
            raise ValueError("unique records cannot exceed record count")
        if self.identified_parent_reaction_count > self.parent_reaction_count:
            raise ValueError("identified parents cannot exceed parent count")
        if self.admitted_record_count + self.quarantined_record_count != self.record_count:
            raise ValueError("admission counts must partition records")
        if set(self.task_stats) != set(TASKS):
            raise ValueError(f"task_stats must contain exactly {TASKS}")
        if any(not isinstance(value, TaskAvailabilityStat) for value in self.task_stats.values()):
            raise ValueError("task_stats values must be TaskAvailabilityStat")
        if any(type(value) is not int or value < 0 for value in self.admission_counts.values()):
            raise ValueError("admission counts must be nonnegative integers")
        if any(type(value) is not int or value < 0 for value in self.quarantine_reason_counts.values()):
            raise ValueError("quarantine reason counts must be nonnegative integers")

    @property
    def fingerprint(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "xtbflow-task-availability-audit/v1",
            "record_count": self.record_count,
            "unique_record_count": self.unique_record_count,
            "parent_reaction_count": self.parent_reaction_count,
            "identified_parent_reaction_count": self.identified_parent_reaction_count,
            "admitted_record_count": self.admitted_record_count,
            "quarantined_record_count": self.quarantined_record_count,
            "task_stats": {task: self.task_stats[task].to_dict() for task in TASKS},
            "admission_counts": dict(sorted(self.admission_counts.items())),
            "quarantine_reason_counts": dict(sorted(self.quarantine_reason_counts.items())),
            "policy_note": self.policy_note,
            "audit_fingerprint": self.fingerprint_without_self(),
        }

    def fingerprint_without_self(self) -> str:
        payload = {
            "record_count": self.record_count,
            "unique_record_count": self.unique_record_count,
            "parent_reaction_count": self.parent_reaction_count,
            "identified_parent_reaction_count": self.identified_parent_reaction_count,
            "admitted_record_count": self.admitted_record_count,
            "quarantined_record_count": self.quarantined_record_count,
            "task_stats": {task: self.task_stats[task].to_dict() for task in TASKS},
            "admission_counts": dict(sorted(self.admission_counts.items())),
            "quarantine_reason_counts": dict(sorted(self.quarantine_reason_counts.items())),
            "policy_note": self.policy_note,
        }
        return canonical_hash(payload)

    def write_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def audit_task_availability(records: Iterable[PublicRecord]) -> TaskAvailabilityAudit:
    """Audit records without silently deduplicating or manufacturing labels."""

    rows = list(records)
    if any(not isinstance(record, PublicRecord) for record in rows):
        raise TypeError("audit_task_availability expects PublicRecord instances")

    ids = [record.record_id for record in rows]
    unique_ids = set(ids)
    parent_ids = {record.parent_reaction_id for record in rows}
    identified_parents = {_parent_key(record) for record in rows} - {None}
    admission_counts = Counter(record.admission for record in rows)
    quarantine_reasons = Counter(
        reason for record in rows for reason in record.quarantine_reasons
    )

    stats: dict[str, TaskAvailabilityStat] = {}
    for task in TASKS:
        declared_rows = [record for record in rows if _task_flags(record)[task]]
        admitted_rows = [record for record in declared_rows if record.admission != QUARANTINED]
        declared_parents = {_parent_key(record) for record in declared_rows} - {None}
        admitted_parents = {_parent_key(record) for record in admitted_rows} - {None}
        stats[task] = TaskAvailabilityStat(
            declared_records=len(declared_rows),
            admitted_records=len(admitted_rows),
            declared_parent_reactions=len(declared_parents),
            admitted_parent_reactions=len(admitted_parents),
        )

    return TaskAvailabilityAudit(
        record_count=len(rows),
        unique_record_count=len(unique_ids),
        parent_reaction_count=len(parent_ids),
        identified_parent_reaction_count=len(identified_parents),
        admitted_record_count=admission_counts.get("train", 0)
        + admission_counts.get("validation", 0)
        + admission_counts.get("test", 0)
        + admission_counts.get("diagnostic", 0),
        quarantined_record_count=admission_counts.get(QUARANTINED, 0),
        task_stats=stats,
        admission_counts=dict(admission_counts),
        quarantine_reason_counts=dict(quarantine_reasons),
    )
