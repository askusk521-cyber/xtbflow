"""Serializable layers for bounded independent search attempts.

The four records are intentionally separate: a frozen seed is an input, a
proposal is a reactant-side hypothesis, an attempt is a metered execution,
and validation records what was actually observed.  The proposal layer cannot
carry a product or TS answer, and a validation record cannot be manufactured
from a proposal by copying its event.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping

from .records import (
    TARGET_DERIVED_FIELD_EXACT,
    TARGET_DERIVED_FIELD_PARTS,
    canonical_hash,
    reject_target_derived_fields,
)


ATTEMPT_SCHEMA = "xtbflow-independent-search-attempt/v1"
PROPOSAL_SCHEMA = "xtbflow-independent-search-proposal/v1"
VALIDATION_SCHEMA = "xtbflow-independent-validation/v1"

SEED_STATUSES = frozenset({"not_searched", "unsafe", "unsupported"})
ATTEMPT_STATUSES = frozenset({"not_searched", "unsafe", "unsupported", "numerical_failed", "budget_exhausted", "alternative_channel", "semiempirical_connected", "reference_single_point", "reference_path"})
FORBIDDEN_TARGET_KEYS = frozenset(
    {
        "product",
        "ts",
        "ts_geometry",
        "reference_mode",
        "active_water_from_ts",
        "target_active_water",
        "sealed_test_label",
        "energy",
        "forces",
    }
    | TARGET_DERIVED_FIELD_PARTS
    | TARGET_DERIVED_FIELD_EXACT
)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _object(value: Any, name: str, *, forbid_targets: bool = False) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    result = dict(value)
    if forbid_targets:
        reject_target_derived_fields(result, name=name)
    try:
        json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON serializable") from exc
    return result


@dataclass(frozen=True)
class ProposalRecord:
    proposal_id: str
    seed_id: str
    generator: str
    raw_slot_id: int
    proposed_event: Mapping[str, Any]
    initial_geometry: Mapping[str, Any]
    status: str = "unsupported"
    generation_calls: int = 0
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "proposal_id", _text(self.proposal_id, "proposal_id"))
        object.__setattr__(self, "seed_id", _text(self.seed_id, "seed_id"))
        object.__setattr__(self, "generator", _text(self.generator, "generator"))
        if type(self.raw_slot_id) is not int or self.raw_slot_id < 0:
            raise ValueError("raw_slot_id must be a nonnegative integer")
        if self.status not in SEED_STATUSES:
            raise ValueError(f"proposal status must be one of {sorted(SEED_STATUSES)}")
        object.__setattr__(self, "proposed_event", _object(self.proposed_event, "proposed_event", forbid_targets=True))
        object.__setattr__(self, "initial_geometry", _object(self.initial_geometry, "initial_geometry", forbid_targets=True))
        object.__setattr__(self, "provenance", _object(self.provenance, "provenance", forbid_targets=True))
        object.__setattr__(self, "generation_calls", _nonnegative_int(self.generation_calls, "generation_calls"))

    @property
    def record_hash(self) -> str:
        return canonical_hash(self._payload())

    def _payload(self) -> dict[str, Any]:
        """Return the hash input without the derived hash field itself."""

        return {
            "schema": PROPOSAL_SCHEMA,
            "proposal_id": self.proposal_id,
            "seed_id": self.seed_id,
            "generator": self.generator,
            "raw_slot_id": self.raw_slot_id,
            "proposed_event": dict(self.proposed_event),
            "initial_geometry": dict(self.initial_geometry),
            "status": self.status,
            "generation_calls": self.generation_calls,
            "provenance": dict(self.provenance),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["record_hash"] = self.record_hash
        return payload


@dataclass(frozen=True)
class AttemptRecord:
    attempt_id: str
    seed_id: str
    proposal_id: str
    stage: str
    status: str
    ledger_id: str
    protocol_id: str
    calculator: str
    raw_slot_id: int
    calculator_calls: int = 0
    retry_index: int = 0
    failure_reason: str | None = None
    output_index: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("attempt_id", "seed_id", "proposal_id", "stage", "ledger_id", "protocol_id", "calculator"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.status not in ATTEMPT_STATUSES:
            raise ValueError(f"unsupported attempt status: {self.status}")
        if type(self.raw_slot_id) is not int or self.raw_slot_id < 0:
            raise ValueError("raw_slot_id must be a nonnegative integer")
        object.__setattr__(self, "calculator_calls", _nonnegative_int(self.calculator_calls, "calculator_calls"))
        object.__setattr__(self, "retry_index", _nonnegative_int(self.retry_index, "retry_index"))
        if self.failure_reason is not None:
            object.__setattr__(self, "failure_reason", _text(self.failure_reason, "failure_reason"))
        if self.output_index is not None:
            object.__setattr__(self, "output_index", _text(self.output_index, "output_index"))
        object.__setattr__(self, "metadata", _object(self.metadata, "metadata"))
        if self.status in {"numerical_failed", "budget_exhausted", "unsafe", "unsupported"} and not self.failure_reason:
            raise ValueError("failed or unsupported attempts require failure_reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ATTEMPT_SCHEMA,
            "attempt_id": self.attempt_id,
            "seed_id": self.seed_id,
            "proposal_id": self.proposal_id,
            "stage": self.stage,
            "status": self.status,
            "ledger_id": self.ledger_id,
            "protocol_id": self.protocol_id,
            "calculator": self.calculator,
            "raw_slot_id": self.raw_slot_id,
            "calculator_calls": self.calculator_calls,
            "retry_index": self.retry_index,
            "failure_reason": self.failure_reason,
            "output_index": self.output_index,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ValidationRecord:
    validation_id: str
    attempt_id: str
    observed_event: Mapping[str, Any] | None
    evidence_level: str
    protocol_id: str
    status: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("validation_id", "attempt_id", "evidence_level", "protocol_id", "status"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.observed_event is not None:
            object.__setattr__(self, "observed_event", _object(self.observed_event, "observed_event"))
        object.__setattr__(self, "metadata", _object(self.metadata, "metadata"))
        if self.status in {"reference_path", "reference_single_point", "semiempirical_connected", "alternative_channel"} and self.observed_event is None:
            raise ValueError("successful validation statuses require observed_event")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": VALIDATION_SCHEMA,
            "validation_id": self.validation_id,
            "attempt_id": self.attempt_id,
            "observed_event": None if self.observed_event is None else dict(self.observed_event),
            "evidence_level": self.evidence_level,
            "protocol_id": self.protocol_id,
            "status": self.status,
            "metadata": dict(self.metadata),
        }


def write_attempt_jsonl(path: str, records: list[AttemptRecord] | tuple[AttemptRecord, ...]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
