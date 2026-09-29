"""Append-only search-attempt records for reaction-channel discovery."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
import re
from pathlib import Path
from typing import Any, Mapping

from .records import canonical_hash


SEARCH_ATTEMPT_SCHEMA = "xtbflow-search-attempt/v1"
EVIDENCE_STATES = frozenset(
    {
        "unattempted",
        "attempted_unresolved",
        "valid_alternative",
        "se_validated",
        "reference_single_point_labeled",
        "reference_ts_connected",
        "solution_prediction_validated",
    }
)
PROTOCOL_REQUIRED_FIELDS = frozenset(
    {
        "protocol_id",
        "protocol_sha256",
        "method",
        "software",
        "version",
        "units",
    }
)
ATTEMPT_IMMUTABLE_FIELDS = (
    "seed_id",
    "parent_system_id",
    "family_id",
    "proposal_method",
    "proposal_revision",
    "random_seed",
    "proposed_event",
    "initial_geometry_locator",
    "initial_geometry_sha256",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    result = dict(value)
    try:
        canonical_hash(result)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON-compatible") from exc
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a strict integer >= {minimum}")
    return value


def _validate_protocol(payload: Mapping[str, Any]) -> dict[str, Any]:
    protocol = _mapping(payload, "calculator_protocol")
    missing = PROTOCOL_REQUIRED_FIELDS - set(protocol)
    if missing:
        raise ValueError(
            f"calculator protocol is missing required fields: {sorted(missing)}"
        )
    for name in ("protocol_id", "method", "software", "version"):
        _text(protocol[name], f"calculator_protocol.{name}")
    if not isinstance(protocol["protocol_sha256"], str) or not _SHA256.fullmatch(
        protocol["protocol_sha256"]
    ):
        raise ValueError("calculator_protocol.protocol_sha256 must be a SHA-256")
    units = _mapping(protocol["units"], "calculator_protocol.units")
    if not units or any(
        not isinstance(key, str)
        or not key.strip()
        or not isinstance(value, str)
        or not value.strip()
        for key, value in units.items()
    ):
        raise ValueError(
            "calculator_protocol.units must map names to explicit string units"
        )
    protocol["units"] = units
    return protocol


@dataclass(frozen=True)
class SearchAttempt:
    """One immutable attempt version; later updates append a new version."""

    attempt_id: str
    attempt_version: int
    seed_id: str
    parent_system_id: str
    family_id: str
    proposal_method: str
    proposal_revision: str
    random_seed: int
    proposed_event: Mapping[str, Any]
    initial_geometry_locator: str
    initial_geometry_sha256: str
    evidence_status: str
    calculator_protocols: tuple[Mapping[str, Any], ...] = field(
        default_factory=tuple
    )
    calculator_calls: Mapping[str, int] = field(default_factory=dict)
    observed_event: Mapping[str, Any] | None = None
    wall_seconds: float = 0.0
    retry_count: int = 0
    failure_reason: str | None = None
    raw_log_locator: str | None = None
    raw_log_sha256: str | None = None
    created_at: str = ""
    schema_version: str = SEARCH_ATTEMPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "attempt_id",
            "seed_id",
            "parent_system_id",
            "family_id",
            "proposal_method",
            "proposal_revision",
            "initial_geometry_locator",
            "evidence_status",
            "created_at",
            "schema_version",
        ):
            _text(getattr(self, name), name)
        if self.schema_version != SEARCH_ATTEMPT_SCHEMA:
            raise ValueError(f"unsupported attempt schema: {self.schema_version}")
        if self.evidence_status not in EVIDENCE_STATES:
            raise ValueError(f"unsupported evidence status: {self.evidence_status}")
        _strict_int(self.attempt_version, "attempt_version", minimum=1)
        _strict_int(self.random_seed, "random_seed")
        _strict_int(self.retry_count, "retry_count")
        if not _SHA256.fullmatch(self.initial_geometry_sha256):
            raise ValueError(
                "initial_geometry_sha256 must be a lowercase SHA-256"
            )
        if (
            isinstance(self.wall_seconds, bool)
            or not isinstance(self.wall_seconds, (int, float))
            or self.wall_seconds < 0
        ):
            raise ValueError("wall_seconds must be non-negative")
        object.__setattr__(self, "wall_seconds", float(self.wall_seconds))

        proposed_event = _mapping(self.proposed_event, "proposed_event")
        if not proposed_event:
            raise ValueError("proposed_event cannot be empty")
        object.__setattr__(self, "proposed_event", proposed_event)
        if self.observed_event is not None:
            observed_event = _mapping(self.observed_event, "observed_event")
            if not observed_event:
                raise ValueError("observed_event cannot be empty when present")
            object.__setattr__(self, "observed_event", observed_event)

        protocols = tuple(
            _validate_protocol(row) for row in self.calculator_protocols
        )
        protocol_ids = [row["protocol_id"] for row in protocols]
        if len(set(protocol_ids)) != len(protocol_ids):
            raise ValueError("calculator protocol_id values must be unique")
        object.__setattr__(self, "calculator_protocols", protocols)

        calls = _mapping(self.calculator_calls, "calculator_calls")
        if any(
            not isinstance(key, str)
            or type(value) is not int
            or value < 0
            for key, value in calls.items()
        ):
            raise ValueError(
                "calculator_calls must map protocol IDs to non-negative integers"
            )
        unknown_call_keys = set(calls) - set(protocol_ids)
        if unknown_call_keys:
            raise ValueError(
                "calculator_calls reference undeclared protocols: "
                f"{sorted(unknown_call_keys)}"
            )
        object.__setattr__(self, "calculator_calls", calls)

        if self.failure_reason is not None and (
            not isinstance(self.failure_reason, str)
            or not self.failure_reason.strip()
        ):
            raise ValueError(
                "failure_reason must be null or a non-empty string"
            )
        if self.raw_log_locator is not None:
            _text(self.raw_log_locator, "raw_log_locator")
            if (
                not isinstance(self.raw_log_sha256, str)
                or not _SHA256.fullmatch(self.raw_log_sha256)
            ):
                raise ValueError(
                    "raw_log_sha256 is required when a raw log is present"
                )
        elif self.raw_log_sha256 is not None:
            raise ValueError(
                "raw_log_sha256 cannot be set without raw_log_locator"
            )

        total_calls = sum(calls.values())
        if self.evidence_status == "unattempted":
            if total_calls != 0 or protocols:
                raise ValueError(
                    "unattempted records cannot report calculator work"
                )
            if (
                self.observed_event is not None
                or self.raw_log_locator is not None
            ):
                raise ValueError(
                    "unattempted records cannot report observations"
                )
        elif total_calls > 0 and self.raw_log_locator is None:
            raise ValueError(
                "executed calculator calls require a hashed raw log"
            )
        if self.evidence_status in {
            "valid_alternative",
            "se_validated",
            "reference_single_point_labeled",
            "reference_ts_connected",
            "solution_prediction_validated",
        } and total_calls == 0:
            raise ValueError(
                f"{self.evidence_status} requires at least one calculator call"
            )
        if self.evidence_status in {
            "valid_alternative",
            "reference_ts_connected",
            "solution_prediction_validated",
        } and self.observed_event is None:
            raise ValueError(
                f"{self.evidence_status} requires a separately recorded "
                "observed_event"
            )
        if self.evidence_status == "reference_ts_connected" and not protocols:
            raise ValueError(
                "reference_ts_connected requires a calculator protocol"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "attempt_id": self.attempt_id,
            "attempt_version": self.attempt_version,
            "seed_id": self.seed_id,
            "parent_system_id": self.parent_system_id,
            "family_id": self.family_id,
            "proposal_method": self.proposal_method,
            "proposal_revision": self.proposal_revision,
            "random_seed": self.random_seed,
            "proposed_event": dict(self.proposed_event),
            "observed_event": (
                None
                if self.observed_event is None
                else dict(self.observed_event)
            ),
            "initial_geometry_locator": self.initial_geometry_locator,
            "initial_geometry_sha256": self.initial_geometry_sha256,
            "evidence_status": self.evidence_status,
            "calculator_protocols": [
                dict(row) for row in self.calculator_protocols
            ],
            "calculator_calls": dict(self.calculator_calls),
            "wall_seconds": self.wall_seconds,
            "retry_count": self.retry_count,
            "failure_reason": self.failure_reason,
            "raw_log_locator": self.raw_log_locator,
            "raw_log_sha256": self.raw_log_sha256,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SearchAttempt":
        if not isinstance(payload, Mapping):
            raise ValueError("attempt payload must be a mapping")
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unsupported attempt fields: {sorted(unknown)}")
        return cls(**dict(payload))


def _audit_attempt_history(rows: list[SearchAttempt]) -> None:
    grouped: dict[str, list[SearchAttempt]] = defaultdict(list)
    for row in rows:
        grouped[row.attempt_id].append(row)
    for attempt_id, versions in grouped.items():
        versions.sort(key=lambda row: row.attempt_version)
        actual = [row.attempt_version for row in versions]
        expected = list(range(1, len(versions) + 1))
        if actual != expected:
            raise ValueError(
                f"attempt {attempt_id} has a version gap: "
                f"expected {expected}, found {actual}"
            )
        baseline = versions[0]
        for row in versions[1:]:
            changed = [
                name
                for name in ATTEMPT_IMMUTABLE_FIELDS
                if getattr(row, name) != getattr(baseline, name)
            ]
            if changed:
                raise ValueError(
                    f"attempt {attempt_id} rewrites immutable fields: "
                    f"{changed}"
                )


def load_attempt_jsonl(path: str | Path) -> list[SearchAttempt]:
    rows: list[SearchAttempt] = []
    seen: set[tuple[str, int]] = set()
    path = Path(path)
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = SearchAttempt.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid search attempt at line {line_number}: {exc}"
                ) from exc
            key = (row.attempt_id, row.attempt_version)
            if key in seen:
                raise ValueError(
                    f"duplicate attempt version at line {line_number}: {key}"
                )
            seen.add(key)
            rows.append(row)
    rows.sort(key=lambda row: (row.attempt_id, row.attempt_version))
    _audit_attempt_history(rows)
    return rows


def append_attempt_jsonl(
    path: str | Path,
    row: SearchAttempt,
) -> None:
    """Append one version while preserving identity and version continuity."""

    path = Path(path)
    existing = load_attempt_jsonl(path)
    same_attempt = [
        item for item in existing if item.attempt_id == row.attempt_id
    ]
    expected = 1 if not same_attempt else same_attempt[-1].attempt_version + 1
    if row.attempt_version != expected:
        raise ValueError(
            f"attempt {row.attempt_id} must append version {expected}, "
            f"not {row.attempt_version}"
        )
    if same_attempt:
        baseline = same_attempt[0]
        changed = [
            name
            for name in ATTEMPT_IMMUTABLE_FIELDS
            if getattr(row, name) != getattr(baseline, name)
        ]
        if changed:
            raise ValueError(
                f"attempt {row.attempt_id} rewrites immutable fields: "
                f"{changed}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        row.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
