"""Deterministic offline budget scheduling for bounded research comparisons.

The scheduler chooses work from a frozen candidate pool. It records the actual
outcome and cost supplied by an executor, but never calls a calculator or
interprets an outcome as chemical evidence. This keeps fixed and simple
heuristic allocation comparable before live validation services are available.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping


SCHEDULER_SCHEMA = "xtbflow-budget-scheduler/v1"
STRATEGIES = frozenset({"fixed", "heuristic"})
ACTION_KINDS = frozenset(
    {
        "new_event",
        "new_geometry",
        "local_refinement",
        "curvature_probe",
        "reference_label",
        "connectivity_certification",
    }
)
OUTCOME_STATUSES = frozenset(
    {"success", "failure", "timeout", "cancelled", "unresolved", "duplicate"}
)
LIFECYCLE_STATES = frozenset({"running", "stopped", "completed", "budget_exhausted"})
FAILURE_STATUSES = frozenset({"failure", "timeout", "cancelled", "unresolved"})


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _json_mapping(value: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    result = dict(value)
    try:
        json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite JSON-compatible data") from exc
    return result


@dataclass(frozen=True)
class BudgetCandidate:
    """One predeclared action in a shared candidate pool.

    ``estimated_cost_units`` is an abstract, frozen accounting unit. It must
    be mapped to the same calculator/resource accounting for every strategy in
    a comparison. ``information_gain`` is an input-side heuristic hint; it is
    never inferred from a result and never treated as a scientific label.
    """

    candidate_id: str
    family_id: str
    action_kind: str
    estimated_cost_units: int
    priority: int = 0
    information_gain: float = 0.0
    channel_key: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("candidate_id", "family_id", "action_kind"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.action_kind not in ACTION_KINDS:
            raise ValueError(f"unsupported action_kind: {self.action_kind}")
        _positive_int(self.estimated_cost_units, "estimated_cost_units")
        _nonnegative_int(self.priority, "priority")
        if (
            isinstance(self.information_gain, bool)
            or not isinstance(self.information_gain, (int, float))
            or not math.isfinite(float(self.information_gain))
            or self.information_gain < 0
        ):
            raise ValueError("information_gain must be finite and non-negative")
        object.__setattr__(self, "information_gain", float(self.information_gain))
        if self.channel_key is not None:
            object.__setattr__(self, "channel_key", _text(self.channel_key, "channel_key"))
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, "metadata"))


@dataclass(frozen=True)
class SchedulerAction:
    """A selected action awaiting an executor outcome."""

    sequence: int
    candidate: BudgetCandidate

    def __post_init__(self) -> None:
        _positive_int(self.sequence, "sequence")
        if not isinstance(self.candidate, BudgetCandidate):
            raise TypeError("candidate must be a BudgetCandidate")


@dataclass(frozen=True)
class SchedulerRecord:
    """One settled action, including failures and duplicate skips."""

    sequence: int
    candidate_id: str
    action_kind: str
    status: str
    estimated_cost_units: int
    charged_cost_units: int
    channel_key: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        _positive_int(self.sequence, "sequence")
        _text(self.candidate_id, "candidate_id")
        _text(self.action_kind, "action_kind")
        if self.action_kind not in ACTION_KINDS:
            raise ValueError(f"unsupported action_kind: {self.action_kind}")
        if self.status not in OUTCOME_STATUSES:
            raise ValueError(f"unsupported scheduler outcome: {self.status}")
        _positive_int(self.estimated_cost_units, "estimated_cost_units")
        _nonnegative_int(self.charged_cost_units, "charged_cost_units")
        if self.channel_key is not None:
            object.__setattr__(self, "channel_key", _text(self.channel_key, "channel_key"))
        if self.reason is not None:
            object.__setattr__(self, "reason", _text(self.reason, "reason"))
        if self.status == "success" and self.reason is not None:
            raise ValueError("successful records cannot carry a reason")
        if self.status == "duplicate" and self.charged_cost_units != 0:
            raise ValueError("duplicate records must have zero charged cost")
        if self.status in FAILURE_STATUSES and self.reason is None:
            raise ValueError(f"{self.status} records require a reason")


class SchedulerBudgetExceeded(RuntimeError):
    """Raised when a settled action would exceed the frozen budget."""


class BudgetScheduler:
    """A resumable, calculator-free scheduler with an append-only outcome log.

    The scheduler owns selection and accounting only. Callers execute the
    returned :class:`SchedulerAction`, then call :meth:`settle` with the actual
    status and cost. A stop leaves an in-flight action intact; resuming returns
    that same action so work cannot silently disappear across a restart.
    """

    schema = SCHEDULER_SCHEMA

    def __init__(
        self,
        candidates: Iterable[BudgetCandidate],
        budget_units: int,
        *,
        strategy: str = "fixed",
    ) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"unsupported scheduler strategy: {strategy}")
        self.budget_units = _positive_int(budget_units, "budget_units")
        self.strategy = strategy
        self._candidates = tuple(candidates)
        if any(not isinstance(candidate, BudgetCandidate) for candidate in self._candidates):
            raise TypeError("candidates must contain BudgetCandidate values")
        ids = [candidate.candidate_id for candidate in self._candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate_id values must be unique")
        self._candidate_by_id = {candidate.candidate_id: candidate for candidate in self._candidates}
        self._pending: dict[str, BudgetCandidate] = dict(self._candidate_by_id)
        self._records: list[SchedulerRecord] = []
        self._spent_units = 0
        self._accepted_channels: set[str] = set()
        self._next_sequence = 1
        self._in_flight: SchedulerAction | None = None
        self._lifecycle = "running"
        self._stop_reason: str | None = None

    @property
    def status(self) -> str:
        return self._lifecycle

    @property
    def spent_units(self) -> int:
        return self._spent_units

    @property
    def remaining_units(self) -> int:
        return self.budget_units - self._spent_units

    @property
    def records(self) -> tuple[SchedulerRecord, ...]:
        return tuple(self._records)

    @property
    def pending_candidates(self) -> tuple[BudgetCandidate, ...]:
        return tuple(candidate for candidate in self._candidates if candidate.candidate_id in self._pending)

    @property
    def in_flight(self) -> SchedulerAction | None:
        return self._in_flight

    @property
    def stop_reason(self) -> str | None:
        return self._stop_reason

    @property
    def accepted_channels(self) -> tuple[str, ...]:
        return tuple(sorted(self._accepted_channels))

    def _heuristic_key(self, candidate: BudgetCandidate) -> tuple[float, int, str]:
        score = candidate.information_gain / candidate.estimated_cost_units
        return (-score, candidate.priority, candidate.candidate_id)

    def _choose_candidate(self) -> BudgetCandidate | None:
        pending = list(self._pending.values())
        if not pending:
            return None
        if self.strategy == "fixed":
            return next(candidate for candidate in self._candidates if candidate.candidate_id in self._pending)
        fitting = [candidate for candidate in pending if candidate.estimated_cost_units <= self.remaining_units]
        if not fitting:
            return None
        return sorted(fitting, key=self._heuristic_key)[0]

    def _append_record(self, record: SchedulerRecord) -> None:
        candidate = self._candidate_by_id.get(record.candidate_id)
        if candidate is None:
            raise ValueError(f"record references unknown candidate: {record.candidate_id}")
        if candidate.action_kind != record.action_kind or candidate.estimated_cost_units != record.estimated_cost_units:
            raise ValueError("record does not match the frozen candidate contract")
        if record.candidate_id not in self._pending:
            raise ValueError(f"candidate already settled: {record.candidate_id}")
        if record.charged_cost_units > self.remaining_units:
            raise SchedulerBudgetExceeded(
                f"charged_cost_units={record.charged_cost_units} exceeds remaining budget {self.remaining_units}"
            )
        if record.status == "success" and record.channel_key in self._accepted_channels:
            raise ValueError("successful record would duplicate an accepted channel")
        self._records.append(record)
        self._pending.pop(record.candidate_id)
        self._spent_units += record.charged_cost_units
        if record.status == "success" and record.channel_key is not None:
            self._accepted_channels.add(record.channel_key)
        self._next_sequence = max(self._next_sequence, record.sequence + 1)

    def next_action(self) -> SchedulerAction | None:
        """Return the next deterministic action, or ``None`` at a boundary.

        Candidates whose declared ``channel_key`` was already accepted are
        recorded as zero-cost duplicates before the next action is returned.
        A fixed schedule stops at its first over-budget candidate; a heuristic
        schedule chooses any candidate that fits, then stops when none fits.
        """

        if self._in_flight is not None:
            return self._in_flight
        if self._lifecycle != "running":
            return None
        while self._pending:
            candidate = self._choose_candidate()
            if candidate is None:
                self._lifecycle = "budget_exhausted"
                self._stop_reason = "no pending candidate fits the remaining budget"
                return None
            if candidate.channel_key is not None and candidate.channel_key in self._accepted_channels:
                self._append_record(
                    SchedulerRecord(
                        sequence=self._next_sequence,
                        candidate_id=candidate.candidate_id,
                        action_kind=candidate.action_kind,
                        status="duplicate",
                        estimated_cost_units=candidate.estimated_cost_units,
                        charged_cost_units=0,
                        channel_key=candidate.channel_key,
                        reason="channel_already_accepted",
                    )
                )
                continue
            if candidate.estimated_cost_units > self.remaining_units:
                self._lifecycle = "budget_exhausted"
                self._stop_reason = (
                    f"candidate {candidate.candidate_id} requires "
                    f"{candidate.estimated_cost_units} units but only {self.remaining_units} remain"
                )
                return None
            self._in_flight = SchedulerAction(self._next_sequence, candidate)
            return self._in_flight
        self._lifecycle = "completed"
        return None

    def settle(
        self,
        action: SchedulerAction,
        status: str,
        charged_cost_units: int,
        *,
        channel_key: str | None = None,
        reason: str | None = None,
    ) -> SchedulerRecord:
        """Record an executor outcome and advance the scheduler.

        ``charged_cost_units`` is the measured accounting cost, including a
        failed or timed-out attempt. It is not inferred from the estimate.
        The caller must settle the exact action returned by :meth:`next_action`.
        """

        if self._in_flight is None:
            raise ValueError("no in-flight action to settle")
        if action != self._in_flight:
            raise ValueError("action is stale or belongs to another scheduler")
        if status not in OUTCOME_STATUSES:
            raise ValueError(f"unsupported scheduler outcome: {status}")
        _nonnegative_int(charged_cost_units, "charged_cost_units")
        candidate = action.candidate
        if channel_key is None:
            channel_key = candidate.channel_key
        elif candidate.channel_key is not None and channel_key != candidate.channel_key:
            raise ValueError("settled channel_key conflicts with the frozen candidate")
        if status == "duplicate" and charged_cost_units != 0:
            raise ValueError("duplicate outcomes must have zero charged cost")
        if status == "success" and channel_key in self._accepted_channels:
            raise ValueError("successful outcome would duplicate an accepted channel")
        record = SchedulerRecord(
            sequence=action.sequence,
            candidate_id=candidate.candidate_id,
            action_kind=candidate.action_kind,
            status=status,
            estimated_cost_units=candidate.estimated_cost_units,
            charged_cost_units=charged_cost_units,
            channel_key=channel_key,
            reason=reason,
        )
        self._append_record(record)
        self._in_flight = None
        return record

    def stop(self, reason: str) -> None:
        """Pause scheduling while retaining pending and in-flight work."""

        reason = _text(reason, "reason")
        if self._lifecycle in {"completed", "budget_exhausted"}:
            return
        self._lifecycle = "stopped"
        self._stop_reason = reason

    def resume(self) -> str:
        """Resume a stopped schedule and return its lifecycle state."""

        if self._lifecycle == "stopped":
            if not self._pending:
                self._lifecycle = "completed"
            elif self.remaining_units <= 0:
                self._lifecycle = "budget_exhausted"
            else:
                self._lifecycle = "running"
                self._stop_reason = None
        return self._lifecycle

    def summary(self) -> dict[str, Any]:
        failures = [record for record in self._records if record.status in FAILURE_STATUSES]
        return {
            "schema": self.schema,
            "strategy": self.strategy,
            "budget_units": self.budget_units,
            "spent_units": self.spent_units,
            "remaining_units": self.remaining_units,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "records": len(self._records),
            "successful_channels": len(self._accepted_channels),
            "failure_records": len(failures),
            "failure_cost_units": sum(record.charged_cost_units for record in failures),
            "duplicate_records": sum(record.status == "duplicate" for record in self._records),
            "pending_candidates": [candidate.candidate_id for candidate in self.pending_candidates],
            "in_flight_candidate": None if self._in_flight is None else self._in_flight.candidate.candidate_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "strategy": self.strategy,
            "budget_units": self.budget_units,
            "candidates": [asdict(candidate) for candidate in self._candidates],
            "records": [asdict(record) for record in self._records],
            "status": self.status,
            "stop_reason": self.stop_reason,
            "in_flight": None
            if self._in_flight is None
            else {"sequence": self._in_flight.sequence, "candidate_id": self._in_flight.candidate.candidate_id},
            "next_sequence": self._next_sequence,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BudgetScheduler":
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported scheduler schema")
        candidates = tuple(BudgetCandidate(**row) for row in payload.get("candidates", ()))
        scheduler = cls(candidates, payload["budget_units"], strategy=payload["strategy"])
        records = tuple(SchedulerRecord(**row) for row in payload.get("records", ()))
        expected_sequence = 1
        for record in records:
            if record.sequence != expected_sequence:
                raise ValueError("scheduler record sequences must be contiguous")
            scheduler._append_record(record)
            expected_sequence += 1
        scheduler._next_sequence = _positive_int(payload.get("next_sequence", expected_sequence), "next_sequence")
        if scheduler._next_sequence < expected_sequence:
            raise ValueError("next_sequence precedes settled records")
        inflight = payload.get("in_flight")
        if inflight is not None:
            if not isinstance(inflight, Mapping):
                raise ValueError("in_flight must be null or a mapping")
            candidate_id = _text(inflight.get("candidate_id"), "in_flight.candidate_id")
            sequence = _positive_int(inflight.get("sequence"), "in_flight.sequence")
            candidate = scheduler._pending.get(candidate_id)
            if candidate is None:
                raise ValueError("in-flight candidate is already settled or unknown")
            if sequence != scheduler._next_sequence:
                raise ValueError("in-flight sequence does not match next_sequence")
            scheduler._in_flight = SchedulerAction(sequence, candidate)
        status = payload.get("status", "running")
        if status not in LIFECYCLE_STATES:
            raise ValueError(f"unsupported scheduler lifecycle: {status}")
        scheduler._lifecycle = status
        stop_reason = payload.get("stop_reason")
        scheduler._stop_reason = None if stop_reason is None else _text(stop_reason, "stop_reason")
        if status == "completed" and scheduler._pending:
            raise ValueError("completed scheduler still has pending candidates")
        if status == "budget_exhausted" and scheduler._in_flight is not None:
            raise ValueError("budget-exhausted scheduler cannot have an in-flight action")
        return scheduler

    def save(self, path: Path) -> None:
        """Persist a restartable snapshot with an atomic replacement."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path: Path) -> "BudgetScheduler":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = [
    "ACTION_KINDS",
    "BudgetCandidate",
    "BudgetScheduler",
    "FAILURE_STATUSES",
    "LIFECYCLE_STATES",
    "OUTCOME_STATUSES",
    "SCHEDULER_SCHEMA",
    "STRATEGIES",
    "SchedulerAction",
    "SchedulerBudgetExceeded",
    "SchedulerRecord",
]
