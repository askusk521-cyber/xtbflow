"""Persistent, phase-scoped accounting for bounded xtbflow runs.

The legacy ``InferenceCostLedger`` records model-control counters.  This module
adds the outer execution ledger: one event per system attempt, explicit
calculator/protocol identity, resource units, cache and retry status, and a
hard stage budget.  It deliberately uses only the standard library so the
accounting layer can run before optional quantum-chemistry backends are loaded.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import os
import tempfile
from threading import RLock
from typing import Any, Iterable, Mapping
from uuid import uuid4


STATUSES = frozenset({"success", "failure", "timeout", "cancelled"})
OPERATIONS = frozenset({"energy", "forces", "energy_forces", "hessian", "hvp", "train", "other"})


def utc_now() -> str:
    """Return an RFC3339 UTC timestamp suitable for a run record."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _nonnegative_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a nonnegative finite number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a nonnegative finite number")
    return value


def _nonnegative_int(name: str, value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class StageBudget:
    """Hard cumulative limits shared by all runs in one stage ledger."""

    phase: str
    max_gpu_hours: float | None = None
    max_cpu_core_hours: float | None = None
    max_system_evaluations: int | None = None
    max_calculator_calls: int | None = None
    max_storage_gb: float | None = None
    max_concurrent_jobs: int = 1
    max_retries_per_job: int = 1
    max_wall_minutes_per_job: float = 120.0

    def __post_init__(self) -> None:
        if not isinstance(self.phase, str) or not self.phase.strip():
            raise ValueError("phase must be a non-empty string")
        for name in ("max_gpu_hours", "max_cpu_core_hours", "max_storage_gb"):
            value = getattr(self, name)
            if value is not None and (_nonnegative_number(name, value) < 0):
                raise ValueError(f"{name} must be None or nonnegative")
        for name in ("max_system_evaluations", "max_calculator_calls"):
            value = getattr(self, name)
            if value is not None and _nonnegative_int(name, value) < 1:
                raise ValueError(f"{name} must be None or positive")
        if type(self.max_concurrent_jobs) is not int or self.max_concurrent_jobs < 1:
            raise ValueError("max_concurrent_jobs must be a positive integer")
        if type(self.max_retries_per_job) is not int or self.max_retries_per_job < 0:
            raise ValueError("max_retries_per_job must be a nonnegative integer")
        if _nonnegative_number("max_wall_minutes_per_job", self.max_wall_minutes_per_job) <= 0:
            raise ValueError("max_wall_minutes_per_job must be positive")


@dataclass(frozen=True)
class LedgerEvent:
    """One metered system attempt, including failures and cache hits."""

    run_id: str
    phase: str
    calculator: str
    protocol_id: str
    calculator_build_hash: str
    system_id: str
    input_hash: str
    operation: str
    status: str
    attempt: int = 1
    cache_hit: bool = False
    system_evaluations: int = 1
    calculator_calls: int = 1
    cpu_seconds: float = 0.0
    gpu_seconds: float = 0.0
    cpu_cores: int = 1
    storage_bytes: int = 0
    error: str | None = None
    timestamp: str = field(default_factory=utc_now)
    event_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        for name in ("run_id", "phase", "calculator", "protocol_id", "calculator_build_hash", "system_id", "input_hash", "operation", "status"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.operation not in OPERATIONS:
            raise ValueError(f"unsupported operation: {self.operation}")
        if self.status not in STATUSES:
            raise ValueError(f"unsupported status: {self.status}")
        if type(self.attempt) is not int or self.attempt < 1:
            raise ValueError("attempt must be a positive integer")
        for name in ("system_evaluations", "calculator_calls", "storage_bytes"):
            _nonnegative_int(name, getattr(self, name))
        if type(self.cpu_cores) is not int or self.cpu_cores < 1:
            raise ValueError("cpu_cores must be a positive integer")
        _nonnegative_number("cpu_seconds", self.cpu_seconds)
        _nonnegative_number("gpu_seconds", self.gpu_seconds)
        if self.status == "success" and self.error is not None:
            raise ValueError("successful events cannot carry an error")
        if self.cache_hit and self.calculator_calls != 0:
            raise ValueError("cache hits must record zero calculator calls")

    @property
    def cpu_core_seconds(self) -> float:
        return float(self.cpu_seconds) * self.cpu_cores


class BudgetExceeded(RuntimeError):
    """Raised before an event or reservation would exceed a hard stage cap."""


class ConcurrentJobLimit(RuntimeError):
    """Raised when a runner would exceed the shared concurrent-job limit."""


class RunLedger:
    """Thread-safe cumulative ledger with atomic JSON persistence.

    A ledger is shared by every run in a stage.  ``append`` validates the
    projected totals before mutating state, so failed or retried attempts cannot
    silently bypass the cap.  ``save`` uses a same-directory temporary file and
    ``os.replace``; a reader therefore sees either the old complete record or
    the new complete record after a restart.
    """

    schema = "xtbflow-run-ledger/v1"

    def __init__(self, budget: StageBudget, *, ledger_id: str | None = None,
                 events: Iterable[LedgerEvent] = (), reservations: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        self.budget = budget
        self.ledger_id = ledger_id or uuid4().hex
        self.events = list(events)
        self.reservations: dict[str, dict[str, Any]] = dict(reservations or {})
        self._active_jobs: set[str] = set()
        self._lock = RLock()
        self._validate_events()

    def _validate_events(self) -> None:
        for event in self.events:
            if not isinstance(event, LedgerEvent):
                raise TypeError("events must contain LedgerEvent values")
            if event.phase != self.budget.phase:
                raise ValueError("event phase does not match ledger budget")
        self.check_budget()

    def _totals(self, events: Iterable[LedgerEvent] | None = None) -> dict[str, float | int]:
        rows = self.events if events is None else list(events)
        return {
            "gpu_hours": sum(e.gpu_seconds for e in rows) / 3600.0,
            "cpu_core_hours": sum(e.cpu_core_seconds for e in rows) / 3600.0,
            "system_evaluations": sum(e.system_evaluations for e in rows),
            "calculator_calls": sum(e.calculator_calls for e in rows),
            "storage_gb": sum(e.storage_bytes for e in rows) / (1024**3),
            "attempts": len(rows),
            "failures": sum(e.status in {"failure", "timeout", "cancelled"} for e in rows),
            "successes": sum(e.status == "success" for e in rows),
            "retries": sum(e.attempt > 1 for e in rows),
            "cache_hits": sum(e.cache_hit for e in rows),
        }

    def check_budget(self, events: Iterable[LedgerEvent] | None = None) -> None:
        totals = self._totals(events)
        checks = (
            ("gpu_hours", self.budget.max_gpu_hours),
            ("cpu_core_hours", self.budget.max_cpu_core_hours),
            ("system_evaluations", self.budget.max_system_evaluations),
            ("calculator_calls", self.budget.max_calculator_calls),
            ("storage_gb", self.budget.max_storage_gb),
        )
        for name, limit in checks:
            if limit is not None and totals[name] > limit + 1e-12:
                raise BudgetExceeded(f"{name}={totals[name]} exceeds budget {limit}")

    def projected(self, event: LedgerEvent) -> dict[str, float | int]:
        with self._lock:
            self.check_budget([*self.events, event])
            return self._totals([*self.events, event])

    def append(self, event: LedgerEvent, *, persist_path: Path | None = None) -> None:
        with self._lock:
            if event.phase != self.budget.phase:
                raise ValueError("event phase does not match ledger budget")
            prior_attempts = [e for e in self.events if e.run_id == event.run_id and e.system_id == event.system_id]
            if event.attempt != len(prior_attempts) + 1:
                raise ValueError("attempt number must follow the previous system attempts")
            if event.attempt - 1 > self.budget.max_retries_per_job:
                raise BudgetExceeded(f"retry count for {event.run_id}/{event.system_id} exceeds budget")
            self.check_budget([*self.events, event])
            self.events.append(event)
            if persist_path is not None:
                self.save(persist_path)

    def acquire_job(self, run_id: str) -> None:
        with self._lock:
            if run_id in self._active_jobs:
                raise ConcurrentJobLimit(f"run {run_id} is already active")
            if len(self._active_jobs) >= self.budget.max_concurrent_jobs:
                raise ConcurrentJobLimit("stage max_concurrent_jobs reached")
            self._active_jobs.add(run_id)

    def release_job(self, run_id: str) -> None:
        with self._lock:
            self._active_jobs.discard(run_id)

    def completed_systems(self, run_id: str) -> set[tuple[str, str]]:
        """Return ``(system_id, input_hash)`` pairs with a successful attempt."""
        with self._lock:
            return {(e.system_id, e.input_hash) for e in self.events if e.run_id == run_id and e.status == "success"}

    def summary(self) -> dict[str, Any]:
        with self._lock:
            totals = self._totals()
            by_status: dict[str, int] = {}
            by_operation: dict[str, int] = {}
            by_calculator: dict[str, int] = {}
            for event in self.events:
                by_status[event.status] = by_status.get(event.status, 0) + 1
                by_operation[event.operation] = by_operation.get(event.operation, 0) + 1
                by_calculator[event.calculator] = by_calculator.get(event.calculator, 0) + 1
            return {
                "schema": self.schema,
                "ledger_id": self.ledger_id,
                "phase": self.budget.phase,
                "budget": asdict(self.budget),
                "totals": totals,
                "by_status": by_status,
                "by_operation": by_operation,
                "by_calculator": by_calculator,
                "active_jobs": sorted(self._active_jobs),
            }

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema": self.schema,
                "ledger_id": self.ledger_id,
                "budget": asdict(self.budget),
                "events": [asdict(e) for e in self.events],
                "reservations": self.reservations,
            }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunLedger":
        if payload.get("schema") != cls.schema:
            raise ValueError("unsupported run ledger schema")
        budget = StageBudget(**payload["budget"])
        events = [LedgerEvent(**row) for row in payload.get("events", [])]
        return cls(budget, ledger_id=payload["ledger_id"], events=events, reservations=payload.get("reservations"))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"
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
    def load(cls, path: Path) -> "RunLedger":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
