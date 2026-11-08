"""Provider boundary, strict reservations, and deterministic mock adapters.

Real physical and policy providers must pass through this module before their
outputs reach the #166 merge gate.  The boundary validates identity and
target-free inputs, reserves an estimate before invocation, reconciles the
measured cost afterwards, and returns only immutable :class:`GuidanceOutput`
records.  No provider receives a mutable state object.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import re
from threading import RLock
from typing import Any, Mapping, Protocol

from .parallel_guidance import (
    GenerationSnapshot,
    GuidanceCost,
    GuidanceLane,
    GuidanceOutput,
    GuidanceStatus,
    reject_target_derived_fields,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def validate_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256")
    return value


def fingerprint_text(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("fingerprint input must be text")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    result = dict(value)
    try:
        json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON-compatible and finite") from exc
    reject_target_derived_fields(result, name=name)
    return result


class AdapterStatus(str, Enum):
    ACCEPTED = "accepted"
    INVALID = "invalid"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TIMEOUT = "timeout"
    FAILED = "failed"
    DUPLICATE = "duplicate"


class BudgetExceeded(RuntimeError):
    """Raised before an estimate or after reconciliation would exceed a cap."""


@dataclass(frozen=True)
class ProviderRequest:
    """Immutable provider input; no provider receives mutable generation state."""

    snapshot: GenerationSnapshot
    lane: GuidanceLane | str
    provider_revision: str
    config_fingerprint: str
    protocol_fingerprint: str
    feedback: Mapping[str, Any] = field(default_factory=dict)
    action_space: Mapping[str, Any] = field(default_factory=dict)
    request_fingerprint: str = ""

    def __post_init__(self) -> None:
        lane = self.lane if isinstance(self.lane, GuidanceLane) else GuidanceLane(self.lane)
        validate_sha256(self.provider_revision, "provider_revision")
        validate_sha256(self.config_fingerprint, "config_fingerprint")
        validate_sha256(self.protocol_fingerprint, "protocol_fingerprint")
        feedback = _json_mapping(self.feedback, "feedback")
        action_space = _json_mapping(self.action_space, "action_space")
        object.__setattr__(self, "lane", lane)
        object.__setattr__(self, "feedback", feedback)
        object.__setattr__(self, "action_space", action_space)
        payload = {
            "snapshot_id": self.snapshot.snapshot_id,
            "lane": lane.value,
            "provider_revision": self.provider_revision,
            "config_fingerprint": self.config_fingerprint,
            "protocol_fingerprint": self.protocol_fingerprint,
            "feedback": feedback,
            "action_space": action_space,
        }
        computed = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        if self.request_fingerprint and self.request_fingerprint != computed:
            raise ValueError("request_fingerprint does not match provider request")
        object.__setattr__(self, "request_fingerprint", computed)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot.snapshot_id,
            "lane": self.lane.value,
            "provider_revision": self.provider_revision,
            "config_fingerprint": self.config_fingerprint,
            "protocol_fingerprint": self.protocol_fingerprint,
            "feedback": self.feedback,
            "action_space": self.action_space,
            "request_fingerprint": self.request_fingerprint,
        }


class GuidanceProvider(Protocol):
    lane: GuidanceLane
    provider_revision: str
    config_fingerprint: str
    protocol_fingerprint: str

    def estimate(self, request: ProviderRequest) -> GuidanceCost:
        ...

    def invoke(self, request: ProviderRequest) -> GuidanceOutput:
        ...


PhysicalGuidanceProvider = GuidanceProvider
PolicyGuidanceProvider = GuidanceProvider


@dataclass(frozen=True)
class BudgetReservation:
    _budget: "GuidanceBudget" = field(repr=False, compare=False)
    reserved: GuidanceCost
    _closed: bool = field(default=False, repr=False, compare=False)

    def reconcile(self, measured: GuidanceCost) -> GuidanceCost:
        return self._budget._reconcile(self, measured)

    def release(self) -> GuidanceCost:
        return self._budget._reconcile(self, GuidanceCost.zero())


class GuidanceBudget:
    """Thread-safe strict budget with estimate reservation and reconciliation."""

    def __init__(self, limit: GuidanceCost) -> None:
        self.limit = limit
        self._reserved = GuidanceCost.zero()
        self._charged = GuidanceCost.zero()
        self._lock = RLock()

    @property
    def reserved(self) -> GuidanceCost:
        with self._lock:
            return self._reserved

    @property
    def charged(self) -> GuidanceCost:
        with self._lock:
            return self._charged

    @property
    def remaining(self) -> GuidanceCost:
        with self._lock:
            return GuidanceCost(
                model_forwards=max(0, self.limit.model_forwards - self._reserved.model_forwards - self._charged.model_forwards),
                calculator_calls=max(0, self.limit.calculator_calls - self._reserved.calculator_calls - self._charged.calculator_calls),
                wall_seconds=max(0.0, self.limit.wall_seconds - self._reserved.wall_seconds - self._charged.wall_seconds),
                cpu_seconds=max(0.0, self.limit.cpu_seconds - self._reserved.cpu_seconds - self._charged.cpu_seconds),
                gpu_seconds=max(0.0, self.limit.gpu_seconds - self._reserved.gpu_seconds - self._charged.gpu_seconds),
            )

    def reserve(self, estimate: GuidanceCost) -> BudgetReservation:
        with self._lock:
            candidate = self._add(self._add(self._reserved, self._charged), estimate)
            if not self._within(candidate, self.limit):
                raise BudgetExceeded("estimated provider cost exceeds remaining budget")
            self._reserved = self._add(self._reserved, estimate)
            return BudgetReservation(self, estimate)

    def _reconcile(self, reservation: BudgetReservation, measured: GuidanceCost) -> GuidanceCost:
        with self._lock:
            if reservation._closed:
                raise ValueError("budget reservation is already reconciled")
            if not isinstance(measured, GuidanceCost):
                raise ValueError("measured cost must be GuidanceCost")
            remaining_reserved = self._sub(self._reserved, reservation.reserved)
            candidate = self._add(self._add(remaining_reserved, self._charged), measured)
            if not self._within(candidate, self.limit):
                raise BudgetExceeded("measured provider cost exceeds hard budget")
            self._reserved = remaining_reserved
            self._charged = self._add(self._charged, measured)
            object.__setattr__(reservation, "_closed", True)
            return measured

    @staticmethod
    def _add(left: GuidanceCost, right: GuidanceCost) -> GuidanceCost:
        return GuidanceCost(
            model_forwards=left.model_forwards + right.model_forwards,
            calculator_calls=left.calculator_calls + right.calculator_calls,
            wall_seconds=left.wall_seconds + right.wall_seconds,
            cpu_seconds=left.cpu_seconds + right.cpu_seconds,
            gpu_seconds=left.gpu_seconds + right.gpu_seconds,
        )

    @staticmethod
    def _sub(left: GuidanceCost, right: GuidanceCost) -> GuidanceCost:
        return GuidanceCost(
            model_forwards=left.model_forwards - right.model_forwards,
            calculator_calls=left.calculator_calls - right.calculator_calls,
            wall_seconds=left.wall_seconds - right.wall_seconds,
            cpu_seconds=left.cpu_seconds - right.cpu_seconds,
            gpu_seconds=left.gpu_seconds - right.gpu_seconds,
        )

    @staticmethod
    def _within(value: GuidanceCost, limit: GuidanceCost) -> bool:
        return (
            value.model_forwards <= limit.model_forwards
            and value.calculator_calls <= limit.calculator_calls
            and value.wall_seconds <= limit.wall_seconds
            and value.cpu_seconds <= limit.cpu_seconds
            and value.gpu_seconds <= limit.gpu_seconds
        )


@dataclass(frozen=True)
class AdapterLedgerEntry:
    request_fingerprint: str
    lane: GuidanceLane
    provider_revision: str
    config_fingerprint: str
    protocol_fingerprint: str
    status: AdapterStatus
    failure_stage: str | None
    error: str | None
    estimated_cost: GuidanceCost
    measured_cost: GuidanceCost
    charged_cost: GuidanceCost

    def to_mapping(self) -> dict[str, Any]:
        return {
            "request_fingerprint": self.request_fingerprint,
            "lane": self.lane.value,
            "provider_revision": self.provider_revision,
            "config_fingerprint": self.config_fingerprint,
            "protocol_fingerprint": self.protocol_fingerprint,
            "status": self.status.value,
            "failure_stage": self.failure_stage,
            "error": self.error,
            "estimated_cost": self.estimated_cost.to_mapping(),
            "measured_cost": self.measured_cost.to_mapping(),
            "charged_cost": self.charged_cost.to_mapping(),
        }


@dataclass(frozen=True)
class AdapterLedger:
    entries: tuple[AdapterLedgerEntry, ...] = ()

    def append(self, entry: AdapterLedgerEntry) -> "AdapterLedger":
        return AdapterLedger(self.entries + (entry,))

    def to_mapping(self) -> dict[str, Any]:
        return {"schema": "xtbflow-adapter-ledger/v1", "entries": [entry.to_mapping() for entry in self.entries]}


@dataclass(frozen=True)
class AdapterRun:
    status: AdapterStatus
    output: GuidanceOutput | None
    request_fingerprint: str
    failure_stage: str | None
    error: str | None
    estimated_cost: GuidanceCost
    measured_cost: GuidanceCost
    charged_cost: GuidanceCost
    ledger_entry: AdapterLedgerEntry


class AdapterRunner:
    """Execute a provider only after a successful budget reservation."""

    def __init__(self, budget: GuidanceBudget) -> None:
        self.budget = budget
        self.ledger = AdapterLedger()
        self._seen: set[str] = set()

    def run(self, request: ProviderRequest, provider: GuidanceProvider) -> AdapterRun:
        self._validate_provider_identity(request, provider)
        if request.request_fingerprint in self._seen:
            return self._finish(request, provider, AdapterStatus.DUPLICATE, None, "duplicate_action", None, GuidanceCost.zero(), GuidanceCost.zero(), GuidanceCost.zero())
        self._seen.add(request.request_fingerprint)
        try:
            estimate = provider.estimate(request)
            if not isinstance(estimate, GuidanceCost):
                raise ValueError("provider estimate must be GuidanceCost")
            reservation = self.budget.reserve(estimate)
        except BudgetExceeded as exc:
            return self._finish(request, provider, AdapterStatus.BUDGET_EXHAUSTED, None, "budget_reservation", str(exc), GuidanceCost.zero(), GuidanceCost.zero(), GuidanceCost.zero())
        except Exception as exc:
            return self._finish(request, provider, AdapterStatus.INVALID, None, "provider_estimate", f"{type(exc).__name__}: {exc}", GuidanceCost.zero(), GuidanceCost.zero(), GuidanceCost.zero())

        try:
            output = provider.invoke(request)
            if not isinstance(output, GuidanceOutput):
                raise ValueError("provider must return GuidanceOutput")
            if output.lane is not request.lane:
                raise ValueError("provider output lane does not match request lane")
            if output.input_fingerprint != request.snapshot.snapshot_id:
                raise ValueError("provider output input fingerprint does not match snapshot")
            measured = output.measured_cost
            charged = reservation.reconcile(measured)
            provenance = {
                "adapter": {
                    "request_fingerprint": request.request_fingerprint,
                    "provider_revision": request.provider_revision,
                    "config_fingerprint": request.config_fingerprint,
                    "protocol_fingerprint": request.protocol_fingerprint,
                },
                "provider": output.to_mapping()["provenance"],
            }
            output = replace(output, estimated_cost=estimate, measured_cost=charged, provenance=provenance)
            return self._finish(request, provider, AdapterStatus.ACCEPTED, output, None, None, estimate, measured, charged)
        except TimeoutError as exc:
            measured = GuidanceCost.zero()
            charged = reservation.reconcile(measured)
            return self._finish(request, provider, AdapterStatus.TIMEOUT, None, "provider_timeout", str(exc), estimate, measured, charged)
        except BudgetExceeded as exc:
            charged = reservation.reconcile(reservation.reserved)
            return self._finish(request, provider, AdapterStatus.BUDGET_EXHAUSTED, None, "budget_reconcile", str(exc), estimate, reservation.reserved, charged)
        except ValueError as exc:
            measured = GuidanceCost.zero()
            charged = reservation.reconcile(measured)
            return self._finish(request, provider, AdapterStatus.INVALID, None, "provider_output", str(exc), estimate, measured, charged)
        except Exception as exc:
            measured = GuidanceCost.zero()
            charged = reservation.reconcile(measured)
            return self._finish(request, provider, AdapterStatus.FAILED, None, "provider_execution", f"{type(exc).__name__}: {exc}", estimate, measured, charged)

    @staticmethod
    def _validate_provider_identity(request: ProviderRequest, provider: GuidanceProvider) -> None:
        if provider.lane is not request.lane:
            raise ValueError("provider lane does not match request lane")
        validate_sha256(provider.provider_revision, "provider.provider_revision")
        validate_sha256(provider.config_fingerprint, "provider.config_fingerprint")
        validate_sha256(provider.protocol_fingerprint, "provider.protocol_fingerprint")
        if provider.provider_revision != request.provider_revision:
            raise ValueError("provider revision does not match request")
        if provider.config_fingerprint != request.config_fingerprint:
            raise ValueError("provider config does not match request")
        if provider.protocol_fingerprint != request.protocol_fingerprint:
            raise ValueError("provider protocol does not match request")

    def _finish(
        self,
        request: ProviderRequest,
        provider: GuidanceProvider,
        status: AdapterStatus,
        output: GuidanceOutput | None,
        failure_stage: str | None,
        error: str | None,
        estimated: GuidanceCost,
        measured: GuidanceCost,
        charged: GuidanceCost,
    ) -> AdapterRun:
        entry = AdapterLedgerEntry(
            request_fingerprint=request.request_fingerprint,
            lane=request.lane,
            provider_revision=provider.provider_revision,
            config_fingerprint=provider.config_fingerprint,
            protocol_fingerprint=provider.protocol_fingerprint,
            status=status,
            failure_stage=failure_stage,
            error=error,
            estimated_cost=estimated,
            measured_cost=measured,
            charged_cost=charged,
        )
        self.ledger = self.ledger.append(entry)
        return AdapterRun(status, output, request.request_fingerprint, failure_stage, error, estimated, measured, charged, entry)


class MockPhysicalProvider:
    lane = GuidanceLane.GEOMETRY
    provider_revision = fingerprint_text("mock-physical-v1")
    config_fingerprint = fingerprint_text("mock-physical-config-v1")
    protocol_fingerprint = fingerprint_text("mock-physical-protocol-v1")

    def __init__(self, *, delta: tuple[tuple[float, float, float], ...] = ((0.03, 0.0, 0.0), (0.0, 0.0, 0.0))) -> None:
        self.delta = delta
        self.invocations = 0
        self.reserved_at_invoke: list[GuidanceCost] = []

    def estimate(self, request: ProviderRequest) -> GuidanceCost:
        return GuidanceCost(model_forwards=1, calculator_calls=1, wall_seconds=0.05)

    def invoke(self, request: ProviderRequest) -> GuidanceOutput:
        self.invocations += 1
        return GuidanceOutput(
            lane=self.lane,
            provider="mock-physical-v1",
            input_fingerprint=request.snapshot.snapshot_id,
            geometry_delta=self.delta,
            trust_radius=0.1,
            estimated_cost=self.estimate(request),
            measured_cost=GuidanceCost(model_forwards=1, calculator_calls=1, wall_seconds=0.04),
            provenance={"mock": "physical"},
        )


class MockPolicyProvider:
    lane = GuidanceLane.EVENT
    provider_revision = fingerprint_text("mock-policy-v1")
    config_fingerprint = fingerprint_text("mock-policy-config-v1")
    protocol_fingerprint = fingerprint_text("mock-policy-protocol-v1")

    def __init__(self, *, delta: tuple[float, ...] = (0.04, -0.02, 0.0, 0.01)) -> None:
        self.delta = delta
        self.invocations = 0

    def estimate(self, request: ProviderRequest) -> GuidanceCost:
        return GuidanceCost(model_forwards=1, wall_seconds=0.02)

    def invoke(self, request: ProviderRequest) -> GuidanceOutput:
        self.invocations += 1
        return GuidanceOutput(
            lane=self.lane,
            provider="mock-policy-v1",
            input_fingerprint=request.snapshot.snapshot_id,
            event_delta=self.delta,
            trust_radius=0.2,
            estimated_cost=self.estimate(request),
            measured_cost=GuidanceCost(model_forwards=1, wall_seconds=0.01),
            provenance={"mock": "policy"},
        )


__all__ = [
    "AdapterStatus", "BudgetExceeded", "validate_sha256", "fingerprint_text", "ProviderRequest",
    "GuidanceProvider", "PhysicalGuidanceProvider", "PolicyGuidanceProvider", "GuidanceBudget",
    "BudgetReservation", "AdapterLedgerEntry", "AdapterLedger", "AdapterRun", "AdapterRunner",
    "MockPhysicalProvider", "MockPolicyProvider",
]
