"""Prepare, run, resume and summarize bounded execution plans."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .ledger import BudgetExceeded, LedgerEvent, RunLedger, StageBudget, utc_now
from .probe import probe_environment


@dataclass(frozen=True)
class RunSpec:
    """Frozen identity and budget inputs for one resumable run."""

    run_id: str
    phase: str
    task: str
    calculator: str
    protocol_id: str
    calculator_build_hash: str
    code_commit: str
    environment_digest: str
    ledger_id: str
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for name in ("run_id", "phase", "task", "calculator", "protocol_id", "calculator_build_hash", "code_commit", "environment_digest", "ledger_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class SystemInput:
    """Stable identity for one system; content is supplied to the executor."""

    system_id: str
    input_hash: str
    payload: Any = None

    def __post_init__(self) -> None:
        if not self.system_id.strip() or not self.input_hash.strip():
            raise ValueError("system_id and input_hash are required")


@dataclass(frozen=True)
class ExecutionResult:
    """Metered result returned by a calculator adapter."""

    operation: str = "energy_forces"
    status: str = "success"
    cache_hit: bool = False
    calculator_calls: int = 1
    cpu_seconds: float = 0.0
    gpu_seconds: float = 0.0
    cpu_cores: int = 1
    storage_bytes: int = 0
    error: str | None = None


class RunManager:
    """A small deterministic runner whose ledger is the source of truth."""

    def __init__(self, ledger: RunLedger, run_root: Path) -> None:
        self.ledger = ledger
        self.run_root = Path(run_root)
        self.run_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def input_hash(payload: Any) -> str:
        """Hash JSON-compatible input payloads canonically for idempotency."""

        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def prepare(self, spec: RunSpec, *, environment: Mapping[str, Any] | None = None) -> Path:
        """Freeze a run manifest before any calculator call is made."""

        if spec.ledger_id != self.ledger.ledger_id or spec.phase != self.ledger.budget.phase:
            raise ValueError("run spec does not match the shared ledger")
        path = self.run_root / f"{spec.run_id}.json"
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            if old.get("spec") != asdict(spec):
                raise ValueError("run id already exists with a different frozen spec")
            return path
        payload = {
            "schema": "xtbflow-run-manifest/v1",
            "status": "prepared",
            "spec": asdict(spec),
            "budget": asdict(self.ledger.budget),
            "environment": dict(environment or probe_environment()),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def dry_run(self, spec: RunSpec, systems: Iterable[SystemInput]) -> dict[str, Any]:
        """Validate a plan without mutating the ledger or running a calculator."""

        systems = list(systems)
        if len({s.system_id for s in systems}) != len(systems):
            raise ValueError("duplicate system_id in plan")
        completed = self.ledger.completed_systems(spec.run_id)
        pending = [s for s in systems if (s.system_id, s.input_hash) not in completed]
        planned = self.ledger._totals()
        planned["system_evaluations"] += len(pending)
        if self.ledger.budget.max_system_evaluations is not None and planned["system_evaluations"] > self.ledger.budget.max_system_evaluations:
            raise BudgetExceeded("dry-run plan exceeds cumulative system evaluation budget")
        return {"run_id": spec.run_id, "ledger_id": self.ledger.ledger_id, "total_systems": len(systems), "pending_systems": len(pending), "already_completed": len(systems) - len(pending), "projected": planned}

    def run(self, spec: RunSpec, systems: Iterable[SystemInput], executor: Callable[..., ExecutionResult], *, resume: bool = True) -> dict[str, Any]:
        """Execute one system at a time, recording every attempt before resume."""

        self.prepare(spec)
        systems = list(systems)
        plan = self.dry_run(spec, systems)
        completed = self.ledger.completed_systems(spec.run_id) if resume else set()
        self.ledger.acquire_job(spec.run_id)
        self._set_manifest_status(spec, "running")
        try:
            for system in systems:
                if resume and (system.system_id, system.input_hash) in completed:
                    continue
                attempts = [e for e in self.ledger.events if e.run_id == spec.run_id and e.system_id == system.system_id]
                attempt = len(attempts) + 1
                try:
                    result = executor(system) if len(inspect.signature(executor).parameters) else executor()
                    if not isinstance(result, ExecutionResult):
                        raise TypeError("executor must return ExecutionResult")
                except TimeoutError as exc:
                    result = ExecutionResult(status="timeout", error=str(exc))
                except Exception as exc:  # Record failure before propagating no state loss.
                    result = ExecutionResult(status="failure", error=f"{type(exc).__name__}: {exc}")
                event = LedgerEvent(
                    run_id=spec.run_id,
                    phase=spec.phase,
                    calculator=spec.calculator,
                    protocol_id=spec.protocol_id,
                    calculator_build_hash=spec.calculator_build_hash,
                    system_id=system.system_id,
                    input_hash=system.input_hash,
                    operation=result.operation,
                    status=result.status,
                    attempt=attempt,
                    cache_hit=result.cache_hit,
                    calculator_calls=result.calculator_calls,
                    cpu_seconds=result.cpu_seconds,
                    gpu_seconds=result.gpu_seconds,
                    cpu_cores=result.cpu_cores,
                    storage_bytes=result.storage_bytes,
                    error=result.error,
                )
                self.ledger.append(event, persist_path=self.run_root / f"{spec.ledger_id}.json")
                completed.add((system.system_id, system.input_hash))
        except Exception:
            self._set_manifest_status(spec, "failed")
            raise
        else:
            self._set_manifest_status(spec, "completed")
        finally:
            self.ledger.release_job(spec.run_id)
        return self.ledger.summary() | {"plan": plan}

    def _set_manifest_status(self, spec: RunSpec, status: str) -> None:
        """Update only the lifecycle field while retaining the frozen identity."""

        path = self.run_root / f"{spec.run_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["status"] = status
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def summary(self) -> dict[str, Any]:
        """Return read-only cumulative totals for reports and dashboards."""

        return self.ledger.summary()
