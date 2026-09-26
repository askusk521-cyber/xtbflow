"""Reference transition-state validation records and resumable orchestration."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

from .connectivity import ConnectivityEvidence, observed_event
from .modes import ModeEvidence, validate_mode


@dataclass(frozen=True)
class TSValidationConfig:
    """Frozen success criteria, selected before comparing candidate sources."""

    gradient_tolerance: float = 1e-4
    negative_mode_threshold: float = 1e-4
    max_negative_modes: int = 1
    require_connectivity: bool = True
    max_calls: int = 32

    def __post_init__(self) -> None:
        for name in ("gradient_tolerance", "negative_mode_threshold"):
            value = getattr(self, name)
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.max_negative_modes) is not int or self.max_negative_modes < 1 or type(self.max_calls) is not int or self.max_calls < 1:
            raise ValueError("max_negative_modes and max_calls must be positive integers")


@dataclass(frozen=True)
class TSValidationRecord:
    candidate_id: str
    source: str
    status: str
    path_status: str
    gradient_norm: float | None
    mode_status: str
    observed_event: tuple[tuple[int, int, int], ...] | None
    calculator_calls: int
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"success", "failure", "not_converged"}:
            raise ValueError("unsupported TS validation status")
        if self.path_status not in {"not_run", "not_validated", "validated"}:
            raise ValueError("unsupported path status")
        if type(self.calculator_calls) is not int or self.calculator_calls < 0:
            raise ValueError("calculator_calls must be nonnegative")
        if self.gradient_norm is not None and (not math.isfinite(float(self.gradient_norm)) or self.gradient_norm < 0):
            raise ValueError("gradient_norm must be finite and nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_ts_evidence(candidate_id: str, source: str, evidence: Mapping[str, Any], config: TSValidationConfig, *, calls: int = 1) -> TSValidationRecord:
    """Validate an observed stationary point; missing evidence fails closed."""

    if not candidate_id.strip() or not source.strip():
        raise ValueError("candidate_id and source are required")
    try:
        gradient = float(evidence["gradient_norm"])
        converged = bool(evidence["converged"])
        modes = ModeEvidence(tuple(float(value) for value in evidence["hessian_eigenvalues"]), str(evidence.get("mode_unit", "hessian_eigenvalue")))
    except (KeyError, TypeError, ValueError) as exc:
        return TSValidationRecord(candidate_id, source, "failure", "not_validated", None, "missing_evidence", None, calls, str(exc))
    if calls < 0 or calls > config.max_calls:
        return TSValidationRecord(candidate_id, source, "failure", "not_validated", gradient, "call_budget", None, calls, "calculator call budget exceeded")
    if not converged:
        return TSValidationRecord(candidate_id, source, "not_converged", "not_validated", gradient, "not_converged", None, calls, "stationary point did not converge")
    if not math.isfinite(gradient) or gradient > config.gradient_tolerance:
        return TSValidationRecord(candidate_id, source, "failure", "not_validated", gradient, "gradient_tolerance", None, calls, "gradient norm exceeds TS threshold")
    mode_ok, mode_status = validate_mode(modes, negative_threshold=config.negative_mode_threshold, max_negative_modes=config.max_negative_modes)
    if not mode_ok:
        return TSValidationRecord(candidate_id, source, "failure", "not_validated", gradient, mode_status, None, calls, mode_status)
    connectivity = evidence.get("connectivity")
    observed = None
    path_status = "not_run"
    if config.require_connectivity:
        if not isinstance(connectivity, ConnectivityEvidence):
            return TSValidationRecord(candidate_id, source, "failure", "not_validated", gradient, mode_status, None, calls, "connectivity evidence is required")
        observed = observed_event(connectivity)
        path_status = "validated" if bool(evidence.get("path_connected", False)) else "not_validated"
        if path_status != "validated":
            return TSValidationRecord(candidate_id, source, "failure", path_status, gradient, mode_status, observed, calls, "bidirectional endpoint path is not connected")
    return TSValidationRecord(candidate_id, source, "success", path_status, gradient, mode_status, observed, calls)


def run_resumable_validation(candidates: Mapping[str, Any], searcher: Callable[[str, Any], Mapping[str, Any]], config: TSValidationConfig, checkpoint: str | Path) -> tuple[TSValidationRecord, ...]:
    """Run candidates with a JSON checkpoint; the searcher is always injected."""

    target = Path(checkpoint)
    prior: dict[str, dict[str, Any]] = {}
    if target.exists():
        prior = json.loads(target.read_text(encoding="utf-8"))
    output: list[TSValidationRecord] = []
    for candidate_id, candidate in candidates.items():
        if candidate_id in prior:
            output.append(TSValidationRecord(**prior[candidate_id]))
            continue
        evidence = searcher(candidate_id, candidate)
        record = validate_ts_evidence(candidate_id, str(evidence.get("source", "injected")), evidence, config, calls=int(evidence.get("calculator_calls", 1)))
        prior[candidate_id] = record.to_dict()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(prior, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        output.append(record)
    return tuple(output)

