"""Reference transition-state validation records and resumable orchestration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
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
    reference_protocol_id: str = "unqualified"
    reference_build_hash: str = "unqualified"
    verification_config_version: str = "ts-validation-v1"

    def __post_init__(self) -> None:
        for name in ("gradient_tolerance", "negative_mode_threshold"):
            value = getattr(self, name)
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.max_negative_modes) is not int or self.max_negative_modes < 1 or type(self.max_calls) is not int or self.max_calls < 1:
            raise ValueError("max_negative_modes and max_calls must be positive integers")
        if type(self.require_connectivity) is not bool:
            raise ValueError("require_connectivity must be boolean")
        for name in ("reference_protocol_id", "reference_build_hash", "verification_config_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")

    @property
    def identity(self) -> str:
        payload = asdict(self)
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


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
    candidate_hash: str = ""
    validation_identity: str = ""
    cache_key: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"success", "failure", "not_converged"}:
            raise ValueError("unsupported TS validation status")
        if self.path_status not in {"not_run", "not_validated", "validated"}:
            raise ValueError("unsupported path status")
        if type(self.calculator_calls) is not int or self.calculator_calls < 0:
            raise ValueError("calculator_calls must be nonnegative")
        if self.gradient_norm is not None and (not math.isfinite(float(self.gradient_norm)) or self.gradient_norm < 0):
            raise ValueError("gradient_norm must be finite and nonnegative")
        for name in ("candidate_hash", "validation_identity", "cache_key"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"candidate contains non-serializable value {type(value).__name__}")


def candidate_content_hash(candidate: Any) -> str:
    """Hash the full candidate payload, preserving geometry and state identity."""

    return hashlib.sha256(json.dumps(_jsonable(candidate), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validation_cache_key(candidate_id: str, candidate: Any, config: TSValidationConfig) -> tuple[str, str, str]:
    content_hash = candidate_content_hash(candidate)
    identity = config.identity
    key_payload = {"candidate_id": candidate_id, "candidate_hash": content_hash, "validation_identity": identity}
    key = hashlib.sha256(json.dumps(key_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return content_hash, identity, key


def _record(candidate_id: str, source: str, status: str, path_status: str, gradient_norm: float | None, mode_status: str, observed: tuple[tuple[int, int, int], ...] | None, calls: int, error: str | None, *, candidate_hash: str = "", validation_identity: str = "", cache_key: str = "") -> TSValidationRecord:
    return TSValidationRecord(candidate_id, source, status, path_status, gradient_norm, mode_status, observed, calls, error, candidate_hash, validation_identity, cache_key)


def validate_ts_evidence(candidate_id: str, source: str, evidence: Mapping[str, Any], config: TSValidationConfig, *, calls: int = 1, candidate_hash: str = "", validation_identity: str = "", cache_key: str = "") -> TSValidationRecord:
    """Validate an observed stationary point; missing evidence fails closed."""

    if not candidate_id.strip() or not source.strip():
        raise ValueError("candidate_id and source are required")
    try:
        gradient = float(evidence["gradient_norm"])
        converged = evidence["converged"]
        if type(converged) is not bool:
            raise ValueError("converged must be a boolean")
        modes = ModeEvidence(tuple(float(value) for value in evidence["hessian_eigenvalues"]), str(evidence.get("mode_unit", "hessian_eigenvalue")))
    except (KeyError, TypeError, ValueError) as exc:
        return _record(candidate_id, source, "failure", "not_validated", None, "missing_evidence", None, calls, str(exc), candidate_hash=candidate_hash, validation_identity=validation_identity, cache_key=cache_key)
    if calls < 0 or calls > config.max_calls:
        return _record(candidate_id, source, "failure", "not_validated", gradient, "call_budget", None, calls, "calculator call budget exceeded", candidate_hash=candidate_hash, validation_identity=validation_identity, cache_key=cache_key)
    if not converged:
        return _record(candidate_id, source, "not_converged", "not_validated", gradient, "not_converged", None, calls, "stationary point did not converge", candidate_hash=candidate_hash, validation_identity=validation_identity, cache_key=cache_key)
    if not math.isfinite(gradient) or gradient > config.gradient_tolerance:
        return _record(candidate_id, source, "failure", "not_validated", gradient, "gradient_tolerance", None, calls, "gradient norm exceeds TS threshold", candidate_hash=candidate_hash, validation_identity=validation_identity, cache_key=cache_key)
    mode_ok, mode_status = validate_mode(modes, negative_threshold=config.negative_mode_threshold, max_negative_modes=config.max_negative_modes)
    if not mode_ok:
        return _record(candidate_id, source, "failure", "not_validated", gradient, mode_status, None, calls, mode_status, candidate_hash=candidate_hash, validation_identity=validation_identity, cache_key=cache_key)
    connectivity = evidence.get("connectivity")
    observed = None
    path_status = "not_run"
    if config.require_connectivity:
        if not isinstance(connectivity, ConnectivityEvidence):
            return _record(candidate_id, source, "failure", "not_validated", gradient, mode_status, None, calls, "connectivity evidence is required", candidate_hash=candidate_hash, validation_identity=validation_identity, cache_key=cache_key)
        observed = observed_event(connectivity)
        path_connected = evidence.get("path_connected")
        if type(path_connected) is not bool:
            return _record(candidate_id, source, "failure", "not_validated", gradient, mode_status, observed, calls, "path_connected must be a boolean", candidate_hash=candidate_hash, validation_identity=validation_identity, cache_key=cache_key)
        path_status = "validated" if path_connected else "not_validated"
        if path_status != "validated":
            return _record(candidate_id, source, "failure", path_status, gradient, mode_status, observed, calls, "bidirectional endpoint path is not connected", candidate_hash=candidate_hash, validation_identity=validation_identity, cache_key=cache_key)
    return _record(candidate_id, source, "success", path_status, gradient, mode_status, observed, calls, None, candidate_hash=candidate_hash, validation_identity=validation_identity, cache_key=cache_key)


def run_resumable_validation(candidates: Mapping[str, Any], searcher: Callable[[str, Any], Mapping[str, Any]], config: TSValidationConfig, checkpoint: str | Path, *, reserve_calls: Callable[[str, Any], None] | None = None) -> tuple[TSValidationRecord, ...]:
    """Run candidates with an identity-bound checkpoint and pre-call budget gate.

    ``reserve_calls`` is intentionally required by production callers.  It
    must reserve the finite calculator allowance before invoking ``searcher``;
    tests may inject a recorder to prove ordering without running a calculator.
    """

    target = Path(checkpoint)
    prior: dict[str, dict[str, Any]] = {}
    if target.exists():
        prior = json.loads(target.read_text(encoding="utf-8"))
    output: list[TSValidationRecord] = []
    for candidate_id, candidate in candidates.items():
        candidate_hash, validation_identity, cache_key = validation_cache_key(candidate_id, candidate, config)
        cached = prior.get(candidate_id)
        if cached is not None and cached.get("cache_key") == cache_key and cached.get("candidate_hash") == candidate_hash and cached.get("validation_identity") == validation_identity:
            output.append(TSValidationRecord(**cached))
            continue
        if reserve_calls is not None:
            reserve_calls(candidate_id, candidate)
        evidence = searcher(candidate_id, candidate)
        raw_calls = evidence.get("calculator_calls")
        if type(raw_calls) is not int:
            raise ValueError("searcher must return an integer calculator_calls field")
        record = validate_ts_evidence(candidate_id, str(evidence.get("source", "injected")), evidence, config, calls=raw_calls, candidate_hash=candidate_hash, validation_identity=validation_identity, cache_key=cache_key)
        prior[candidate_id] = record.to_dict()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(json.dumps(prior, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        temporary.replace(target)
        output.append(record)
    return tuple(output)
