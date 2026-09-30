"""Reference transition-state validation records and resumable orchestration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

from xtbflow.runtime import BudgetExceeded, BudgetTokenRequired, CalculatorCallToken

from .connectivity import ConnectivityEvidence, observed_event
from .modes import MODE_EVIDENCE_CONTRACT, ModeEvidence, validate_mode


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
    verification_config_version: str = "ts-validation-v2"
    mode_evidence_contract: str = MODE_EVIDENCE_CONTRACT

    def __post_init__(self) -> None:
        for name in ("gradient_tolerance", "negative_mode_threshold"):
            value = getattr(self, name)
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.max_negative_modes) is not int or self.max_negative_modes < 1 or type(self.max_calls) is not int or self.max_calls < 1:
            raise ValueError("max_negative_modes and max_calls must be positive integers")
        if type(self.require_connectivity) is not bool:
            raise ValueError("require_connectivity must be boolean")
        for name in ("reference_protocol_id", "reference_build_hash", "verification_config_version", "mode_evidence_contract"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        if self.mode_evidence_contract != MODE_EVIDENCE_CONTRACT:
            raise ValueError(f"mode_evidence_contract must be {MODE_EVIDENCE_CONTRACT!r}")

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
    if type(calls) is not int or calls < 0:
        return _record(candidate_id, source, "failure", "not_validated", None, "call_budget", None, 0, "calculator calls must be a nonnegative integer", candidate_hash=candidate_hash, validation_identity=validation_identity, cache_key=cache_key)
    try:
        raw_gradient = float(evidence["gradient_norm"])
        converged = evidence["converged"]
        if type(converged) is not bool:
            raise ValueError("converged must be a boolean")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return _record(candidate_id, source, "failure", "not_validated", None, "missing_evidence", None, calls, str(exc), candidate_hash=candidate_hash, validation_identity=validation_identity, cache_key=cache_key)
    # A non-finite gradient is evidence of a failed candidate, not a reason to
    # abort the batch.  TSValidationRecord deliberately rejects NaN/Inf, so
    # retain the original condition in ``error`` while storing ``None``.
    if not math.isfinite(raw_gradient) or raw_gradient < 0:
        return _record(
            candidate_id,
            source,
            "failure",
            "not_validated",
            None,
            "gradient_non_finite" if not math.isfinite(raw_gradient) else "gradient_invalid",
            None,
            calls,
            f"gradient norm must be finite and nonnegative; observed {raw_gradient!r}",
            candidate_hash=candidate_hash,
            validation_identity=validation_identity,
            cache_key=cache_key,
        )
    gradient = raw_gradient
    try:
        modes = ModeEvidence(
            tuple(float(value) for value in evidence["hessian_eigenvalues"]),
            evidence["mode_unit"],
            coordinate_system=evidence.get("mode_coordinate_system", "cartesian"),
            mass_weighted=evidence.get("mode_mass_weighted", False),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return _record(candidate_id, source, "failure", "not_validated", gradient, "missing_evidence", None, calls, str(exc), candidate_hash=candidate_hash, validation_identity=validation_identity, cache_key=cache_key)
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


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    """Load a strict candidate-keyed checkpoint or fail closed."""

    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid TS validation checkpoint: {exc}") from exc
    if not isinstance(payload, dict) or any(not isinstance(key, str) or not isinstance(value, dict) for key, value in payload.items()):
        raise ValueError("TS validation checkpoint must map candidate IDs to record objects")
    return payload


def _cached_record(cached: Mapping[str, Any], *, candidate_id: str, candidate_hash: str, validation_identity: str, cache_key: str) -> TSValidationRecord | None:
    """Validate all cache identity and record fields before bypassing work."""

    if cached.get("candidate_id") != candidate_id or cached.get("cache_key") != cache_key or cached.get("candidate_hash") != candidate_hash or cached.get("validation_identity") != validation_identity:
        return None
    try:
        record = TSValidationRecord(**dict(cached))
    except (TypeError, ValueError, KeyError):
        raise ValueError(f"malformed cached TS validation record for {candidate_id}")
    if record.status not in {"success", "failure"}:
        return None
    return record


def _write_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _finish_token_record(
    token: CalculatorCallToken,
    candidate_id: str,
    record: TSValidationRecord,
    prior: dict[str, dict[str, Any]],
    checkpoint: Path,
) -> None:
    """Persist a result before releasing and settling its calculator token.

    The ordering is intentional: a crash after token settlement must not erase
    the durable result that explains the consumed calls.  If checkpoint I/O
    fails, the active reservation remains recoverable instead of disappearing
    from the ledger.
    """

    prior[candidate_id] = record.to_dict()
    _write_checkpoint(checkpoint, prior)
    token.release()
    token.commit(recorded_calls=record.calculator_calls)


def run_resumable_validation(
    candidates: Mapping[str, Any],
    searcher: Callable[..., Mapping[str, Any]],
    config: TSValidationConfig,
    checkpoint: str | Path,
    *,
    reserve_calls: Callable[[str, Any], None] | None = None,
    budget_token_factory: Callable[[str, Any, int, Mapping[str, Any]], CalculatorCallToken] | None = None,
    require_budget_token: bool = False,
) -> tuple[TSValidationRecord, ...]:
    """Run candidates with an identity-bound checkpoint and optional strict budget tokens.

    Legacy callers retain the two-argument searcher and recorder callback.  A
    production caller opts into ``require_budget_token``; in that mode no
    searcher work begins until a token is issued and the returned call count
    exactly matches the token's pre-invocation accounting.
    """

    if require_budget_token and budget_token_factory is None:
        raise ValueError("production TS validation requires budget_token_factory")
    target = Path(checkpoint)
    prior = _load_checkpoint(target)
    output: list[TSValidationRecord] = []
    for candidate_id, candidate in candidates.items():
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("candidate IDs must be nonempty strings")
        candidate_hash, validation_identity, cache_key = validation_cache_key(candidate_id, candidate, config)
        cached = prior.get(candidate_id)
        if cached is not None:
            record = _cached_record(cached, candidate_id=candidate_id, candidate_hash=candidate_hash, validation_identity=validation_identity, cache_key=cache_key)
            if record is not None:
                output.append(record)
                continue
        token: CalculatorCallToken | None = None
        if require_budget_token:
            token = budget_token_factory(cache_key, candidate, config.max_calls, {"candidate_id": candidate_id, "candidate_hash": candidate_hash, "validation_identity": validation_identity, "cache_key": cache_key})
            if not isinstance(token, CalculatorCallToken):
                raise TypeError("budget_token_factory must return CalculatorCallToken")
            if not token.durable or token.persist_path is None:
                raise ValueError("production TS validation requires a durable calculator token")
            if token.token_id != cache_key:
                raise ValueError("budget token identity must equal the validation cache key")
            if token.reserved_calls < config.max_calls:
                raise ValueError("budget token capacity is smaller than TS validation max_calls")
            evidence: Mapping[str, Any]
            try:
                evidence = searcher(candidate_id, candidate, token)
            except (BudgetExceeded, BudgetTokenRequired) as exc:
                # A token failure is itself a candidate outcome.  Persist the
                # calls already consumed before the guard fired, release any
                # unused allowance, and continue with later candidates.  This
                # keeps a malformed/over-budget candidate from erasing the
                # recovery boundary for the rest of the batch.  If the shared
                # stage budget is exhausted, the next token factory call will
                # still raise before any new search work begins.
                consumed = token.consumed_calls
                failure = _record(
                    candidate_id,
                    "searcher",
                    "failure",
                    "not_validated",
                    None,
                    "call_budget",
                    None,
                    consumed,
                    f"{type(exc).__name__}: {exc}",
                    candidate_hash=candidate_hash,
                    validation_identity=validation_identity,
                    cache_key=cache_key,
                )
                _finish_token_record(token, candidate_id, failure, prior, target)
                output.append(failure)
                continue
            except Exception as exc:
                consumed = token.consumed_calls
                failure = _record(candidate_id, "searcher", "failure", "not_validated", None, "searcher_exception", None, consumed, f"{type(exc).__name__}: {exc}", candidate_hash=candidate_hash, validation_identity=validation_identity, cache_key=cache_key)
                _finish_token_record(token, candidate_id, failure, prior, target)
                output.append(failure)
                continue
            if not isinstance(evidence, Mapping):
                failure = _record(
                    candidate_id,
                    "searcher",
                    "failure",
                    "not_validated",
                    None,
                    "malformed_evidence",
                    None,
                    token.consumed_calls,
                    "searcher must return a mapping",
                    candidate_hash=candidate_hash,
                    validation_identity=validation_identity,
                    cache_key=cache_key,
                )
                _finish_token_record(token, candidate_id, failure, prior, target)
                output.append(failure)
                continue
            raw_calls = evidence.get("calculator_calls")
            if type(raw_calls) is not int:
                failure = _record(
                    candidate_id,
                    "searcher",
                    "failure",
                    "not_validated",
                    None,
                    "malformed_evidence",
                    None,
                    token.consumed_calls,
                    "searcher must return an integer calculator_calls field",
                    candidate_hash=candidate_hash,
                    validation_identity=validation_identity,
                    cache_key=cache_key,
                )
                _finish_token_record(token, candidate_id, failure, prior, target)
                output.append(failure)
                continue
            if raw_calls != token.consumed_calls:
                failure = _record(
                    candidate_id,
                    "searcher",
                    "failure",
                    "not_validated",
                    None,
                    "call_budget",
                    None,
                    token.consumed_calls,
                    "searcher calculator_calls does not match consumed token calls",
                    candidate_hash=candidate_hash,
                    validation_identity=validation_identity,
                    cache_key=cache_key,
                )
                _finish_token_record(token, candidate_id, failure, prior, target)
                output.append(failure)
                continue
        else:
            if reserve_calls is not None:
                reserve_calls(candidate_id, candidate)
            evidence = searcher(candidate_id, candidate)
            if not isinstance(evidence, Mapping):
                raise ValueError("searcher must return a mapping")
            raw_calls = evidence.get("calculator_calls")
            if type(raw_calls) is not int:
                raise ValueError("searcher must return an integer calculator_calls field")
        try:
            record = validate_ts_evidence(candidate_id, str(evidence.get("source", "injected")), evidence, config, calls=raw_calls, candidate_hash=candidate_hash, validation_identity=validation_identity, cache_key=cache_key)
        except Exception as exc:
            # Evidence checks are part of candidate processing.  Persist the
            # failure and settle any reserved calls before moving to the next
            # candidate, so one malformed result cannot terminate a batch.
            consumed = token.consumed_calls if token is not None else raw_calls
            record = _record(
                candidate_id,
                str(evidence.get("source", "injected")),
                "failure",
                "not_validated",
                None,
                "evidence_validation_exception",
                None,
                consumed,
                f"{type(exc).__name__}: {exc}",
                candidate_hash=candidate_hash,
                validation_identity=validation_identity,
                cache_key=cache_key,
            )
        if token is not None:
            _finish_token_record(token, candidate_id, record, prior, target)
        else:
            prior[candidate_id] = record.to_dict()
            _write_checkpoint(target, prior)
        output.append(record)
    return tuple(output)
