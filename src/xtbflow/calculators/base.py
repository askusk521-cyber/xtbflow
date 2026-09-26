"""Contracts for explicit energy/force calculator adapters.

The adapters in this module deliberately stop at a small, auditable protocol:
an implementation must provide a pinned method identity, explicit charge and
multiplicity, finite Angstrom coordinates, convergence status, and forces in
the declared sign convention.  No external calculator API is guessed here;
the concrete adapters accept an injected callable until an installed backend
has been independently qualified.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any, Callable, Iterable, Mapping, Sequence


OPERATIONS = frozenset({"energy", "forces", "energy_forces"})
RESULT_STATUSES = frozenset({"success", "failure", "unavailable", "not_converged"})
ERROR_CATEGORIES = frozenset({"input", "unavailable", "execution", "convergence", "nonfinite", "unsupported_operation", "protocol"})


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    try:
        _canonical(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON-compatible") from exc
    return dict(value)


@dataclass(frozen=True)
class CalculatorProtocol:
    """Frozen calculator identity and numerical convention.

    ``backend`` is part of the identity so a CPU and CUDA calculation cannot
    silently share a cache entry.  ``parameters`` carries explicit SCC,
    electronic-temperature, initialization, and environment choices; unknown
    values remain visible instead of being filled with defaults by an adapter.
    """

    protocol_id: str
    calculator: str
    method: str
    backend: str = "cpu"
    coordinate_unit: str = "angstrom"
    energy_unit: str = "hartree"
    force_unit: str = "hartree/angstrom"
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("protocol_id", "calculator", "method", "backend", "coordinate_unit", "energy_unit", "force_unit"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.backend not in {"cpu", "cuda"}:
            raise ValueError("backend must be 'cpu' or 'cuda'")
        if self.coordinate_unit != "angstrom":
            raise ValueError("calculator coordinates must be in angstrom")
        if self.energy_unit != "hartree" or self.force_unit != "hartree/angstrom":
            raise ValueError("v0 calculator units are hartree and hartree/angstrom")
        _mapping(self.parameters, "parameters")

    @property
    def identity(self) -> str:
        """Stable hash for cache keys and run manifests."""

        payload = asdict(self)
        payload["parameters"] = dict(self.parameters)
        return hashlib.sha256(_canonical(payload).encode()).hexdigest()


@dataclass(frozen=True)
class MolecularSystem:
    """One explicit molecular input; missing charge/spin cannot be defaulted."""

    symbols: tuple[str, ...]
    coordinates: tuple[tuple[float, float, float], ...]
    charge: int
    multiplicity: int
    environment: Mapping[str, Any] = field(default_factory=dict)
    system_id: str = ""

    def __post_init__(self) -> None:
        if not self.symbols or any(not isinstance(symbol, str) or not symbol.strip() for symbol in self.symbols):
            raise ValueError("symbols must be a nonempty sequence of element names")
        if len(self.coordinates) != len(self.symbols):
            raise ValueError("coordinates must match the atom inventory")
        for row in self.coordinates:
            if len(row) != 3:
                raise ValueError("each coordinate must have three components")
            for value in row:
                _finite_number("coordinate", value)
        if type(self.charge) is not int:
            raise ValueError("charge must be an explicit integer")
        if type(self.multiplicity) is not int or self.multiplicity < 1:
            raise ValueError("multiplicity must be an explicit positive integer")
        _mapping(self.environment, "environment")
        if self.system_id and not self.system_id.strip():
            raise ValueError("system_id cannot be blank")

    def payload(self) -> dict[str, Any]:
        """Return only deployment-visible, JSON-compatible input fields."""

        return {
            "symbols": list(self.symbols),
            "coordinates": [[float(value) for value in row] for row in self.coordinates],
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "environment": dict(self.environment),
        }

    @property
    def input_hash(self) -> str:
        return hashlib.sha256(_canonical(self.payload()).encode()).hexdigest()

    def with_coordinates(self, coordinates: Sequence[Sequence[float]]) -> "MolecularSystem":
        return MolecularSystem(
            self.symbols,
            tuple(tuple(float(value) for value in row) for row in coordinates),
            self.charge,
            self.multiplicity,
            self.environment,
            self.system_id,
        )


@dataclass(frozen=True)
class CalculatorCapabilities:
    """Observed capability matrix; unknown is distinct from unavailable."""

    calculator: str
    status: str
    detail: str
    version: str | None = None
    build_hash: str | None = None
    operations: tuple[str, ...] = ()
    backends: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.calculator.strip() or self.status not in {"pass", "unknown", "unavailable"} or not self.detail.strip():
            raise ValueError("capability identity, status, and detail are required")
        if any(operation not in OPERATIONS for operation in self.operations):
            raise ValueError("capabilities contain an unsupported operation")
        if any(backend not in {"cpu", "cuda"} for backend in self.backends):
            raise ValueError("capabilities contain an unsupported backend")


@dataclass(frozen=True)
class CalculationResult:
    """One calculation result or an explicit fail-closed outcome."""

    calculator: str
    protocol_id: str
    input_hash: str
    charge: int
    multiplicity: int
    operation: str
    status: str = "success"
    converged: bool = True
    energy: float | None = None
    forces: tuple[tuple[float, float, float], ...] | None = None
    error_category: str | None = None
    error_message: str | None = None
    cache_hit: bool = False
    calculator_calls: int = 1
    calculator_build_hash: str | None = None
    input_file_hash: str | None = None
    path_status: str = "not_requested"

    def __post_init__(self) -> None:
        for name in ("calculator", "protocol_id", "input_hash"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.operation not in OPERATIONS:
            raise ValueError(f"unsupported operation: {self.operation}")
        if self.status not in RESULT_STATUSES:
            raise ValueError(f"unsupported result status: {self.status}")
        if type(self.charge) is not int or type(self.multiplicity) is not int or self.multiplicity < 1:
            raise ValueError("result charge/multiplicity must be explicit")
        if type(self.converged) is not bool:
            raise ValueError("converged must be boolean")
        if self.status == "success" and not self.converged:
            raise ValueError("successful results must be converged")
        if self.status == "not_converged" and self.converged:
            raise ValueError("not_converged results must report converged=false")
        if self.status == "success":
            if self.operation in {"energy", "energy_forces"}:
                _finite_number("energy", self.energy)
            if self.operation in {"forces", "energy_forces"}:
                if self.forces is None:
                    raise ValueError("successful force operations require forces")
                _validate_forces(self.forces)
        if self.error_category is not None and self.error_category not in ERROR_CATEGORIES:
            raise ValueError(f"unsupported error category: {self.error_category}")
        if self.status != "success" and not self.error_category:
            raise ValueError("failed results require an error category")
        if self.cache_hit and self.calculator_calls != 0:
            raise ValueError("cache hits must record zero calculator calls")
        if type(self.calculator_calls) is not int or self.calculator_calls < 0:
            raise ValueError("calculator_calls must be a nonnegative integer")
        for name in ("calculator_build_hash", "input_file_hash"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty string when provided")
        if self.path_status not in {"not_requested", "not_run", "not_validated", "validated"}:
            raise ValueError("unsupported path validation status")

    @classmethod
    def failure(cls, system: MolecularSystem, protocol: CalculatorProtocol, operation: str, *, status: str, category: str, message: str, calculator_calls: int = 0) -> "CalculationResult":
        return cls(
            calculator=protocol.calculator,
            protocol_id=protocol.protocol_id,
            input_hash=system.input_hash,
            charge=system.charge,
            multiplicity=system.multiplicity,
            operation=operation,
            status=status,
            converged=False,
            error_category=category,
            error_message=message,
            calculator_calls=calculator_calls,
        )

    def to_execution_result(self):
        """Map into the runtime ledger's metered result without hiding failure."""

        from xtbflow.runtime.runner import ExecutionResult

        status = "success" if self.status == "success" else "failure"
        return ExecutionResult(
            operation=self.operation,
            status=status,
            cache_hit=self.cache_hit,
            calculator_calls=self.calculator_calls,
            error=self.error_message or self.error_category,
        )


def _validate_forces(forces: Sequence[Sequence[float]]) -> None:
    if not forces:
        raise ValueError("forces must contain one row per atom")
    for row in forces:
        if len(row) != 3:
            raise ValueError("each force row must have three components")
        for value in row:
            _finite_number("force", value)


class CalculatorError(RuntimeError):
    """Base error for a calculator adapter; callers can classify it safely."""

    category = "execution"


class CalculatorUnavailable(CalculatorError):
    category = "unavailable"


class CalculatorConvergenceError(CalculatorError):
    category = "convergence"


class CalculatorProtocolError(CalculatorError):
    category = "protocol"


class CalculatorBackend(ABC):
    """Common adapter interface with per-system failure isolation."""

    protocol: CalculatorProtocol

    @property
    @abstractmethod
    def capabilities(self) -> CalculatorCapabilities:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, system: MolecularSystem, *, operation: str = "energy_forces") -> CalculationResult:
        raise NotImplementedError

    def evaluate_batch(self, systems: Iterable[MolecularSystem], *, operation: str = "energy_forces") -> tuple[CalculationResult, ...]:
        """Evaluate ragged inputs while recording one explicit result per system."""

        results: list[CalculationResult] = []
        for system in systems:
            try:
                results.append(self.evaluate(system, operation=operation))
            except CalculatorError as exc:
                calls = 0 if exc.category in {"unavailable", "input", "protocol"} else 1
                results.append(CalculationResult.failure(system, self.protocol, operation, status="failure", category=exc.category, message=str(exc), calculator_calls=calls))
            except Exception as exc:  # Backend failures must not erase neighboring systems.
                results.append(CalculationResult.failure(system, self.protocol, operation, status="failure", category="execution", message=f"{type(exc).__name__}: {exc}", calculator_calls=1))
        return tuple(results)


def coerce_backend_output(
    output: CalculationResult | Mapping[str, Any],
    system: MolecularSystem,
    protocol: CalculatorProtocol,
    *,
    operation: str,
) -> CalculationResult:
    """Validate an injected backend payload without supplying hidden defaults."""

    if isinstance(output, CalculationResult):
        result = output
    elif isinstance(output, Mapping):
        if "converged" not in output or "charge" not in output or "multiplicity" not in output:
            raise CalculatorProtocolError("backend output must state converged, charge, and multiplicity")
        try:
            result = CalculationResult(
                calculator=protocol.calculator,
                protocol_id=protocol.protocol_id,
                input_hash=system.input_hash,
                charge=output["charge"],
                multiplicity=output["multiplicity"],
                operation=operation,
                status="success" if output["converged"] else "not_converged",
                converged=output["converged"],
                energy=output.get("energy"),
                forces=tuple(tuple(float(value) for value in row) for row in output["forces"]) if output.get("forces") is not None else None,
                error_category="convergence" if not output["converged"] else None,
                error_message=output.get("error"),
            )
        except (TypeError, ValueError) as exc:
            raise CalculatorProtocolError(f"backend output violates the calculator contract: {exc}") from exc
    else:
        raise CalculatorProtocolError("backend output must be CalculationResult or a mapping")
    if result.calculator != protocol.calculator or result.protocol_id != protocol.protocol_id:
        raise CalculatorProtocolError("backend result identity does not match the requested protocol")
    if result.input_hash != system.input_hash or result.charge != system.charge or result.multiplicity != system.multiplicity:
        raise CalculatorProtocolError("backend result state does not match the requested system")
    if result.operation != operation:
        raise CalculatorProtocolError("backend result operation does not match the request")
    if result.status == "success" and result.forces is not None and len(result.forces) != len(system.symbols):
        raise CalculatorProtocolError("backend force rows do not match the atom inventory")
    return result


def finite_difference_forces(backend: CalculatorBackend, system: MolecularSystem, *, step: float = 1e-4) -> tuple[tuple[float, float, float], ...]:
    """Estimate ``F=-dE/dR`` using central differences and energy calls only."""

    step = _finite_number("step", step)
    if step <= 0:
        raise ValueError("step must be positive")
    values: list[list[float]] = [[0.0, 0.0, 0.0] for _ in system.symbols]
    for atom in range(len(system.symbols)):
        for axis in range(3):
            plus = [list(row) for row in system.coordinates]
            minus = [list(row) for row in system.coordinates]
            plus[atom][axis] += step
            minus[atom][axis] -= step
            plus_result = backend.evaluate(system.with_coordinates(plus), operation="energy")
            minus_result = backend.evaluate(system.with_coordinates(minus), operation="energy")
            if plus_result.status != "success" or minus_result.status != "success" or plus_result.energy is None or minus_result.energy is None:
                raise CalculatorError("finite-difference force requires two converged energy results")
            values[atom][axis] = -(plus_result.energy - minus_result.energy) / (2.0 * step)
    return tuple(tuple(row) for row in values)
