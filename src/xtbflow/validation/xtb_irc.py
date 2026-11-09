"""Bounded xTB stationary-point and bidirectional path validation.

This module is a small certification adapter, not a reaction-search model.  It
uses the repository calculator protocol for every energy/force call, charges
each call through a durable :class:`CalculatorCallToken`, builds an explicit
finite-difference Cartesian Hessian, and follows both signs of the selected
negative mode into bounded endpoint minimisations.  Missing optional ASE,
calculator failures, and exhausted budgets are returned as structured
outcomes.  No reference TS, product, or sealed label is read by this module.

The resulting path is a bounded mass-unweighted gradient-following smoke
protocol.  A successful result is suitable for the repository's strict
evidence validator, but it is not a mechanism, barrier, or discovery claim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from xtbflow.calculators import (
    CalculatorBackend,
    CalculatorError,
    CalculationResult,
    MolecularSystem,
)
from xtbflow.runtime import BudgetExceeded, CalculatorCallToken

from .connectivity import ConnectivityEvidence
from .saddle_refinement import SaddleRefinementConfig, refine_saddle
from .ts_search import TSValidationConfig, TSValidationRecord, validate_ts_evidence


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _identity(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _coords(value: Sequence[Sequence[float]], name: str = "coordinates") -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1:] != (3,) or len(array) == 0:
        raise ValueError(f"{name} must have shape [N,3]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array.copy()


@dataclass(frozen=True)
class XTBIRCConfig:
    """Frozen, bounded controls for one candidate certification attempt."""

    hessian_step_angstrom: float = 0.003
    gradient_tolerance: float = 1.0e-4
    negative_mode_threshold: float = 1.0e-4
    irc_initial_step_angstrom: float = 0.20
    irc_step_angstrom: float = 0.10
    irc_min_step_angstrom: float = 1.0e-4
    max_irc_steps: int = 12
    endpoint_fmax_hartree_per_angstrom: float = 2.0e-4
    max_endpoint_steps: int = 250
    endpoint_maxstep_angstrom: float = 0.05
    max_calls: int = 256
    require_ase: bool = True
    saddle_refinement_enabled: bool = False
    saddle_refinement_max_steps: int = 2
    saddle_refinement_trust_radius_angstrom: float = 0.03
    saddle_refinement_strategy: str = "newton"
    protocol_version: str = "xtbflow-xtb-irc-v1"

    def __post_init__(self) -> None:
        for name in (
            "hessian_step_angstrom",
            "gradient_tolerance",
            "negative_mode_threshold",
            "irc_initial_step_angstrom",
            "irc_step_angstrom",
            "irc_min_step_angstrom",
            "endpoint_fmax_hartree_per_angstrom",
            "endpoint_maxstep_angstrom",
        ):
            value = _finite(name, getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if type(self.max_irc_steps) is not int or self.max_irc_steps < 1:
            raise ValueError("max_irc_steps must be a positive integer")
        if type(self.max_endpoint_steps) is not int or self.max_endpoint_steps < 1:
            raise ValueError("max_endpoint_steps must be a positive integer")
        if type(self.max_calls) is not int or self.max_calls < 1:
            raise ValueError("max_calls must be a positive integer")
        if type(self.require_ase) is not bool:
            raise ValueError("require_ase must be boolean")
        if type(self.saddle_refinement_enabled) is not bool:
            raise ValueError("saddle_refinement_enabled must be boolean")
        if type(self.saddle_refinement_max_steps) is not int or self.saddle_refinement_max_steps < 1:
            raise ValueError("saddle_refinement_max_steps must be a positive integer")
        if not math.isfinite(float(self.saddle_refinement_trust_radius_angstrom)) or self.saddle_refinement_trust_radius_angstrom <= 0:
            raise ValueError("saddle_refinement_trust_radius_angstrom must be positive")
        if self.saddle_refinement_strategy not in {"newton", "mode_follow"}:
            raise ValueError("saddle_refinement_strategy must be 'newton' or 'mode_follow'")
        if not isinstance(self.protocol_version, str) or not self.protocol_version.strip():
            raise ValueError("protocol_version must be nonempty")

    @property
    def identity(self) -> str:
        return _identity(asdict(self))

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class XTBIRCResult:
    """Serializable result with explicit failure and claim boundaries."""

    status: str
    reason_code: str
    candidate_id: str
    calculator_calls: int
    config_identity: str
    validation: Mapping[str, Any] | None
    evidence: Mapping[str, Any]
    claim_level: str
    claim_limit: str

    def __post_init__(self) -> None:
        if self.status not in {"success", "attempted_unresolved", "blocked_backend", "budget_exhausted", "execution_failed"}:
            raise ValueError("unsupported xTB IRC status")
        if not self.reason_code.strip() or not self.candidate_id.strip():
            raise ValueError("reason_code and candidate_id are required")
        if type(self.calculator_calls) is not int or self.calculator_calls < 0:
            raise ValueError("calculator_calls must be nonnegative")
        if self.claim_level not in {"software_demo", "proxy_metric", "strict_validation"}:
            raise ValueError("unsupported claim level")

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Evaluation:
    energy: float
    forces: np.ndarray
    result: CalculationResult


def _evaluate(
    backend: CalculatorBackend,
    template: MolecularSystem,
    coordinates: np.ndarray,
    token: CalculatorCallToken,
) -> _Evaluation:
    system = template.with_coordinates(coordinates.tolist())
    result = backend.evaluate(system, operation="energy_forces", budget_token=token)
    if result.status != "success" or result.energy is None or result.forces is None:
        raise CalculatorError(result.error_message or f"calculator returned {result.status}")
    forces = _coords(result.forces, "calculator forces")
    if len(forces) != len(template.symbols):
        raise ValueError("calculator force shape does not match atom inventory")
    return _Evaluation(float(result.energy), forces, result)


def _gradient_norm(forces: np.ndarray) -> float:
    return float(np.max(np.abs(forces)))


_COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}


def _bond_matrix(symbols: Sequence[str], coordinates: np.ndarray) -> tuple[tuple[int, ...], ...]:
    """Build an observed single-bond graph from endpoint distances.

    This graph is intentionally an endpoint observation, never a proposed
    product.  Multiple-bond perception is outside this bounded smoke scope;
    an unknown element or ambiguous distance fails closed.
    """

    n = len(symbols)
    if len(coordinates) != n:
        raise ValueError("bond graph atom inventory mismatch")
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        if symbols[i] not in _COVALENT_RADII:
            raise ValueError(f"no covalent radius for {symbols[i]!r}")
        for j in range(i + 1, n):
            if symbols[j] not in _COVALENT_RADII:
                raise ValueError(f"no covalent radius for {symbols[j]!r}")
            distance = float(np.linalg.norm(coordinates[i] - coordinates[j]))
            threshold = 1.25 * (_COVALENT_RADII[symbols[i]] + _COVALENT_RADII[symbols[j]])
            if distance <= threshold:
                matrix[i][j] = matrix[j][i] = 1
    return tuple(tuple(row) for row in matrix)


def _ase_endpoint_minimum(
    backend: CalculatorBackend,
    template: MolecularSystem,
    start: np.ndarray,
    token: CalculatorCallToken,
    config: XTBIRCConfig,
) -> tuple[np.ndarray, _Evaluation, bool, int]:
    """Refine one IRC endpoint through ASE BFGS using the shared adapter."""

    try:
        from ase import Atoms
        from ase.calculators.calculator import Calculator, all_changes
        from ase.optimize import BFGS
        from ase.units import Hartree
    except ImportError as exc:  # pragma: no cover - optional boundary
        raise RuntimeError(f"ASE endpoint backend unavailable: {exc}") from exc

    class _Calculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self) -> None:
            super().__init__()
            self.last: _Evaluation | None = None

        def calculate(self, atoms: Any = None, properties: Any = ("energy", "forces"), system_changes: Any = all_changes) -> None:
            super().calculate(atoms, properties, system_changes)
            current = _coords(atoms.get_positions())
            self.last = _evaluate(backend, template, current, token)
            self.results = {
                "energy": self.last.energy * Hartree,
                "forces": self.last.forces * Hartree,
            }

    atoms = Atoms(symbols=template.symbols, positions=start)
    calculator = _Calculator()
    atoms.calc = calculator
    optimizer = BFGS(atoms, logfile=None, maxstep=config.endpoint_maxstep_angstrom)
    converged = bool(optimizer.run(
        fmax=config.endpoint_fmax_hartree_per_angstrom * Hartree,
        steps=config.max_endpoint_steps,
    ))
    # ASE may have a cached result after the final step.  Force one explicit
    # contract evaluation so the returned evidence and call ledger agree.
    final = _evaluate(backend, template, _coords(atoms.get_positions()), token)
    converged = converged and _gradient_norm(final.forces) <= config.endpoint_fmax_hartree_per_angstrom
    return _coords(atoms.get_positions()), final, converged, int(optimizer.nsteps)


def _follow_path(
    backend: CalculatorBackend,
    template: MolecularSystem,
    start: np.ndarray,
    direction: np.ndarray,
    token: CalculatorCallToken,
    config: XTBIRCConfig,
) -> tuple[np.ndarray, _Evaluation, list[dict[str, Any]]]:
    """Take bounded downhill steps before endpoint minimisation."""

    current = start.copy()
    current_eval = _evaluate(backend, template, current, token)
    frames: list[dict[str, Any]] = [{"coordinates_angstrom": current.tolist(), "energy_hartree": current_eval.energy}]
    step = config.irc_step_angstrom
    for _ in range(config.max_irc_steps):
        if _gradient_norm(current_eval.forces) <= config.endpoint_fmax_hartree_per_angstrom:
            break
        downhill = current_eval.forces / max(float(np.linalg.norm(current_eval.forces)), 1.0e-12)
        trial = current + step * downhill
        trial_eval = _evaluate(backend, template, trial, token)
        if trial_eval.energy <= current_eval.energy + 1.0e-8:
            current, current_eval = trial, trial_eval
            frames.append({"coordinates_angstrom": current.tolist(), "energy_hartree": current_eval.energy})
            step = min(config.irc_step_angstrom, step * 1.2)
        else:
            step *= 0.5
            if step < config.irc_min_step_angstrom:
                break
    return current, current_eval, frames


def run_xtb_irc_certification(
    backend: CalculatorBackend,
    system: MolecularSystem,
    *,
    token: CalculatorCallToken,
    candidate_id: str,
    config: XTBIRCConfig = XTBIRCConfig(),
) -> XTBIRCResult:
    """Run one bounded xTB Hessian/dual-path certification attempt."""

    if not isinstance(token, CalculatorCallToken):
        raise TypeError("xTB IRC certification requires a CalculatorCallToken")
    if not token.durable:
        raise ValueError("xTB IRC certification requires a durable calculator token")
    if config.require_ase:
        try:
            import ase  # noqa: F401
        except ImportError as exc:
            return XTBIRCResult(
                "blocked_backend", "CERTIFICATION_BACKEND_UNAVAILABLE", candidate_id,
                token.consumed_calls, config.identity, None,
                {"backend_error": f"ASE endpoint backend unavailable: {exc}"},
                "software_demo",
                "No TS/IRC certification is claimed when the endpoint backend is unavailable.",
            )

    coordinates = _coords(system.coordinates)
    q = int(coordinates.size)
    calls_before = token.consumed_calls
    try:
        saddle_refinement_evidence: dict[str, Any] | None = None
        if config.saddle_refinement_enabled:
            refinement = refine_saddle(
                backend,
                system,
                coordinates,
                token=token,
                config=SaddleRefinementConfig(
                    hessian_step_angstrom=config.hessian_step_angstrom,
                    gradient_tolerance=config.gradient_tolerance,
                    negative_mode_threshold=config.negative_mode_threshold,
                    trust_radius_angstrom=config.saddle_refinement_trust_radius_angstrom,
                    max_steps=config.saddle_refinement_max_steps,
                    max_calls=config.max_calls,
                    strategy=config.saddle_refinement_strategy,
                ),
            )
            saddle_refinement_evidence = refinement.to_mapping()
            coordinates = _coords(refinement.coordinates)
            if refinement.status != "success":
                return XTBIRCResult(
                    "attempted_unresolved" if refinement.status != "budget_exhausted" else "budget_exhausted",
                    f"SADDLE_REFINEMENT_{refinement.reason_code}",
                    candidate_id,
                    token.consumed_calls - calls_before,
                    config.identity,
                    None,
                    {"saddle_refinement": saddle_refinement_evidence},
                    "proxy_metric",
                    "Saddle refinement did not reach a strict stationary point; no TS/IRC certification is claimed.",
                )
        base = _evaluate(backend, system, coordinates, token)
        hessian = np.zeros((q, q), dtype=float)
        step = config.hessian_step_angstrom
        flat = coordinates.reshape(-1)
        for index in range(q):
            plus = flat.copy()
            minus = flat.copy()
            plus[index] += step
            minus[index] -= step
            plus_eval = _evaluate(backend, system, plus.reshape(coordinates.shape), token)
            minus_eval = _evaluate(backend, system, minus.reshape(coordinates.shape), token)
            hessian[:, index] = -(
                plus_eval.forces.reshape(-1) - minus_eval.forces.reshape(-1)
            ) / (2.0 * step)
        hessian = 0.5 * (hessian + hessian.T)
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        negative = np.flatnonzero(eigenvalues < -config.negative_mode_threshold)
        mode = eigenvectors[:, int(np.argmin(eigenvalues))].reshape(coordinates.shape)
        mode /= max(float(np.linalg.norm(mode)), 1.0e-12)
        ts_converged = _gradient_norm(base.forces) <= config.gradient_tolerance

        endpoints: list[dict[str, Any]] = []
        for sign in (-1.0, 1.0):
            seed = coordinates + sign * config.irc_initial_step_angstrom * mode
            path_start, path_eval, frames = _follow_path(backend, system, seed, sign * mode, token, config)
            endpoint, endpoint_eval, endpoint_converged, optimizer_steps = _ase_endpoint_minimum(
                backend, system, path_start, token, config
            )
            frames.append({"coordinates_angstrom": endpoint.tolist(), "energy_hartree": endpoint_eval.energy})
            endpoints.append({
                "direction": int(sign),
                "coordinates_angstrom": endpoint.tolist(),
                "energy_hartree": endpoint_eval.energy,
                "gradient_norm_hartree_per_angstrom": _gradient_norm(endpoint_eval.forces),
                "minimum_converged": endpoint_converged,
                "optimizer_steps": optimizer_steps,
                "path": frames,
            })

        endpoint_ok = all(bool(row["minimum_converged"]) for row in endpoints)
        product = min(endpoints, key=lambda row: float(row["energy_hartree"]))
        reactant_bonds = _bond_matrix(system.symbols, coordinates)
        product_bonds = _bond_matrix(system.symbols, _coords(product["coordinates_angstrom"]))
        connectivity = ConnectivityEvidence(reactant_bonds, product_bonds)
        path_connected = endpoint_ok and all(len(row["path"]) >= 2 for row in endpoints)
        evidence = {
            "gradient_norm": _gradient_norm(base.forces),
            "converged": bool(ts_converged),
            "hessian_eigenvalues": eigenvalues.tolist(),
            "mode_unit": "hartree_per_angstrom2",
            "mode_coordinate_system": "cartesian",
            "mode_mass_weighted": False,
            "negative_mode_count": int(len(negative)),
            "negative_mode": mode.tolist(),
            "ts_energy_hartree": base.energy,
            "endpoints": endpoints,
            "irc_method": "bounded_cartesian_gradient_following_v1",
            "connectivity": connectivity,
            "path_connected": bool(path_connected),
            "protocol_version": config.protocol_version,
        }
        if saddle_refinement_evidence is not None:
            evidence["saddle_refinement"] = saddle_refinement_evidence
        validation_config = TSValidationConfig(
            gradient_tolerance=config.gradient_tolerance,
            negative_mode_threshold=config.negative_mode_threshold,
            max_negative_modes=1,
            require_connectivity=True,
            max_calls=config.max_calls,
            reference_protocol_id=base.result.protocol_id,
            reference_build_hash=base.result.calculator_build_hash or "unavailable",
            verification_config_version=config.protocol_version,
        )
        record = validate_ts_evidence(
            candidate_id,
            "xtbflow-xtb-irc-v1",
            evidence,
            validation_config,
            calls=token.consumed_calls - calls_before,
        )
        status = "success" if record.status == "success" else "attempted_unresolved"
        reason = "CERTIFIED" if status == "success" else (record.mode_status or "TS_VALIDATION_FAILED")
        return XTBIRCResult(
            status,
            reason,
            candidate_id,
            token.consumed_calls - calls_before,
            config.identity,
            record.to_dict(),
            evidence,
            "strict_validation" if status == "success" else "proxy_metric",
            "A successful record validates this frozen xTB software protocol; it does not establish a discovered pathway, barrier, mechanism, or chemical accuracy.",
        )
    except BudgetExceeded as exc:
        return XTBIRCResult(
            "budget_exhausted", "BUDGET_EXCEEDED", candidate_id,
            token.consumed_calls - calls_before, config.identity, None,
            {"error": f"{type(exc).__name__}: {exc}"}, "software_demo",
            "The attempt is unresolved and all consumed calls remain charged.",
        )
    except RuntimeError as exc:
        message = str(exc)
        status = "blocked_backend" if "ASE endpoint backend unavailable" in message else "execution_failed"
        return XTBIRCResult(
            status, "CERTIFICATION_BACKEND_UNAVAILABLE" if status == "blocked_backend" else "CERTIFICATION_EXECUTION",
            candidate_id, token.consumed_calls - calls_before, config.identity, None,
            {"error": f"{type(exc).__name__}: {exc}"}, "software_demo",
            "No TS/IRC certification is claimed after an unavailable or failed backend stage.",
        )
    except Exception as exc:
        return XTBIRCResult(
            "execution_failed", "CERTIFICATION_EXECUTION", candidate_id,
            token.consumed_calls - calls_before, config.identity, None,
            {"error": f"{type(exc).__name__}: {exc}"}, "software_demo",
            "No TS/IRC certification is claimed after an execution failure.",
        )


__all__ = ["XTBIRCConfig", "XTBIRCResult", "run_xtb_irc_certification"]
