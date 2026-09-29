"""Bounded ASE dimer search over a metered xtbflow calculator backend."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from xtbflow.calculators import CalculatorBackend, MolecularSystem
from xtbflow.runtime import (
    BudgetTokenRequired,
    CalculatorCallToken,
)

from .ase_calculator import MeteredASECalculator

try:
    from ase import Atoms
    from ase.constraints import FixConstraint
    from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate
    from ase.units import Hartree
except ImportError as exc:  # pragma: no cover - optional dependency boundary.
    raise ImportError(
        "xtbflow.validation.ase_dimer requires the optional ASE dependency"
    ) from exc

@dataclass(frozen=True)
class ASEDimerConfig:
    """Frozen ASE dimer controls with explicit units and a stable identity."""

    fmax_eV_per_angstrom: float = 0.05
    max_steps: int = 50
    dimer_separation_angstrom: float = 1.0e-3
    max_num_rot: int = 1
    maximum_translation_angstrom: float = 0.1
    trial_translation_step_angstrom: float = 1.0e-3
    random_seed: int = 0
    remove_rigid_body_modes: bool = True
    rigid_body_tolerance: float = 1.0e-10
    driver_version: str = "ase-dimer-v2"

    def __post_init__(self) -> None:
        for name in (
            "fmax_eV_per_angstrom",
            "dimer_separation_angstrom",
            "maximum_translation_angstrom",
            "trial_translation_step_angstrom",
            "rigid_body_tolerance",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if type(self.max_num_rot) is not int or self.max_num_rot < 1:
            raise ValueError("max_num_rot must be a positive integer")
        if type(self.random_seed) is not int:
            raise ValueError("random_seed must be an integer")
        if type(self.remove_rigid_body_modes) is not bool:
            raise ValueError("remove_rigid_body_modes must be boolean")
        if not isinstance(self.driver_version, str) or not self.driver_version.strip():
            raise ValueError("driver_version must be a nonempty string")

    @property
    def identity(self) -> str:
        encoded = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ASEDimerResult:
    """One dimer-search outcome; mode/path qualification is not implied."""

    status: str
    converged: bool
    coordinates_angstrom: tuple[tuple[float, float, float], ...]
    energy_hartree: float | None
    forces_hartree_per_angstrom: tuple[tuple[float, float, float], ...] | None
    gradient_norm_hartree_per_angstrom: float | None
    eigenmode: tuple[tuple[float, float, float], ...] | None
    curvature_eV_per_angstrom2: float | None
    rigid_body_rank: int
    optimizer_steps: int
    calculator_calls: int
    artifact_files: tuple[str, ...]
    config_identity: str
    error: str | None = None
    def __post_init__(self) -> None:
        if self.status not in {"success", "not_converged", "failure"}:
            raise ValueError("unsupported ASE dimer status")
        if type(self.converged) is not bool:
            raise ValueError("converged must be boolean")
        if type(self.rigid_body_rank) is not int or self.rigid_body_rank < 0:
            raise ValueError("rigid_body_rank must be nonnegative")
        if type(self.optimizer_steps) is not int or self.optimizer_steps < 0:
            raise ValueError("optimizer_steps must be nonnegative")
        if type(self.calculator_calls) is not int or self.calculator_calls < 0:
            raise ValueError("calculator_calls must be nonnegative")
        if not isinstance(self.config_identity, str) or not self.config_identity:
            raise ValueError("config_identity is required")

    def stationary_fields(self) -> dict[str, Any]:
        """Return search-only fields without fabricating mode/path evidence."""

        return {
            "source": "ase-dimer",
            "converged": self.converged,
            "gradient_norm": self.gradient_norm_hartree_per_angstrom,
            "calculator_calls": self.calculator_calls,
            "search_status": self.status,
            "search_error": self.error,
            "search_coordinates_angstrom": self.coordinates_angstrom,
            "dimer_curvature_eV_per_angstrom2": self.curvature_eV_per_angstrom2,
            "dimer_eigenmode": self.eigenmode,
            "rigid_body_rank": self.rigid_body_rank,
            "artifact_files": self.artifact_files,
            "config_identity": self.config_identity,
        }


@dataclass(frozen=True)
class DimerPreflightResult:
    """Three-call force/curvature screen before a bounded dimer search."""

    status: str
    normalized_mode: tuple[tuple[float, float, float], ...]
    energy_hartree: float | None
    projected_forces_hartree_per_angstrom: (
        tuple[tuple[float, float, float], ...] | None
    )
    projected_force_norm_hartree_per_angstrom: float | None
    hessian_vector_product_hartree_per_angstrom2: (
        tuple[tuple[float, float, float], ...] | None
    )
    mode_curvature_hartree_per_angstrom2: float | None
    mode_curvature_eV_per_angstrom2: float | None
    rigid_body_rank: int
    calculator_calls: int
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"success", "failure"}:
            raise ValueError("unsupported dimer preflight status")
        if type(self.rigid_body_rank) is not int or self.rigid_body_rank < 0:
            raise ValueError("rigid_body_rank must be nonnegative")
        if type(self.calculator_calls) is not int or self.calculator_calls < 0:
            raise ValueError("calculator_calls must be nonnegative")
        if self.status == "success" and self.error is not None:
            raise ValueError("successful preflight cannot carry an error")


def rigid_body_basis(
    coordinates: Sequence[Sequence[float]],
    *,
    tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Return an orthonormal Cartesian basis for translations and rotations."""

    positions = np.asarray(coordinates, dtype=float)
    if positions.ndim != 2 or positions.shape[1:] != (3,) or len(positions) < 1:
        raise ValueError("coordinates must have shape [N,3] with N >= 1")
    if not np.isfinite(positions).all():
        raise ValueError("coordinates must contain finite values")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")

    centered = positions - positions.mean(axis=0, keepdims=True)
    axes = np.eye(3, dtype=float)
    columns = [np.tile(axis, (len(positions), 1)).reshape(-1) for axis in axes]
    for axis in axes:
        angular = np.broadcast_to(axis, centered.shape)
        columns.append(np.cross(angular, centered).reshape(-1))
    matrix = np.stack(columns, axis=1)
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    threshold = tolerance * max(float(singular_values[0]), 1.0)
    rank = int(np.count_nonzero(singular_values > threshold))
    return left[:, :rank]


def project_rigid_body_components(
    coordinates: Sequence[Sequence[float]],
    vectors: Sequence[Sequence[float]],
    *,
    tolerance: float = 1.0e-10,
) -> tuple[np.ndarray, int]:
    """Remove instantaneous rigid translations and rotations from vectors."""

    positions = np.asarray(coordinates, dtype=float)
    values = np.asarray(vectors, dtype=float)
    if values.shape != positions.shape:
        raise ValueError("vectors must have the same [N,3] shape as coordinates")
    if not np.isfinite(values).all():
        raise ValueError("vectors must contain finite values")
    basis = rigid_body_basis(positions, tolerance=tolerance)
    flat = values.reshape(-1)
    projected = flat - basis @ (basis.T @ flat)
    return projected.reshape(values.shape), int(basis.shape[1])


class RigidBodyProjection(FixConstraint):
    """Project ASE coordinate updates and forces out of rigid-body modes."""

    def __init__(self, tolerance: float = 1.0e-10) -> None:
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        self.tolerance = float(tolerance)

    def get_removed_dof(self, atoms: Any) -> int:
        basis = rigid_body_basis(
            atoms.get_positions(), tolerance=self.tolerance
        )
        return int(basis.shape[1])

    def adjust_positions(self, atoms: Any, new: np.ndarray) -> None:
        current = np.asarray(atoms.get_positions(), dtype=float)
        projected, _ = project_rigid_body_components(
            current, np.asarray(new, dtype=float) - current, tolerance=self.tolerance
        )
        new[:] = current + projected

    def adjust_forces(self, atoms: Any, forces: np.ndarray) -> None:
        projected, _ = project_rigid_body_components(
            atoms.get_positions(), forces, tolerance=self.tolerance
        )
        forces[:] = projected

    def todict(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "kwargs": {"tolerance": self.tolerance},
        }


def _normalized_mode(
    value: Sequence[Sequence[float]],
    coordinates: Sequence[Sequence[float]],
    config: ASEDimerConfig,
) -> tuple[np.ndarray, int]:
    mode = np.asarray(value, dtype=float)
    positions = np.asarray(coordinates, dtype=float)
    if mode.shape != positions.shape or mode.ndim != 2 or mode.shape[1:] != (3,):
        raise ValueError("initial_mode must have shape [N,3]")
    if not np.isfinite(mode).all():
        raise ValueError("initial_mode must contain finite values")
    rigid_rank = 0
    if config.remove_rigid_body_modes:
        mode, rigid_rank = project_rigid_body_components(
            positions, mode, tolerance=config.rigid_body_tolerance
        )
    norm = float(np.linalg.norm(mode))
    if norm <= config.rigid_body_tolerance:
        raise ValueError("initial_mode has no internal component after rigid-body projection")
    return mode / norm, rigid_rank


def _coordinates(value: Any) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(item) for item in row) for row in value)


def run_dimer_preflight(
    backend: CalculatorBackend,
    system: MolecularSystem,
    initial_mode: Sequence[Sequence[float]],
    *,
    budget_token: CalculatorCallToken,
    step_angstrom: float = 1.0e-3,
    config: ASEDimerConfig = ASEDimerConfig(),
    require_durable_token: bool = True,
) -> DimerPreflightResult:
    """Measure projected force and directional curvature with at most 3 calls.

    One E/F call evaluates the declared geometry. Two additional E/F calls
    form a central finite-difference Hessian-vector product along the declared
    internal mode. The numerical result is deliberately separate from the
    caller's pass/fail thresholds.
    """

    from xtbflow.physics.curvature import finite_difference_hvp

    if not isinstance(budget_token, CalculatorCallToken):
        raise BudgetTokenRequired("dimer preflight requires a calculator token")
    if require_durable_token and not budget_token.durable:
        raise ValueError("production dimer preflight requires a durable token")
    if (
        not isinstance(step_angstrom, (int, float))
        or not math.isfinite(float(step_angstrom))
        or float(step_angstrom) <= 0.0
    ):
        raise ValueError("step_angstrom must be finite and positive")

    mode, rigid_body_rank = _normalized_mode(
        initial_mode, system.coordinates, config
    )
    calls_before = budget_token.consumed_calls
    normalized_mode = _coordinates(mode)

    try:
        base = backend.evaluate(
            system, operation="energy_forces", budget_token=budget_token
        )
    except Exception as error:
        return DimerPreflightResult(
            status="failure",
            normalized_mode=normalized_mode,
            energy_hartree=None,
            projected_forces_hartree_per_angstrom=None,
            projected_force_norm_hartree_per_angstrom=None,
            hessian_vector_product_hartree_per_angstrom2=None,
            mode_curvature_hartree_per_angstrom2=None,
            mode_curvature_eV_per_angstrom2=None,
            rigid_body_rank=rigid_body_rank,
            calculator_calls=budget_token.consumed_calls - calls_before,
            error=f"{type(error).__name__}: {error}",
        )

    if base.status != "success" or base.energy is None or base.forces is None:
        detail = base.error_message or base.error_category or (
            "base energy/force evaluation did not return complete evidence"
        )
        return DimerPreflightResult(
            status="failure",
            normalized_mode=normalized_mode,
            energy_hartree=base.energy,
            projected_forces_hartree_per_angstrom=None,
            projected_force_norm_hartree_per_angstrom=None,
            hessian_vector_product_hartree_per_angstrom2=None,
            mode_curvature_hartree_per_angstrom2=None,
            mode_curvature_eV_per_angstrom2=None,
            rigid_body_rank=rigid_body_rank,
            calculator_calls=budget_token.consumed_calls - calls_before,
            error=detail,
        )

    if config.remove_rigid_body_modes:
        projected_forces, projected_rank = project_rigid_body_components(
            system.coordinates,
            base.forces,
            tolerance=config.rigid_body_tolerance,
        )
        if projected_rank != rigid_body_rank:
            raise RuntimeError("inconsistent rigid-body rank during dimer preflight")
    else:
        projected_forces = np.asarray(base.forces, dtype=float)
    projected_force_norm = float(np.linalg.norm(projected_forces))

    try:
        hvp_result = finite_difference_hvp(
            backend,
            system,
            mode,
            step=float(step_angstrom),
            budget_token=budget_token,
        )
    except Exception as error:
        return DimerPreflightResult(
            status="failure",
            normalized_mode=normalized_mode,
            energy_hartree=float(base.energy),
            projected_forces_hartree_per_angstrom=_coordinates(projected_forces),
            projected_force_norm_hartree_per_angstrom=projected_force_norm,
            hessian_vector_product_hartree_per_angstrom2=None,
            mode_curvature_hartree_per_angstrom2=None,
            mode_curvature_eV_per_angstrom2=None,
            rigid_body_rank=rigid_body_rank,
            calculator_calls=budget_token.consumed_calls - calls_before,
            error=f"{type(error).__name__}: {error}",
        )

    if (
        hvp_result.status != "success"
        or hvp_result.hessian_vector_product is None
    ):
        return DimerPreflightResult(
            status="failure",
            normalized_mode=normalized_mode,
            energy_hartree=float(base.energy),
            projected_forces_hartree_per_angstrom=_coordinates(projected_forces),
            projected_force_norm_hartree_per_angstrom=projected_force_norm,
            hessian_vector_product_hartree_per_angstrom2=None,
            mode_curvature_hartree_per_angstrom2=None,
            mode_curvature_eV_per_angstrom2=None,
            rigid_body_rank=rigid_body_rank,
            calculator_calls=budget_token.consumed_calls - calls_before,
            error=hvp_result.error or "directional curvature evaluation failed",
        )

    hessian_vector = np.asarray(
        hvp_result.hessian_vector_product, dtype=float
    )
    curvature_hartree = float(np.sum(mode * hessian_vector))
    return DimerPreflightResult(
        status="success",
        normalized_mode=normalized_mode,
        energy_hartree=float(base.energy),
        projected_forces_hartree_per_angstrom=_coordinates(projected_forces),
        projected_force_norm_hartree_per_angstrom=projected_force_norm,
        hessian_vector_product_hartree_per_angstrom2=_coordinates(
            hessian_vector
        ),
        mode_curvature_hartree_per_angstrom2=curvature_hartree,
        mode_curvature_eV_per_angstrom2=curvature_hartree * Hartree,
        rigid_body_rank=rigid_body_rank,
        calculator_calls=budget_token.consumed_calls - calls_before,
        error=None,
    )


def _artifact_index(root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    )


def _append_trajectory_frame(
    path: Path,
    dimer_atoms: Any,
    *,
    step: int,
    calculator_calls: int,
) -> None:
    energy = getattr(dimer_atoms, "energy0", None)
    forces = getattr(dimer_atoms, "forces0", None)
    modes = getattr(dimer_atoms, "eigenmodes", None)
    curvatures = getattr(dimer_atoms, "curvatures", None)
    payload: dict[str, Any] = {
        "step": int(step),
        "calculator_calls": int(calculator_calls),
        "coordinates_angstrom": [
            list(row) for row in _coordinates(dimer_atoms.get_positions())
        ],
        "energy_hartree": (
            float(energy) / Hartree
            if energy is not None and math.isfinite(float(energy))
            else None
        ),
        "forces_hartree_per_angstrom": (
            [list(row) for row in _coordinates(np.asarray(forces, dtype=float) / Hartree)]
            if forces is not None
            else None
        ),
        "curvature_eV_per_angstrom2": (
            float(curvatures[0]) if curvatures else None
        ),
        "eigenmode": (
            [list(row) for row in _coordinates(modes[0])] if modes else None
        ),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def _failure_result(
    *,
    atoms: Any,
    calls: int,
    steps: int,
    rigid_body_rank: int,
    root: Path,
    config: ASEDimerConfig,
    error: Exception,
) -> ASEDimerResult:
    eigenmode = None
    curvature = None
    modes = getattr(atoms, "eigenmodes", None)
    curvatures = getattr(atoms, "curvatures", None)
    if modes:
        candidate = np.asarray(modes[0], dtype=float)
        if candidate.ndim == 2 and candidate.shape[1:] == (3,):
            eigenmode = _coordinates(candidate)
    if curvatures:
        candidate_curvature = float(curvatures[0])
        if math.isfinite(candidate_curvature):
            curvature = candidate_curvature
    return ASEDimerResult(
        status="failure",
        converged=False,
        coordinates_angstrom=_coordinates(atoms.get_positions()),
        energy_hartree=None,
        forces_hartree_per_angstrom=None,
        gradient_norm_hartree_per_angstrom=None,
        eigenmode=eigenmode,
        curvature_eV_per_angstrom2=curvature,
        rigid_body_rank=rigid_body_rank,
        optimizer_steps=steps,
        calculator_calls=calls,
        artifact_files=_artifact_index(root),
        config_identity=config.identity,
        error=f"{type(error).__name__}: {error}",
    )


def run_ase_dimer_search(
    backend: CalculatorBackend,
    system: MolecularSystem,
    initial_mode: Sequence[Sequence[float]],
    *,
    budget_token: CalculatorCallToken,
    artifact_dir: str | Path,
    config: ASEDimerConfig = ASEDimerConfig(),
    require_durable_token: bool = True,
    environment: Mapping[str, Any] | None = None,
) -> ASEDimerResult:
    """Run ASE's minimum-mode dimer search with hard per-call accounting."""

    if not isinstance(budget_token, CalculatorCallToken):
        raise BudgetTokenRequired("ASE dimer search requires a calculator token")
    if require_durable_token and not budget_token.durable:
        raise ValueError("production ASE dimer search requires a durable token")
    mode, rigid_body_rank = _normalized_mode(
        initial_mode, system.coordinates, config
    )
    root = Path(artifact_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    atoms = Atoms(system.symbols, positions=np.asarray(system.coordinates, dtype=float))
    if config.remove_rigid_body_modes:
        atoms.set_constraint(RigidBodyProjection(config.rigid_body_tolerance))
    atoms.calc = MeteredASECalculator(
        backend,
        charge=system.charge,
        multiplicity=system.multiplicity,
        budget_token=budget_token,
        environment=environment or system.environment,
        system_id=system.system_id or "ase-dimer-candidate",
    )
    calls_before = budget_token.consumed_calls
    optimizer_steps = 0
    optimizer: Any = None
    dimer_atoms: Any = None
    try:
        with DimerControl(
            initial_eigenmode_method="displacement",
            displacement_method="vector",
            dimer_separation=config.dimer_separation_angstrom,
            max_num_rot=config.max_num_rot,
            maximum_translation=config.maximum_translation_angstrom,
            trial_trans_step=config.trial_translation_step_angstrom,
            logfile=str(root / "dimer.log"),
            eigenmode_logfile=str(root / "eigenmode.log"),
        ) as control:
            dimer_atoms = MinModeAtoms(
                atoms,
                control=control,
                eigenmodes=[mode.copy()],
                random_seed=config.random_seed,
            )
            trajectory_path = root / "trajectory.jsonl"
            if trajectory_path.exists():
                raise FileExistsError(f"trajectory already exists: {trajectory_path}")
            with MinModeTranslate(
                dimer_atoms,
                trajectory=None,
                logfile=str(root / "optimizer.log"),
            ) as optimizer:
                optimizer.attach(
                    lambda: _append_trajectory_frame(
                        trajectory_path,
                        dimer_atoms,
                        step=int(optimizer.nsteps),
                        calculator_calls=budget_token.consumed_calls - calls_before,
                    ),
                    interval=1,
                )
                converged = bool(
                    optimizer.run(
                        fmax=config.fmax_eV_per_angstrom,
                        steps=config.max_steps,
                    )
                )
                optimizer_steps = int(optimizer.nsteps)
            real_forces = np.asarray(dimer_atoms.get_forces(real=True), dtype=float)
            energy_hartree = float(dimer_atoms.get_potential_energy()) / Hartree
            forces_hartree = real_forces / Hartree
            gradient_norm = float(np.linalg.norm(forces_hartree))
            final_mode = np.asarray(dimer_atoms.get_eigenmode(), dtype=float)
            curvature = float(dimer_atoms.get_curvature())
    except Exception as error:
        current = dimer_atoms if dimer_atoms is not None else atoms
        return _failure_result(
            atoms=current,
            calls=budget_token.consumed_calls - calls_before,
            steps=int(getattr(optimizer, "nsteps", optimizer_steps)),
            rigid_body_rank=rigid_body_rank,
            root=root,
            config=config,
            error=error,
        )

    return ASEDimerResult(
        status="success" if converged else "not_converged",
        converged=converged,
        coordinates_angstrom=_coordinates(dimer_atoms.get_positions()),
        energy_hartree=energy_hartree,
        forces_hartree_per_angstrom=_coordinates(forces_hartree),
        gradient_norm_hartree_per_angstrom=gradient_norm,
        eigenmode=_coordinates(final_mode),
        curvature_eV_per_angstrom2=curvature,
        rigid_body_rank=rigid_body_rank,
        optimizer_steps=optimizer_steps,
        calculator_calls=budget_token.consumed_calls - calls_before,
        artifact_files=_artifact_index(root),
        config_identity=config.identity,
        error=None if converged else "dimer search did not converge within max_steps",
    )
