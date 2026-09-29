"""ASE minimum relaxation and small-system path validation primitives."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from xtbflow.calculators import CalculatorBackend, MolecularSystem
from xtbflow.runtime import BudgetTokenRequired, CalculatorCallToken

from .ase_calculator import MeteredASECalculator
from .ase_dimer import RigidBodyProjection, rigid_body_basis

try:
    from ase import Atoms
    from ase.optimize import BFGS
    from ase.units import Hartree
except ImportError as exc:  # pragma: no cover - optional dependency boundary.
    raise ImportError(
        "xtbflow.validation.ase_path requires the optional ASE dependency"
    ) from exc


@dataclass(frozen=True)
class ASERelaxConfig:
    """Frozen controls for one metered local-minimum relaxation."""

    fmax_eV_per_angstrom: float = 0.01
    max_steps: int = 100
    maximum_step_angstrom: float = 0.05
    remove_rigid_body_modes: bool = True
    rigid_body_tolerance: float = 1.0e-10
    driver_version: str = "ase-bfgs-minimum-v1"

    def __post_init__(self) -> None:
        for name in (
            "fmax_eV_per_angstrom",
            "maximum_step_angstrom",
            "rigid_body_tolerance",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
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
class ASERelaxResult:
    """One local-minimum search outcome without path qualification."""

    status: str
    converged: bool
    coordinates_angstrom: tuple[tuple[float, float, float], ...]
    energy_hartree: float | None
    forces_hartree_per_angstrom: tuple[tuple[float, float, float], ...] | None
    gradient_norm_hartree_per_angstrom: float | None
    rigid_body_rank: int
    optimizer_steps: int
    calculator_calls: int
    artifact_files: tuple[str, ...]
    config_identity: str
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"success", "not_converged", "failure"}:
            raise ValueError("unsupported ASE relaxation status")
        if type(self.converged) is not bool:
            raise ValueError("converged must be boolean")
        for name in ("rigid_body_rank", "optimizer_steps", "calculator_calls"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(self.config_identity, str) or not self.config_identity:
            raise ValueError("config_identity is required")
        if self.status == "success" and self.error is not None:
            raise ValueError("successful relaxation cannot carry an error")


def internal_cartesian_basis(
    coordinates: Sequence[Sequence[float]],
    *,
    tolerance: float = 1.0e-10,
) -> tuple[np.ndarray, int]:
    """Return an orthonormal Cartesian complement to rigid translations/rotations."""

    positions = np.asarray(coordinates, dtype=float)
    if positions.ndim != 2 or positions.shape[1:] != (3,) or len(positions) < 1:
        raise ValueError("coordinates must have shape [N,3] with N >= 1")
    if not np.isfinite(positions).all():
        raise ValueError("coordinates must contain finite values")
    rigid = rigid_body_basis(positions, tolerance=tolerance)
    complete, _ = np.linalg.qr(rigid, mode="complete")
    rank = int(rigid.shape[1])
    internal_flat = complete[:, rank:].T
    internal = internal_flat.reshape(internal_flat.shape[0], len(positions), 3)
    if internal.size:
        gram = internal_flat @ internal_flat.T
        if not np.allclose(gram, np.eye(len(internal_flat)), atol=1.0e-10, rtol=0.0):
            raise RuntimeError("internal Cartesian basis is not orthonormal")
        if not np.allclose(internal_flat @ rigid, 0.0, atol=1.0e-10, rtol=0.0):
            raise RuntimeError("internal Cartesian basis is not orthogonal to rigid modes")
    return internal, rank


def reconstruct_projected_mode(
    basis: Sequence[Sequence[Sequence[float]]],
    eigenvectors: Sequence[Sequence[float]],
    index: int,
) -> np.ndarray:
    """Lift one projected-Hessian eigenvector back to normalized Cartesian space."""

    directions = np.asarray(basis, dtype=float)
    vectors = np.asarray(eigenvectors, dtype=float)
    if directions.ndim != 3 or directions.shape[2] != 3:
        raise ValueError("basis must have shape [K,N,3]")
    if vectors.shape != (directions.shape[0], directions.shape[0]):
        raise ValueError("eigenvectors must have shape [K,K]")
    if type(index) is not int or not 0 <= index < directions.shape[0]:
        raise ValueError("index is outside the projected basis")
    if not np.isfinite(directions).all() or not np.isfinite(vectors).all():
        raise ValueError("basis and eigenvectors must be finite")
    coefficients = vectors[:, index]
    mode = np.tensordot(coefficients, directions, axes=(0, 0))
    norm = float(np.linalg.norm(mode))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("reconstructed mode is zero or nonfinite")
    return mode / norm


def signed_atom_plane_distance(
    coordinates: Sequence[Sequence[float]],
    *,
    atom_index: int,
    plane_indices: Sequence[int],
) -> float:
    """Return signed distance from one atom to an ordered three-atom plane."""

    positions = np.asarray(coordinates, dtype=float)
    if positions.ndim != 2 or positions.shape[1:] != (3,) or not np.isfinite(positions).all():
        raise ValueError("coordinates must be finite [N,3]")
    if type(atom_index) is not int or not 0 <= atom_index < len(positions):
        raise ValueError("atom_index is outside the coordinate array")
    indices = tuple(int(value) for value in plane_indices)
    if len(indices) != 3 or len(set(indices)) != 3:
        raise ValueError("plane_indices must contain three distinct atom indices")
    if atom_index in indices or any(value < 0 or value >= len(positions) for value in indices):
        raise ValueError("plane indices must be valid and exclude atom_index")
    first, second, third = (positions[value] for value in indices)
    normal = np.cross(second - first, third - first)
    norm = float(np.linalg.norm(normal))
    if norm <= 1.0e-12:
        raise ValueError("plane atoms are collinear")
    normal /= norm
    return float(np.dot(positions[atom_index] - first, normal))


def _coordinates(value: Any) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(item) for item in row) for row in value)


def _artifact_index(root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        )
    )


def _append_relaxation_frame(
    path: Path,
    atoms: Any,
    *,
    step: int,
    calculator_calls: int,
) -> None:
    energy = float(atoms.get_potential_energy()) / Hartree
    forces = np.asarray(atoms.get_forces(apply_constraint=False), dtype=float) / Hartree
    payload = {
        "step": int(step),
        "calculator_calls": int(calculator_calls),
        "coordinates_angstrom": [list(row) for row in _coordinates(atoms.get_positions())],
        "energy_hartree": energy,
        "forces_hartree_per_angstrom": [list(row) for row in _coordinates(forces)],
        "gradient_norm_hartree_per_angstrom": float(np.linalg.norm(forces)),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def run_ase_minimum_relaxation(
    backend: CalculatorBackend,
    system: MolecularSystem,
    *,
    budget_token: CalculatorCallToken,
    artifact_dir: str | Path,
    config: ASERelaxConfig = ASERelaxConfig(),
    require_durable_token: bool = True,
    environment: Mapping[str, Any] | None = None,
) -> ASERelaxResult:
    """Run a bounded BFGS minimum search with exact backend-call accounting."""

    if not isinstance(budget_token, CalculatorCallToken):
        raise BudgetTokenRequired("ASE minimum relaxation requires a calculator token")
    if require_durable_token and not budget_token.durable:
        raise ValueError("production ASE relaxation requires a durable token")
    root = Path(artifact_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("ASE relaxation artifact directory must be empty")

    atoms = Atoms(system.symbols, positions=np.asarray(system.coordinates, dtype=float))
    rigid_rank = 0
    if config.remove_rigid_body_modes:
        constraint = RigidBodyProjection(config.rigid_body_tolerance)
        atoms.set_constraint(constraint)
        rigid_rank = constraint.get_removed_dof(atoms)
    atoms.calc = MeteredASECalculator(
        backend,
        charge=system.charge,
        multiplicity=system.multiplicity,
        budget_token=budget_token,
        environment=environment or system.environment,
        system_id=system.system_id or "ase-minimum-candidate",
    )
    trajectory_path = root / "trajectory.jsonl"
    calls_before = budget_token.consumed_calls
    optimizer_steps = 0
    optimizer: Any = None
    try:
        optimizer = BFGS(
            atoms,
            restart=root / "bfgs.restart.json",
            logfile=root / "optimizer.log",
            trajectory=None,
            maxstep=config.maximum_step_angstrom,
        )
        optimizer.attach(
            lambda: _append_relaxation_frame(
                trajectory_path,
                atoms,
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
        energy_hartree = float(atoms.get_potential_energy()) / Hartree
        physical_forces = (
            np.asarray(atoms.get_forces(apply_constraint=False), dtype=float) / Hartree
        )
        gradient_norm = float(np.linalg.norm(physical_forces))
    except Exception as error:
        return ASERelaxResult(
            status="failure",
            converged=False,
            coordinates_angstrom=_coordinates(atoms.get_positions()),
            energy_hartree=None,
            forces_hartree_per_angstrom=None,
            gradient_norm_hartree_per_angstrom=None,
            rigid_body_rank=rigid_rank,
            optimizer_steps=int(getattr(optimizer, "nsteps", optimizer_steps)),
            calculator_calls=budget_token.consumed_calls - calls_before,
            artifact_files=_artifact_index(root),
            config_identity=config.identity,
            error=f"{type(error).__name__}: {error}",
        )

    return ASERelaxResult(
        status="success" if converged else "not_converged",
        converged=converged,
        coordinates_angstrom=_coordinates(atoms.get_positions()),
        energy_hartree=energy_hartree,
        forces_hartree_per_angstrom=_coordinates(physical_forces),
        gradient_norm_hartree_per_angstrom=gradient_norm,
        rigid_body_rank=rigid_rank,
        optimizer_steps=optimizer_steps,
        calculator_calls=budget_token.consumed_calls - calls_before,
        artifact_files=_artifact_index(root),
        config_identity=config.identity,
        error=None if converged else "minimum relaxation did not converge within max_steps",
    )
