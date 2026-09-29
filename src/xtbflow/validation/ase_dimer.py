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
    driver_version: str = "ase-dimer-v1"

    def __post_init__(self) -> None:
        for name in (
            "fmax_eV_per_angstrom",
            "dimer_separation_angstrom",
            "maximum_translation_angstrom",
            "trial_translation_step_angstrom",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if type(self.max_num_rot) is not int or self.max_num_rot < 1:
            raise ValueError("max_num_rot must be a positive integer")
        if type(self.random_seed) is not int:
            raise ValueError("random_seed must be an integer")
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
            "artifact_files": self.artifact_files,
            "config_identity": self.config_identity,
        }

def _normalized_mode(
    value: Sequence[Sequence[float]], atom_count: int
) -> np.ndarray:
    mode = np.asarray(value, dtype=float)
    if mode.shape != (atom_count, 3):
        raise ValueError("initial_mode must have shape [N,3]")
    if not np.isfinite(mode).all():
        raise ValueError("initial_mode must contain finite values")
    norm = float(np.linalg.norm(mode))
    if norm == 0.0:
        raise ValueError("initial_mode must be nonzero")
    return mode / norm


def _coordinates(value: Any) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(item) for item in row) for row in value)


def _artifact_index(root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    )


def _failure_result(
    *,
    atoms: Any,
    calls: int,
    steps: int,
    root: Path,
    config: ASEDimerConfig,
    error: Exception,
) -> ASEDimerResult:
    return ASEDimerResult(
        status="failure",
        converged=False,
        coordinates_angstrom=_coordinates(atoms.get_positions()),
        energy_hartree=None,
        forces_hartree_per_angstrom=None,
        gradient_norm_hartree_per_angstrom=None,
        eigenmode=None,
        curvature_eV_per_angstrom2=None,
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
    mode = _normalized_mode(initial_mode, len(system.symbols))
    root = Path(artifact_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    atoms = Atoms(system.symbols, positions=np.asarray(system.coordinates, dtype=float))
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
            with MinModeTranslate(
                dimer_atoms,
                trajectory=str(root / "trajectory.traj"),
                logfile=str(root / "optimizer.log"),
            ) as optimizer:
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
            steps=optimizer_steps,
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
        optimizer_steps=optimizer_steps,
        calculator_calls=budget_token.consumed_calls - calls_before,
        artifact_files=_artifact_index(root),
        config_identity=config.identity,
        error=None if converged else "dimer search did not converge within max_steps",
    )
