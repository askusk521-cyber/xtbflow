"""Optional ASE adapter for metered xtbflow energy/force backends."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from xtbflow.calculators import (
    CalculatorBackend,
    CalculatorError,
    MolecularSystem,
)
from xtbflow.runtime import CalculatorCallToken

try:
    from ase.calculators.calculator import Calculator, all_changes
    from ase.units import Hartree
except ImportError as exc:  # pragma: no cover - exercised only without ASE.
    raise ImportError(
        "xtbflow.validation.ase_calculator requires the optional ASE dependency"
    ) from exc


class MeteredASECalculator(Calculator):
    """Expose one xtbflow backend to ASE while preserving call accounting.

    The xtbflow public contract uses Hartree and Hartree/Angstrom. ASE uses eV
    and eV/Angstrom, so conversion occurs once at this adapter boundary.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        backend: CalculatorBackend,
        *,
        charge: int,
        multiplicity: int,
        budget_token: CalculatorCallToken | None,
        environment: Mapping[str, Any] | None = None,
        system_id: str = "ase-reference-driver",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if type(charge) is not int:
            raise ValueError("charge must be an explicit integer")
        if type(multiplicity) is not int or multiplicity < 1:
            raise ValueError("multiplicity must be a positive integer")
        self.backend = backend
        self.charge = charge
        self.multiplicity = multiplicity
        self.budget_token = budget_token
        self.environment = dict(environment or {})
        self.system_id = system_id
        self.last_xtbflow_result = None

    def molecular_system(self, atoms: Any) -> MolecularSystem:
        return MolecularSystem(
            tuple(atoms.get_chemical_symbols()),
            tuple(
                tuple(float(value) for value in row)
                for row in atoms.get_positions()
            ),
            self.charge,
            self.multiplicity,
            self.environment,
            self.system_id,
        )

    def calculate(
        self,
        atoms: Any = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        system = self.molecular_system(self.atoms)
        result = self.backend.evaluate(
            system,
            operation="energy_forces",
            budget_token=self.budget_token,
        )
        if result.status != "success":
            raise CalculatorError(
                result.error_message or f"backend returned {result.status}"
            )
        if result.energy is None or result.forces is None:
            raise CalculatorError("backend did not return both energy and forces")
        self.last_xtbflow_result = result
        self.results = {
            "energy": float(result.energy) * Hartree,
            "forces": np.asarray(result.forces, dtype=float) * Hartree,
        }
