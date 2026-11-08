"""Deterministic calculator used by the bounded software MVP.

The mock implements the same :class:`CalculatorBackend` protocol as the
optional xTB adapter.  It is a harmonic proxy around one frozen input
geometry; its output is useful for exercising schemas, QC accounting and
replay, but it is never a chemistry result.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

from xtbflow.runtime.ledger import BudgetTokenRequired, CalculatorCallToken

from .base import (
    CalculatorBackend,
    CalculatorCapabilities,
    CalculatorProtocol,
    CalculationResult,
    MolecularSystem,
)


def _source_hash() -> str:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return "unavailable"


class MockCalculator(CalculatorBackend):
    """A finite, deterministic harmonic proxy with the shared calculator API."""

    def __init__(
        self,
        reference_coordinates: Sequence[Sequence[float]],
        *,
        force_constant: float = 1.0,
        protocol: CalculatorProtocol | None = None,
        require_budget_token: bool = False,
    ) -> None:
        if force_constant <= 0:
            raise ValueError("force_constant must be positive")
        self.reference_coordinates = tuple(tuple(float(value) for value in row) for row in reference_coordinates)
        if not self.reference_coordinates or any(len(row) != 3 for row in self.reference_coordinates):
            raise ValueError("reference_coordinates must be a nonempty N x 3 sequence")
        self.force_constant = float(force_constant)
        self.protocol = protocol or CalculatorProtocol(
            protocol_id="mvp-mock-harmonic-v1",
            calculator="mvp_mock",
            method="deterministic_harmonic",
            parameters={
                "force_constant": self.force_constant,
                "reference_coordinates_angstrom": [list(row) for row in self.reference_coordinates],
            },
        )
        if self.protocol.calculator != "mvp_mock":
            raise ValueError("MockCalculator requires protocol.calculator='mvp_mock'")
        self.require_budget_token = bool(require_budget_token)
        self.build_hash = _source_hash()

    @property
    def capabilities(self) -> CalculatorCapabilities:
        return CalculatorCapabilities(
            calculator="mvp_mock",
            status="unknown",
            detail="deterministic harmonic proxy; software-demo evidence only",
            build_hash=self.build_hash,
            operations=("energy", "forces", "energy_forces"),
            backends=(self.protocol.backend,),
            qualification="callable",
        )

    def evaluate(
        self,
        system: MolecularSystem,
        *,
        operation: str = "energy_forces",
        budget_token: CalculatorCallToken | None = None,
    ) -> CalculationResult:
        if operation not in {"energy", "forces", "energy_forces"}:
            raise ValueError(f"unsupported operation: {operation}")
        if len(system.symbols) != len(self.reference_coordinates):
            raise ValueError("mock reference geometry does not match the input atom count")
        if self.require_budget_token and budget_token is None:
            raise BudgetTokenRequired("mock production calls require a calculator budget token")
        if budget_token is not None:
            budget_token.consume()

        displacements: tuple[tuple[float, float, float], ...] = tuple(
            tuple(
                float(coordinate) - float(reference)
                for coordinate, reference in zip(row, reference_row)
            )
            for row, reference_row in zip(system.coordinates, self.reference_coordinates)
        )
        forces: tuple[tuple[float, float, float], ...] = tuple(
            tuple(-self.force_constant * value for value in row) for row in displacements
        )
        energy = 0.5 * self.force_constant * sum(value * value for row in displacements for value in row)
        result = CalculationResult(
            calculator=self.protocol.calculator,
            protocol_id=self.protocol.protocol_id,
            input_hash=system.input_hash,
            charge=system.charge,
            multiplicity=system.multiplicity,
            operation=operation,
            energy=energy if operation in {"energy", "energy_forces"} else None,
            forces=forces if operation in {"forces", "energy_forces"} else None,
            calculator_build_hash=self.build_hash,
            metadata={
                "implementation": "xtbflow.mvp.MockCalculator",
                "evidence_level": "software_demo",
                "proxy_metric": "harmonic_energy_and_force_norm",
                "force_constant": self.force_constant,
            },
        )
        return result
