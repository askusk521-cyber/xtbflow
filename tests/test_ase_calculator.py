from __future__ import annotations

import pytest

ase = pytest.importorskip("ase")
from ase import Atoms
from ase.units import Hartree

from xtbflow.calculators import CalculatorProtocol, XTBloomAdapter
from xtbflow.runtime import RunLedger, StageBudget
from xtbflow.validation.ase_calculator import MeteredASECalculator


def protocol():
    return CalculatorProtocol(
        protocol_id="quadratic-ase-v1",
        calculator="xtbloom",
        method="quadratic-fixture",
        parameters={"fixture": True},
    )


def test_ase_adapter_converts_units_and_consumes_one_token_per_geometry():
    calls = []

    def evaluator(system, operation):
        calls.append((system.input_hash, system.charge, system.multiplicity))
        energy = sum(value * value for row in system.coordinates for value in row)
        forces = tuple(
            tuple(-2.0 * value for value in row) for row in system.coordinates
        )
        return {
            "charge": system.charge,
            "multiplicity": system.multiplicity,
            "converged": True,
            "energy": energy,
            "forces": forces,
        }

    ledger = RunLedger(StageBudget("ase-fixture", max_calculator_calls=2))
    token = ledger.issue_calculator_token("ase-fixture", 2)
    backend = XTBloomAdapter(
        protocol=protocol(), evaluator=evaluator, require_budget_token=True
    )
    atoms = Atoms("H", positions=[[1.0, 0.0, 0.0]])
    atoms.calc = MeteredASECalculator(
        backend,
        charge=1,
        multiplicity=2,
        budget_token=token,
        environment={"periodic": False},
    )

    assert atoms.get_potential_energy() == pytest.approx(Hartree)
    assert atoms.get_forces()[0, 0] == pytest.approx(-2.0 * Hartree)
    assert token.consumed_calls == 1
    assert calls[0][1:] == (1, 2)

    # ASE reuses the cached result for an unchanged geometry.
    assert atoms.get_potential_energy() == pytest.approx(Hartree)
    assert token.consumed_calls == 1

    atoms.positions[0, 0] = 1.1
    assert atoms.get_forces()[0, 0] == pytest.approx(-2.2 * Hartree)
    assert token.consumed_calls == 2
    assert len(calls) == 2
