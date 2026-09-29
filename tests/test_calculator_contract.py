from __future__ import annotations

import math
import sys
import types

import numpy as np
import pytest

from xtbflow.calculators import (
    CalculatorCapabilities,
    CalculatorProtocol,
    CalculatorProtocolError,
    CalculatorUnavailable,
    MolecularSystem,
    XTBloomAdapter,
    XTBOracleAdapter,
    finite_difference_forces,
)
from xtbflow.runtime import BudgetExceeded, RunLedger, StageBudget


def system(*, system_id: str = "water") -> MolecularSystem:
    return MolecularSystem(
        symbols=("O", "H", "H"),
        coordinates=((0.0, 0.0, 0.0), (0.95, 0.0, 0.0), (-0.2, 0.9, 0.0)),
        charge=0,
        multiplicity=1,
        environment={"solvent": "explicit_water_cluster"},
        system_id=system_id,
    )


def protocol(calculator: str = "xtbloom") -> CalculatorProtocol:
    return CalculatorProtocol(
        protocol_id="mock-gfn2-v1",
        calculator=calculator,
        method="GFN2-xTB",
        parameters={"scc_tolerance": 1e-8, "electronic_temperature": 300.0},
    )


def quadratic(item: MolecularSystem, operation: str):
    energy = sum(value * value for row in item.coordinates for value in row)
    forces = [[-2.0 * value for value in row] for row in item.coordinates]
    result = {"charge": item.charge, "multiplicity": item.multiplicity, "converged": True}
    if operation in {"energy", "energy_forces"}:
        result["energy"] = energy
    if operation in {"forces", "energy_forces"}:
        result["forces"] = forces
    return result


def test_protocol_and_input_hash_include_explicit_state_and_backend():
    first = protocol()
    second = CalculatorProtocol(**{**first.__dict__, "backend": "cuda"})
    assert first.identity != second.identity
    # Labels are audit metadata; the cache identity is molecular input only.
    assert system().input_hash == system(system_id="different-label").input_hash


def test_injected_evaluator_validates_energy_forces_and_finite_difference():
    adapter = XTBloomAdapter(protocol=protocol(), evaluator=quadratic, version="test", build_hash="fixture")
    result = adapter.evaluate(system(), operation="energy_forces")
    assert result.status == "success"
    assert result.energy == pytest.approx(sum(value * value for row in system().coordinates for value in row))
    assert result.forces[0] == (0.0, 0.0, 0.0)
    estimated = finite_difference_forces(adapter, system(), step=1e-5)
    assert estimated[1][0] == pytest.approx(-1.9, rel=1e-5, abs=1e-5)
    assert estimated[1][1] == pytest.approx(0.0, abs=1e-8)


def test_batch_isolates_backend_failures_and_preserves_protocol_identity():
    def evaluator(item: MolecularSystem, operation: str):
        if item.system_id == "bad":
            raise RuntimeError("SCC did not converge")
        return quadratic(item, operation)

    adapter = XTBOracleAdapter(protocol=protocol("xtb_oracle"), evaluator=evaluator, implementation="tblite")
    results = adapter.evaluate_batch((system(system_id="good"), system(system_id="bad")), operation="energy")
    assert [item.status for item in results] == ["success", "failure"]
    assert results[1].error_category == "execution"
    assert adapter.capabilities.status == "unknown"
    assert adapter.capabilities.qualification == "callable"


def test_tblite_native_adapter_converts_public_units_at_the_boundary(monkeypatch):
    captured = {}

    class FakeResult:
        def get(self, name):
            if name == "energy":
                return -1.25
            if name == "gradient":
                return np.asarray([[0.25, -0.5, 0.0]], dtype=np.float64)
            raise AssertionError(name)

    class FakeCalculator:
        def __init__(self, method, numbers, positions, **kwargs):
            captured.update(method=method, numbers=numbers, positions=positions, kwargs=kwargs)

        def set(self, name, value):
            captured.setdefault("settings", {})[name] = value

        def singlepoint(self):
            return FakeResult()

    tblite = types.ModuleType("tblite")
    interface = types.ModuleType("tblite.interface")
    interface.Calculator = FakeCalculator
    tblite.interface = interface
    monkeypatch.setitem(sys.modules, "tblite", tblite)
    monkeypatch.setitem(sys.modules, "tblite.interface", interface)

    native_protocol = CalculatorProtocol(
        protocol_id="tblite-gfn2-native-bohr-v1",
        calculator="xtb_oracle",
        method="GFN2-xTB",
        parameters={"accuracy": 1.0, "max_iterations": 100, "electronic_temperature": 300.0, "solvent": None},
    )
    item = MolecularSystem(("H",), ((1.0, 0.0, 0.0),), 0, 3)
    adapter = XTBOracleAdapter(protocol=native_protocol, implementation="tblite")
    result = adapter.evaluate(item)

    assert captured["method"] == "GFN2-xTB"
    assert captured["numbers"].tolist() == [1]
    assert captured["positions"][0, 0] == pytest.approx(1.0 / 0.529177210903)
    assert captured["kwargs"] == {"charge": 0, "uhf": 2}
    assert result.forces[0][0] == pytest.approx(-0.25 / 0.529177210903)
    assert result.metadata["native_coordinate_unit"] == "bohr"
    assert result.metadata["native_gradient_unit"] == "hartree/bohr"
    assert result.metadata["normalized_force_unit"] == "hartree/angstrom"
    assert result.metadata["adapter_revision"] == "xtb-oracle-native-units-v1"

def test_calculator_token_is_consumed_at_adapter_boundary():
    ledger = RunLedger(StageBudget("P1", max_calculator_calls=2))
    token = ledger.issue_calculator_token("adapter", 2)
    calls = []

    def evaluator(item, operation):
        calls.append(operation)
        return quadratic(item, operation)

    adapter = XTBloomAdapter(protocol=protocol(), evaluator=evaluator)
    adapter.evaluate(system(), operation="energy", budget_token=token)
    assert calls == ["energy"]
    assert token.consumed_calls == 1
    adapter.evaluate(system(), operation="energy", budget_token=token)
    assert calls == ["energy", "energy"]
    assert token.consumed_calls == 2
    with pytest.raises(BudgetExceeded):
        adapter.evaluate(system(), operation="energy", budget_token=token)
    assert token.consumed_calls == 2

    adapter = XTBloomAdapter()
    with pytest.raises(CalculatorUnavailable):
        adapter.evaluate(system(), operation="energy")


def test_production_adapter_blocks_the_fourth_call_before_evaluator():
    ledger = RunLedger(StageBudget("P1", max_calculator_calls=3))
    token = ledger.issue_calculator_token("xtbloom-production", 3)
    calls = []

    def evaluator(item, operation):
        calls.append((item.input_hash, operation))
        return quadratic(item, operation)

    adapter = XTBloomAdapter(protocol=protocol(), evaluator=evaluator, require_budget_token=True)
    for _ in range(3):
        adapter.evaluate(system(), operation="energy", budget_token=token)
    with pytest.raises(BudgetExceeded):
        adapter.evaluate(system(), operation="energy", budget_token=token)
    assert len(calls) == 3
    assert token.consumed_calls == 3


def test_missing_or_nonfinite_contract_fields_are_rejected():
    with pytest.raises(ValueError, match="coordinate"):
        MolecularSystem(("H",), ((math.nan, 0.0, 0.0),), 0, 1)
    adapter = XTBloomAdapter(protocol=protocol(), evaluator=lambda item, operation: {
        "charge": item.charge,
        "multiplicity": item.multiplicity,
        "converged": True,
        "energy": float("nan"),
    })
    with pytest.raises((ValueError, CalculatorProtocolError)):
        adapter.evaluate(system(), operation="energy")


def test_capability_matrix_rejects_unknown_operation_and_bad_identity():
    with pytest.raises(ValueError):
        CalculatorCapabilities("mock", "pass", "ok", operations=("hessian",))
    adapter = XTBloomAdapter(protocol=protocol(), evaluator=lambda item, operation: {
        "charge": item.charge,
        "multiplicity": item.multiplicity,
        "converged": True,
        "energy": 0.0,
    })
    with pytest.raises(CalculatorProtocolError):
        adapter.evaluate(system(), operation="forces")
