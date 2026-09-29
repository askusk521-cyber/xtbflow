from __future__ import annotations

import pytest

pytest.importorskip("ase")

from xtbflow.calculators import CalculatorProtocol, MolecularSystem, XTBloomAdapter
from xtbflow.runtime import RunLedger, StageBudget
from xtbflow.validation.ase_dimer import ASEDimerConfig, run_ase_dimer_search


def _protocol() -> CalculatorProtocol:
    return CalculatorProtocol(
        protocol_id="analytic-saddle-v1",
        calculator="xtbloom",
        method="analytic-saddle-fixture",
        parameters={"fixture": True},
    )


def _system() -> MolecularSystem:
    return MolecularSystem(
        ("H",),
        ((0.2, 0.15, -0.1),),
        0,
        1,
        {"periodic": False},
        "analytic-saddle",
    )


def _backend(calls: list[str]) -> XTBloomAdapter:
    def evaluator(system, operation):
        calls.append(system.input_hash)
        x, y, z = system.coordinates[0]
        return {
            "charge": system.charge,
            "multiplicity": system.multiplicity,
            "converged": True,
            "energy": 0.5 * (-x * x + 2.0 * y * y + 3.0 * z * z),
            "forces": ((x, -2.0 * y, -3.0 * z),),
        }

    return XTBloomAdapter(
        protocol=_protocol(), evaluator=evaluator, require_budget_token=True
    )


def _token(tmp_path, calls: int):
    ledger = RunLedger(
        StageBudget("ase-dimer-fixture", max_calculator_calls=calls)
    )
    token = ledger.issue_calculator_token(
        "ase-dimer-fixture",
        calls,
        persist_path=tmp_path / "ledger.json",
    )
    return ledger, token


def test_ase_dimer_converges_analytic_first_order_saddle(tmp_path):
    calls: list[str] = []
    _, token = _token(tmp_path, 100)
    result = run_ase_dimer_search(
        _backend(calls),
        _system(),
        ((1.0, 0.0, 0.0),),
        budget_token=token,
        artifact_dir=tmp_path / "artifacts",
        config=ASEDimerConfig(
            fmax_eV_per_angstrom=0.01,
            max_steps=60,
            max_num_rot=1,
            random_seed=7,
        ),
    )

    assert result.status == "success"
    assert result.converged is True
    assert result.curvature_eV_per_angstrom2 < 0.0
    assert result.gradient_norm_hartree_per_angstrom < 5.0e-4
    assert max(abs(value) for row in result.coordinates_angstrom for value in row) < 0.01
    assert result.calculator_calls == token.consumed_calls == len(calls)
    assert "trajectory.traj" in result.artifact_files
    assert "dimer.log" in result.artifact_files


def test_ase_dimer_budget_stops_before_extra_backend_call(tmp_path):
    calls: list[str] = []
    _, token = _token(tmp_path, 1)
    result = run_ase_dimer_search(
        _backend(calls),
        _system(),
        ((1.0, 0.0, 0.0),),
        budget_token=token,
        artifact_dir=tmp_path / "budget-artifacts",
        config=ASEDimerConfig(max_steps=5),
    )

    assert result.status == "failure"
    assert result.converged is False
    assert result.calculator_calls == token.consumed_calls == 1
    assert len(calls) == 1
    assert "BudgetExceeded" in result.error


def test_ase_dimer_rejects_invalid_mode_before_calculator_call(tmp_path):
    calls: list[str] = []
    _, token = _token(tmp_path, 2)
    with pytest.raises(ValueError, match="shape"):
        run_ase_dimer_search(
            _backend(calls),
            _system(),
            ((1.0, 0.0),),
            budget_token=token,
            artifact_dir=tmp_path / "invalid-mode",
        )
    assert token.consumed_calls == 0
    assert calls == []


def test_gfn2_dimer_pilot_config_is_bounded_and_nonconfirmatory():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    document = json.loads(
        (root / "configs/validation/gfn2_dimer_pilot_v0.1.json").read_text()
    )
    assert document["status"] == "development_pilot_not_reference_admission"
    assert document["budget"]["max_calculator_calls"] == 96
    assert document["budget"]["max_concurrent_jobs"] == 1
    assert document["budget"]["max_retries_per_job"] == 0
    assert document["protocol"]["parameters"]["implementation"] == "tblite"
    assert document["system"]["multiplicity"] == 2
    mode = document["initial_mode"]
    assert [sum(row[axis] for row in mode) for axis in range(3)] == [0.0, 0.0, 0.0]
    assert any("not a reference DFT" in item for item in document["claim_limits"])
