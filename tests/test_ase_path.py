from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ase")

from xtbflow.calculators import CalculatorProtocol, MolecularSystem, XTBloomAdapter
from xtbflow.runtime import RunLedger, StageBudget
from xtbflow.validation.ase_dimer import rigid_body_basis
from xtbflow.validation.ase_path import (
    ASERelaxConfig,
    internal_cartesian_basis,
    reconstruct_projected_mode,
    run_ase_minimum_relaxation,
    signed_atom_plane_distance,
)


def _protocol() -> CalculatorProtocol:
    return CalculatorProtocol(
        protocol_id="analytic-double-well-v1",
        calculator="xtbloom",
        method="analytic-double-well-fixture",
        parameters={"fixture": True},
    )


def _backend(calls: list[str]) -> XTBloomAdapter:
    def evaluator(system, operation):
        calls.append(system.input_hash)
        x, y, z = system.coordinates[0]
        energy = 0.25 * (x * x - 1.0) ** 2 + y * y + 1.5 * z * z
        return {
            "charge": system.charge,
            "multiplicity": system.multiplicity,
            "converged": True,
            "energy": energy,
            "forces": ((x * (1.0 - x * x), -2.0 * y, -3.0 * z),),
        }

    return XTBloomAdapter(
        protocol=_protocol(), evaluator=evaluator, require_budget_token=True
    )


def _system(x: float = 0.2) -> MolecularSystem:
    return MolecularSystem(
        ("H",),
        ((x, 0.1, -0.1),),
        0,
        1,
        {"periodic": False},
        "analytic-double-well",
    )


def _token(tmp_path, calls: int):
    ledger = RunLedger(
        StageBudget("ase-relax-fixture", max_calculator_calls=calls)
    )
    token = ledger.issue_calculator_token(
        "ase-relax-fixture",
        calls,
        persist_path=tmp_path / f"ledger-{calls}.json",
    )
    return ledger, token


def test_internal_cartesian_basis_removes_six_nonlinear_rigid_modes():
    coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.9927, 0.0, 0.0],
            [-0.49635, 0.8597, 0.0],
            [-0.49635, -0.8597, 0.0],
        ]
    )
    basis, rank = internal_cartesian_basis(coordinates)
    rigid = rigid_body_basis(coordinates)
    flat = basis.reshape(len(basis), -1)

    assert rank == 6
    assert basis.shape == (6, 4, 3)
    assert np.allclose(flat @ flat.T, np.eye(6), atol=1.0e-10)
    assert np.allclose(flat @ rigid, 0.0, atol=1.0e-10)


def test_reconstruct_projected_mode_uses_eigenvector_columns():
    basis = np.asarray(
        [
            [[1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0]],
        ]
    )
    eigenvectors = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    mode = reconstruct_projected_mode(basis, eigenvectors, 0)
    assert np.allclose(mode, [[0.0, 1.0, 0.0]])


def test_signed_atom_plane_distance_distinguishes_opposite_pyramids():
    plus = np.asarray(
        [
            [0.0, 0.0, 0.3],
            [1.0, 0.0, 0.0],
            [-0.5, 0.8660254, 0.0],
            [-0.5, -0.8660254, 0.0],
        ]
    )
    minus = plus.copy()
    minus[0, 2] = -0.3

    plus_distance = signed_atom_plane_distance(
        plus, atom_index=0, plane_indices=(1, 2, 3)
    )
    minus_distance = signed_atom_plane_distance(
        minus, atom_index=0, plane_indices=(1, 2, 3)
    )
    assert plus_distance == pytest.approx(0.3)
    assert minus_distance == pytest.approx(-0.3)


def test_ase_minimum_relaxation_reaches_positive_double_well_minimum(tmp_path):
    calls: list[str] = []
    _, token = _token(tmp_path, 100)
    result = run_ase_minimum_relaxation(
        _backend(calls),
        _system(0.2),
        budget_token=token,
        artifact_dir=tmp_path / "positive-minimum",
        config=ASERelaxConfig(
            fmax_eV_per_angstrom=0.005,
            max_steps=80,
            maximum_step_angstrom=0.1,
            remove_rigid_body_modes=False,
        ),
    )

    assert result.status == "success"
    assert result.converged is True
    assert result.calculator_calls == token.consumed_calls == len(calls)
    assert result.calculator_calls < 100
    assert result.coordinates_angstrom[0][0] == pytest.approx(1.0, abs=2.0e-4)
    assert abs(result.coordinates_angstrom[0][1]) < 2.0e-4
    assert abs(result.coordinates_angstrom[0][2]) < 2.0e-4
    assert result.gradient_norm_hartree_per_angstrom < 2.0e-4
    assert "trajectory.jsonl" in result.artifact_files
    assert "optimizer.log" in result.artifact_files


def test_ase_minimum_relaxation_stops_at_budget_boundary(tmp_path):
    calls: list[str] = []
    _, token = _token(tmp_path, 1)
    result = run_ase_minimum_relaxation(
        _backend(calls),
        _system(0.2),
        budget_token=token,
        artifact_dir=tmp_path / "budget-minimum",
        config=ASERelaxConfig(remove_rigid_body_modes=False),
    )

    assert result.status == "failure"
    assert result.converged is False
    assert result.calculator_calls == token.consumed_calls == len(calls) == 1
    assert "BudgetExceeded" in result.error


def test_nh3_path_config_freezes_independent_hessian_and_endpoint_budgets():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    document = json.loads(
        (root / "configs/validation/gfn2_nh3_path_v0.1.json").read_text()
    )
    assert document["schema"] == "xtbflow-gfn2-path-pilot/v1"
    assert document["status"] == "development_path_validation_not_reference_admission"
    assert document["provenance"]["parent_evidence_sha256"] == (
        "3f21c58be89860c641d0f8e70fbed8135047104020dcb08cb8a54fb6a51c2d2c"
    )
    assert document["hessian"]["budget"]["max_calculator_calls"] == 13
    assert document["hessian"]["expected_negative_modes"] == 1
    assert document["hessian"]["minimum_absolute_dimer_mode_overlap"] == 0.95
    assert document["endpoints"]["budget_per_endpoint"]["max_calculator_calls"] == 96
    assert document["endpoints"]["order_parameter"]["kind"] == (
        "signed_atom_plane_distance"
    )
    assert document["endpoints"]["order_parameter"]["require_opposite_signs"] is True
    assert any(
        "preserves bond connectivity" in item
        for item in document["claim_limits"]
    )
