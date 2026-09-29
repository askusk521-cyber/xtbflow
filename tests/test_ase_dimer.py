from __future__ import annotations

import pytest

pytest.importorskip("ase")

from xtbflow.calculators import CalculatorProtocol, MolecularSystem, XTBloomAdapter
from xtbflow.runtime import RunLedger, StageBudget
from xtbflow.validation.ase_dimer import (
    ASEDimerConfig,
    RigidBodyProjection,
    project_rigid_body_components,
    rigid_body_basis,
    run_ase_dimer_search,
    run_dimer_preflight,
)


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


def test_dimer_preflight_measures_force_and_negative_curvature(tmp_path):
    calls: list[str] = []
    _, token = _token(tmp_path, 3)
    result = run_dimer_preflight(
        _backend(calls),
        _system(),
        ((1.0, 0.0, 0.0),),
        budget_token=token,
        step_angstrom=1.0e-4,
        config=ASEDimerConfig(remove_rigid_body_modes=False),
    )

    assert result.status == "success"
    assert result.calculator_calls == token.consumed_calls == len(calls) == 3
    assert result.rigid_body_rank == 0
    assert result.mode_curvature_hartree_per_angstrom2 == pytest.approx(-1.0)
    assert result.mode_curvature_eV_per_angstrom2 == pytest.approx(-27.211386245988)
    assert result.projected_force_norm_hartree_per_angstrom == pytest.approx(
        0.22**0.5
    )


def test_dimer_preflight_fails_closed_when_three_call_budget_is_missing(tmp_path):
    calls: list[str] = []
    _, token = _token(tmp_path, 2)
    result = run_dimer_preflight(
        _backend(calls),
        _system(),
        ((1.0, 0.0, 0.0),),
        budget_token=token,
        config=ASEDimerConfig(remove_rigid_body_modes=False),
    )

    assert result.status == "failure"
    assert result.calculator_calls == token.consumed_calls == len(calls) == 2
    assert "BudgetExceeded" in result.error
    assert result.mode_curvature_hartree_per_angstrom2 is None


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
            remove_rigid_body_modes=False,
        ),
    )

    assert result.status == "success"
    assert result.converged is True
    assert result.curvature_eV_per_angstrom2 < 0.0
    assert result.rigid_body_rank == 0
    assert result.gradient_norm_hartree_per_angstrom < 5.0e-4
    assert max(abs(value) for row in result.coordinates_angstrom for value in row) < 0.01
    assert result.calculator_calls == token.consumed_calls == len(calls)
    assert "trajectory.jsonl" in result.artifact_files
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
        config=ASEDimerConfig(max_steps=5, remove_rigid_body_modes=False),
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
    assert document["search"]["remove_rigid_body_modes"] is True
    mode = document["initial_mode"]
    assert [sum(row[axis] for row in mode) for axis in range(3)] == [0.0, 0.0, 0.0]
    assert any("not a reference DFT" in item for item in document["claim_limits"])


def test_nh3_dimer_pilot_declares_separate_three_call_preflight():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    document = json.loads(
        (root / "configs/validation/gfn2_dimer_nh3_pilot_v0.1.json").read_text()
    )
    assert document["status"] == "development_pilot_not_reference_admission"
    assert document["preflight"]["budget"]["max_calculator_calls"] == 3
    assert document["preflight"]["budget"]["max_concurrent_jobs"] == 1
    assert document["preflight"]["budget"]["max_retries_per_job"] == 0
    assert document["preflight"]["gate"]["require_negative_mode_curvature"] is True
    assert document["budget"]["max_calculator_calls"] == 128
    assert document["system"]["symbols"] == ["N", "H", "H", "H"]
    assert document["system"]["multiplicity"] == 1
    assert document["search"]["remove_rigid_body_modes"] is True
    assert any("three-call preflight" in item for item in document["claim_limits"])


def test_rigid_body_projection_removes_five_linear_molecule_modes():
    import numpy as np

    coordinates = np.asarray([[-0.93, 0.0, 0.0], [0.0, 0.0, 0.0], [0.93, 0.0, 0.0]])
    basis = rigid_body_basis(coordinates)
    assert basis.shape == (9, 5)
    translation = np.asarray([[1.0, -2.0, 0.5]] * 3)
    projected_translation, rank = project_rigid_body_components(
        coordinates, translation
    )
    assert rank == 5
    assert np.linalg.norm(projected_translation) < 1.0e-12

    rotation = np.cross(
        np.broadcast_to(np.asarray([0.0, 1.0, 0.0]), coordinates.shape),
        coordinates,
    )
    projected_rotation, _ = project_rigid_body_components(coordinates, rotation)
    assert np.linalg.norm(projected_rotation) < 1.0e-12

    reaction_mode = np.asarray([[1.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    projected_reaction, _ = project_rigid_body_components(
        coordinates, reaction_mode
    )
    assert np.linalg.norm(projected_reaction - reaction_mode) < 1.0e-12


def test_rigid_body_constraint_projects_forces_and_updates():
    import numpy as np
    from ase import Atoms

    atoms = Atoms("H3", positions=[[-0.93, 0.0, 0.0], [0.0, 0.0, 0.0], [0.93, 0.0, 0.0]])
    constraint = RigidBodyProjection()
    forces = np.asarray([[1.0, 0.0, 0.0]] * 3)
    constraint.adjust_forces(atoms, forces)
    assert np.linalg.norm(forces) < 1.0e-12
    new = atoms.get_positions() + np.asarray([[0.1, 0.2, 0.0]] * 3)
    constraint.adjust_positions(atoms, new)
    assert np.linalg.norm(new - atoms.get_positions()) < 1.0e-12


def test_rigid_projected_dimer_converges_three_atom_saddle(tmp_path):
    import numpy as np

    reference = np.asarray([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mode = np.asarray([[1.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mode /= np.linalg.norm(mode)
    stable = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    stable /= np.linalg.norm(stable)
    start = reference + 0.12 * mode + 0.05 * stable
    calls: list[str] = []

    def evaluator(system, operation):
        calls.append(system.input_hash)
        displacement = np.asarray(system.coordinates) - reference
        reaction = float(np.sum(displacement * mode))
        orthogonal = float(np.sum(displacement * stable))
        centered_yz = displacement[:, 1:] - displacement[:, 1:].mean(
            axis=0, keepdims=True
        )
        energy = -0.5 * reaction**2 + 1.5 * orthogonal**2 + np.sum(centered_yz**2)
        gradient = -reaction * mode + 3.0 * orthogonal * stable
        gradient[:, 1:] += 2.0 * centered_yz
        return {
            "charge": 0,
            "multiplicity": 1,
            "converged": True,
            "energy": float(energy),
            "forces": tuple(map(tuple, -gradient)),
        }
    protocol = CalculatorProtocol(
        "three-atom-saddle-v1",
        "xtbloom",
        "analytic-saddle-fixture",
        parameters={"fixture": True},
    )
    backend = XTBloomAdapter(
        protocol=protocol, evaluator=evaluator, require_budget_token=True
    )
    _, token = _token(tmp_path, 200)
    system = MolecularSystem(
        ("H", "H", "H"),
        tuple(map(tuple, start)),
        0,
        1,
        {"periodic": False},
        "three-atom-saddle",
    )
    result = run_ase_dimer_search(
        backend,
        system,
        mode,
        budget_token=token,
        artifact_dir=tmp_path / "rigid-dimer-artifacts",
        config=ASEDimerConfig(
            fmax_eV_per_angstrom=0.01,
            max_steps=80,
            maximum_translation_angstrom=0.05,
            random_seed=1,
        ),
    )
    assert result.status == "success"
    assert result.converged is True
    assert result.rigid_body_rank == 5
    assert 1 <= result.optimizer_steps <= 10
    assert result.calculator_calls == token.consumed_calls == len(calls)
    assert result.curvature_eV_per_angstrom2 < 0.0
    assert result.gradient_norm_hartree_per_angstrom < 1.0e-10
    coordinates = np.asarray(result.coordinates_angstrom)
    assert np.linalg.norm(coordinates.mean(axis=0)) < 1.0e-12
    assert np.max(np.abs(coordinates - reference)) < 1.0e-10
    trajectory = tmp_path / "rigid-dimer-artifacts/trajectory.jsonl"
    assert len(trajectory.read_text().splitlines()) == result.optimizer_steps + 1
