from __future__ import annotations

import pytest

pytest.importorskip("ase")

import math
import numpy as np

from xtbflow.calculators import (
    CalculatorBackend,
    CalculatorCapabilities,
    CalculatorProtocol,
    CalculationResult,
    MolecularSystem,
)
from xtbflow.runtime import RunLedger, StageBudget
from xtbflow.validation.xtb_irc import XTBIRCConfig, run_xtb_irc_certification
from xtbflow.validation.saddle_refinement import SaddleRefinementConfig, refine_saddle


class AnalyticSaddle(CalculatorBackend):
    """Two-coordinate quartic saddle used only for the adapter contract test."""

    protocol = CalculatorProtocol(
        protocol_id="analytic-saddle-v1",
        calculator="analytic",
        method="quartic-two-coordinate",
        parameters={"fixture": True},
    )

    @property
    def capabilities(self) -> CalculatorCapabilities:
        return CalculatorCapabilities(
            "analytic", "unknown", "test-only analytic backend", "fixture", "analytic-saddle-v1",
            ("energy_forces",), ("cpu",), "unit-tested",
        )

    def evaluate(self, system, *, operation="energy_forces", budget_token=None):
        if budget_token is not None:
            budget_token.consume()
        coordinates = np.asarray(system.coordinates, dtype=float)
        q = (coordinates[1, 0] - coordinates[0, 0]) / math.sqrt(2.0)
        center = (coordinates[1, 0] + coordinates[0, 0]) / math.sqrt(2.0)
        residual = coordinates.copy()
        residual[:, 0] = 0.0
        energy = 0.25 * (q * q - 1.0) ** 2 + 0.5 * center * center + 0.5 * float(np.sum(residual * residual))
        dq = q * (q * q - 1.0)
        dcenter = center
        gradient = residual.copy()
        gradient[0, 0] += (dcenter - dq) / math.sqrt(2.0)
        gradient[1, 0] += (dcenter + dq) / math.sqrt(2.0)
        forces = -gradient
        return CalculationResult(
            calculator=self.protocol.calculator,
            protocol_id=self.protocol.protocol_id,
            input_hash=system.input_hash,
            charge=system.charge,
            multiplicity=system.multiplicity,
            operation=operation,
            energy=float(energy),
            forces=tuple(tuple(float(value) for value in row) for row in forces),
            calculator_build_hash="analytic-saddle-v1",
        )


def _token(tmp_path, calls: int):
    ledger = RunLedger(StageBudget("xtb-irc-fixture", max_calculator_calls=calls))
    token = ledger.issue_calculator_token("xtb-irc-fixture", calls, persist_path=tmp_path / "ledger.json")
    return ledger, token


def _system() -> MolecularSystem:
    return MolecularSystem(
        ("H", "H"),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        0,
        1,
        {"periodic": False},
        "analytic-saddle",
    )


def test_xtb_irc_certifies_one_negative_mode_and_two_minima(tmp_path):
    ledger, token = _token(tmp_path, 256)
    result = run_xtb_irc_certification(
        AnalyticSaddle(),
        _system(),
        token=token,
        candidate_id="analytic-saddle-candidate",
        config=XTBIRCConfig(
            hessian_step_angstrom=0.01,
            endpoint_fmax_hartree_per_angstrom=1.0e-5,
            max_endpoint_steps=80,
            max_calls=256,
        ),
    )
    token.settle(recorded_calls=token.consumed_calls, persist_path=tmp_path / "ledger.json")

    assert result.status == "success"
    assert result.reason_code == "CERTIFIED"
    assert result.validation is not None
    assert result.validation["status"] == "success"
    assert result.evidence["negative_mode_count"] == 1
    assert len(result.evidence["endpoints"]) == 2
    assert all(row["minimum_converged"] for row in result.evidence["endpoints"])
    assert result.calculator_calls == ledger.summary()["totals"]["calculator_calls"]


def test_xtb_irc_budget_failure_is_structured_and_replayable(tmp_path):
    first_ledger, first_token = _token(tmp_path / "first", 2)
    config = XTBIRCConfig(hessian_step_angstrom=0.01, max_calls=2)
    first = run_xtb_irc_certification(
        AnalyticSaddle(), _system(), token=first_token,
        candidate_id="analytic-saddle-candidate", config=config,
    )
    first_token.settle(recorded_calls=first_token.consumed_calls, persist_path=tmp_path / "first" / "ledger.json")

    second_ledger, second_token = _token(tmp_path / "second", 2)
    second = run_xtb_irc_certification(
        AnalyticSaddle(), _system(), token=second_token,
        candidate_id="analytic-saddle-candidate", config=config,
    )
    second_token.settle(recorded_calls=second_token.consumed_calls, persist_path=tmp_path / "second" / "ledger.json")

    assert first.status == second.status == "budget_exhausted"
    assert first.reason_code == second.reason_code == "BUDGET_EXCEEDED"
    assert first.calculator_calls == second.calculator_calls == 2
    assert first.evidence["error"].startswith("BudgetExceeded:")
    assert second_ledger.summary()["totals"]["calculator_calls"] == 2


def test_saddle_refinement_contract_accepts_stationary_analytic_saddle(tmp_path):
    ledger, token = _token(tmp_path, 64)
    result = refine_saddle(
        AnalyticSaddle(),
        _system(),
        _system().coordinates,
        token=token,
        config=SaddleRefinementConfig(max_steps=2, max_calls=64),
    )
    token.settle(recorded_calls=token.consumed_calls, persist_path=tmp_path / "ledger.json")

    assert result.status == "success"
    assert result.reason_code == "STATIONARY_POINT_REACHED"
    assert result.gradient_norm == 0.0
    assert result.calculator_calls == 1
