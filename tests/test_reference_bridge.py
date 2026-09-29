from __future__ import annotations

import json
from pathlib import Path

import pytest

from xtbflow.calculators import CalculationResult, MolecularSystem
from xtbflow.validation import (
    bridge_case_record,
    force_delta_metrics,
    summarize_bridge,
)


def system():
    return MolecularSystem(
        ("H", "H"),
        ((0.0, 0.0, 0.0), (0.8, 0.0, 0.0)),
        0,
        1,
        system_id="hydrogen",
    )


def result(item, calculator, energy, forces):
    return CalculationResult(
        calculator=calculator,
        protocol_id=f"{calculator}-v1",
        input_hash=item.input_hash,
        charge=item.charge,
        multiplicity=item.multiplicity,
        operation="energy_forces",
        energy=energy,
        forces=forces,
    )


def test_force_metrics_use_normalized_component_deltas():
    metrics = force_delta_metrics(
        ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        ((1.3, 0.0, 0.0), (-0.7, 0.0, 0.0)),
    )
    assert metrics["component_mae_hartree_per_angstrom"] == pytest.approx(0.1)
    assert metrics["component_max_abs_hartree_per_angstrom"] == pytest.approx(0.3)


def test_bridge_record_keeps_energy_delta_per_system():
    item = system()
    reference = result(
        item, "cp2k", -1.1, ((0.1, 0.0, 0.0), (-0.1, 0.0, 0.0))
    )
    comparison = result(
        item, "xtb_oracle", -1.0, ((0.2, 0.0, 0.0), (-0.2, 0.0, 0.0))
    )
    record = bridge_case_record(
        item, reference, comparison, selection_reason="fixture"
    )
    assert record["same_geometry_energy_delta_hartree"] == pytest.approx(0.1)
    assert record["input_hash"] == item.input_hash


def test_summary_does_not_pool_absolute_energies():
    records = [
        {
            "status": "success",
            "bridge": {
                "force_delta": {
                    "component_mae_hartree_per_angstrom": 0.1,
                    "component_max_abs_hartree_per_angstrom": 0.3,
                }
            },
        },
        {
            "status": "success",
            "bridge": {
                "force_delta": {
                    "component_mae_hartree_per_angstrom": 0.2,
                    "component_max_abs_hartree_per_angstrom": 0.4,
                }
            },
        },
    ]
    summary = summarize_bridge(records)
    assert summary["force_component_mae_mean_hartree_per_angstrom"] == pytest.approx(0.15)
    assert summary["energy_aggregation"] == "not_reported_across_stoichiometries"


def test_force_metrics_reject_inventory_mismatch():
    with pytest.raises(ValueError, match="inventories differ"):
        force_delta_metrics(((0.0, 0.0, 0.0),), ((0.0, 0.0, 0.0),) * 2)


def test_bridge_record_contains_metrics_not_duplicate_backend_payloads():
    item = system()
    reference = result(
        item, "cp2k", -1.1, ((0.1, 0.0, 0.0), (-0.1, 0.0, 0.0))
    )
    comparison = result(
        item, "xtb_oracle", -1.0, ((0.2, 0.0, 0.0), (-0.2, 0.0, 0.0))
    )
    record = bridge_case_record(
        item, reference, comparison, selection_reason="fixture"
    )
    assert "reference" not in record
    assert "comparison" not in record
    assert "force_delta" in record


def test_frozen_bridge_config_matches_call_budget():
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs/experiments/cp2k_bridge.yaml").read_text(encoding="utf-8")
    )
    assert config["status"] == "development_pilot_not_reference_admission"
    assert len(config["cases"]) == config["budget"]["max_cp2k_calls"]
    assert len(config["cases"]) == config["budget"]["max_gfn2_calls"]
    assert config["analysis"]["aggregate_absolute_energies_across_stoichiometries"] is False
