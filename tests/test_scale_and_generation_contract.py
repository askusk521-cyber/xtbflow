from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from xtbflow.data import (
    nested_parent_subsets,
    parent_equal_mean,
    validate_nested_parent_subsets,
)
from xtbflow.evaluation import GenerationAttemptRecord, GenerationProtocol
from xtbflow.models import JointFlowRuntimeConfig, joint_flow_parameter_count, total_electron_projector


ROOT = Path(__file__).parents[1]


def test_scale_parameter_formula_matches_declared_ladder():
    runtime = JointFlowRuntimeConfig.load(ROOT / "configs/models/joint_flow_runtime_v0.1.json")
    expected = {64: 23387, 128: 79515, 256: 290075, 512: 1104411}
    assert runtime.trainable_parameter_count == expected[64]
    for width, count in expected.items():
        scaled = runtime.with_hidden_dim(width)
        assert scaled.trainable_parameter_count == count
        assert joint_flow_parameter_count(8, 4, width, 16) == count
        model = scaled.build(total_electron_projector(("C", "O", "H"), dtype=torch.float64)).double()
        assert sum(parameter.numel() for parameter in model.parameters()) == count


def test_matrix_validator_reports_primary_factorial_count():
    output = ROOT / "tests/.tmp_joint_scale_matrix.json"
    try:
        subprocess.run(
            [sys.executable, "scripts/validate_joint_scale_data_matrix.py", "--output", str(output)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["status"] == "PASS"
        assert report["primary_run_count"] == 81
        assert report["scientific_execution"] == "blocked_until_admitted_event_ts_manifest"
    finally:
        output.unlink(missing_ok=True)


def test_nested_parent_selection_is_result_blind_and_nested():
    parents = [f"parent-{index}" for index in range(10)]
    subsets = nested_parent_subsets(parents, [2, 5, 10], seed="fixed")
    validate_nested_parent_subsets(subsets)
    assert set(subsets[2]).issubset(subsets[5])
    assert set(subsets[5]).issubset(subsets[10])
    assert parent_equal_mean({"a": 1.0, "b": 3.0}) == 2.0
    with pytest.raises(ValueError, match="not nested"):
        validate_nested_parent_subsets({2: ("a", "b"), 3: ("a", "c", "d")})


def test_generation_protocol_requires_a_real_stochastic_source_for_repeats():
    GenerationProtocol(candidate_cap=1, attempt_seeds=(11,)).validate()
    with pytest.raises(ValueError, match="stochastic"):
        GenerationProtocol(candidate_cap=64, attempt_seeds=(11, 17)).validate()
    protocol = GenerationProtocol(
        candidate_cap=64,
        attempt_seeds=(11, 17),
        initial_state_policy="event_geometry_noise",
    )
    payload = protocol.as_dict()
    assert payload["stochastic_source_declared"] is True


def test_generation_attempt_failure_stages_are_not_collapsed():
    record = GenerationAttemptRecord(
        attempt_id="attempt-1",
        parent_id="parent-1",
        seed=11,
        status="budget_exhausted",
        failure_stage="physical_refinement",
        consumed_calculator_calls=4,
    )
    payload = record.as_dict()
    assert payload["status"] == "budget_exhausted"
    assert payload["failure_stage"] == "physical_refinement"
    with pytest.raises(ValueError, match="failure stage"):
        GenerationAttemptRecord(
            attempt_id="attempt-2",
            parent_id="parent-1",
            seed=17,
            status="execution_failed",
        ).validate()
