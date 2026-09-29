from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from xtbflow.models import CONTROL_MODES, JointFlowRuntimeConfig, total_electron_projector
from xtbflow.training import coupled_control_manifest


ROOT = Path(__file__).parents[1]
PROTOCOL = ROOT / "configs/runs/workflow_b_baseline_ladder_v0.1.json"
RUNTIME = ROOT / "configs/models/joint_flow_runtime_v0.1.json"


def test_workflow_b_protocol_freezes_the_first_matrix_and_scientific_blocker():
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert payload["status"] == "software_protocol_frozen_scientific_execution_blocked"
    assert payload["blocking_dependency"]["issue"] == 58
    assert payload["blocking_dependency"]["spice_energy_force_is_not_sufficient"] is True
    assert [arm["arm_id"] for arm in payload["first_scientific_matrix"]] == [
        "strong_rule",
        "conserved_independent",
        "serial_event_to_geometry",
        "joint_event_geometry",
    ]
    assert tuple(
        arm["learned_control_mode"]
        for arm in payload["first_scientific_matrix"]
        if arm["kind"] == "learned"
    ) == CONTROL_MODES
    assert payload["fairness"]["generator_selection_calculator_calls_per_parent"] == 0
    assert len(payload["fairness"]["training_seeds"]) >= 3


def test_runtime_config_builds_one_model_for_all_registered_controls():
    runtime = JointFlowRuntimeConfig.load(RUNTIME)
    assert runtime.control_modes == CONTROL_MODES
    model = runtime.build(total_electron_projector(("C", "O", "H"), dtype=torch.float64)).double()
    manifest = coupled_control_manifest(
        modes=runtime.control_modes,
        models={mode: model for mode in runtime.control_modes},
        physical_call_budgets={mode: 0 for mode in runtime.control_modes},
    )
    assert manifest["model_identity_status"] == "measured"
    assert manifest["same_generation_weights_for_controls"] is True
    assert manifest["same_physical_budget_for_controls"] is True


def test_workflow_b_validation_script_emits_machine_readable_acceptance(tmp_path: Path):
    output = tmp_path / "workflow-b.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT / "vendor/mechai_reusable"), env.get("PYTHONPATH", "")]
    )
    subprocess.run(
        [sys.executable, "scripts/validate_workflow_b.py", "--output", str(output)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema"] == "xtbflow-workflow-b-software-acceptance/v1"
    assert report["passed"] is True
    assert report["software_arm_execution"]["strong_rule"]["accepted_candidates"] == 1
    assert set(report["software_arm_execution"]["learned_controls"]) == set(CONTROL_MODES)
    assert all(
        item["event_velocity_finite"] and item["geometry_velocity_finite"]
        for item in report["software_arm_execution"]["learned_controls"].values()
    )
    assert report["measured_control_manifest"]["same_generation_weights_for_controls"] is True
    assert report["measured_control_manifest"]["same_physical_budget_for_controls"] is True
    assert any("No real event/TS training" in item for item in report["limits"])
