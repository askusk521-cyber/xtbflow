#!/usr/bin/env python3
"""Validate Workflow B's shared controls and frozen baseline protocol.

This is a bounded CPU software acceptance check.  It does not train on event or
transition-state labels, call an electronic-structure calculator, or establish
that any baseline is scientifically superior.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import torch

from mechai.data.events import LewisState
from xtbflow.models import CONTROL_MODES, JointFlowRuntimeConfig, pack_be, total_electron_projector
from xtbflow.proposals import CoreIntent, CoreRuleConfig, RoleSet, enumerate_core_events
from xtbflow.training import coupled_control_manifest, model_state_identity


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/runs/workflow_b_baseline_ladder_v0.1.json"
PROTOCOL_SCHEMA = ROOT / "schemas/workflow_b_baseline.schema.json"
RUNTIME_CONFIG = ROOT / "configs/models/joint_flow_runtime_v0.1.json"
EXPECTED_ARMS = (
    "strong_rule",
    "conserved_independent",
    "serial_event_to_geometry",
    "current_event_to_geometry",
    "joint_event_geometry",
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_identity() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(porcelain), "dirty_entries": porcelain}


def _validate_required_fields(payload: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    required = schema.get("required")
    if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
        raise ValueError("workflow B schema has an invalid required list")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"workflow B protocol is missing required keys: {missing}")
    expected_schema = schema.get("properties", {}).get("schema", {}).get("const")
    if payload.get("schema") != expected_schema:
        raise ValueError("workflow B protocol schema identity does not match its JSON schema")


def validate_protocol() -> dict[str, object]:
    protocol = _read_object(PROTOCOL)
    schema = _read_object(PROTOCOL_SCHEMA)
    _validate_required_fields(protocol, schema)
    if protocol.get("status") != "software_protocol_frozen_scientific_execution_blocked":
        raise ValueError("workflow B must remain scientifically blocked until paired supervision is admitted")

    blocker = protocol.get("blocking_dependency")
    if not isinstance(blocker, Mapping) or blocker.get("issue") != 58:
        raise ValueError("workflow B must name issue 58 as its paired-supervision blocker")
    if blocker.get("spice_energy_force_is_not_sufficient") is not True:
        raise ValueError("the protocol must not treat SPICE E/F as event/TS supervision")

    matrix = protocol.get("first_scientific_matrix")
    if not isinstance(matrix, list) or tuple(arm.get("arm_id") for arm in matrix) != EXPECTED_ARMS:
        raise ValueError("workflow B baseline arms or their frozen order changed")
    if matrix[0].get("implementation") != "xtbflow.proposals.enumerate_core_events":
        raise ValueError("the strong-rule arm must use the registered reactant-only implementation")
    if enumerate_core_events.__module__ != "xtbflow.proposals.core_rules":
        raise ValueError("strong-rule implementation did not resolve to the expected source module")

    learned_modes = tuple(
        arm.get("learned_control_mode")
        for arm in matrix
        if arm.get("kind") == "learned"
    )
    if learned_modes != CONTROL_MODES:
        raise ValueError("learned baseline arms must map exactly to the registered control modes")
    gate = protocol.get("decision_gate", {})
    if gate.get("title_claim_requires_joint_over_current_one_way") is not True or gate.get("comparison") != "joint_event_geometry versus current_event_to_geometry":
        raise ValueError("coupling attribution requires the current-state one-way control")

    fairness = protocol.get("fairness")
    if not isinstance(fairness, Mapping):
        raise ValueError("workflow B fairness contract must be an object")
    required_true = (
        "same_admitted_record_ids",
        "same_parent_group_split_fingerprint",
        "same_reactant_input_view_fingerprint",
        "same_forbidden_input_firewall",
        "learned_arms_share_runtime_class",
        "learned_arms_share_trainable_parameter_count",
        "learned_arms_start_from_same_state_per_seed",
        "arm_specific_training_checkpoints_are_recorded",
        "same_optimizer_and_batch_exposures",
    )
    if any(fairness.get(key) is not True for key in required_true):
        raise ValueError("workflow B fairness invariants must be explicitly enabled")
    if fairness.get("generator_selection_calculator_calls_per_parent") != 0:
        raise ValueError("generator-only selection must not hide calculator work")
    seeds = fairness.get("training_seeds")
    if not isinstance(seeds, list) or len(seeds) < 3 or any(type(seed) is not int for seed in seeds):
        raise ValueError("workflow B requires at least three explicit integer training seeds")

    runtime = JointFlowRuntimeConfig.load(RUNTIME_CONFIG)
    if runtime.control_modes != CONTROL_MODES:
        raise ValueError("runtime config and frozen learned-control modes differ")

    # Execute every first-matrix arm on one tiny reactant-side fixture.  The
    # learned model is intentionally untrained; this proves routing and shared
    # identities only, not comparative performance.
    torch.manual_seed(20260929)
    symbols = ("C", "O")
    reactant = LewisState(symbols, ((4, 0), (0, 6)), 0, 1)
    rules = enumerate_core_events(
        reactant,
        RoleSet(nucleophiles=(1,), electrophiles=(0,)),
        config=CoreRuleConfig(intents=(CoreIntent.ADDITION,)),
    )
    if len(rules.candidates) != 1:
        raise ValueError("strong-rule software fixture did not produce its expected bounded candidate")

    projector = total_electron_projector(symbols, dtype=torch.float64)
    model = runtime.build(projector).double()
    model_identity = model_state_identity(model)
    event_state = pack_be(torch.tensor([reactant.be], dtype=torch.float64))
    coordinates = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.2, 0.1, 0.0]]],
        dtype=torch.float64,
    )
    node_features = torch.randn(1, 2, runtime.node_feature_dim, dtype=torch.float64)
    atom_mask = torch.ones(1, 2, dtype=torch.bool)
    conditions = torch.zeros(1, runtime.condition_feature_dim, dtype=torch.float64)
    learned_execution: dict[str, dict[str, object]] = {}
    with torch.no_grad():
        for mode in runtime.control_modes:
            output = model.forward_control(
                mode,
                event_state,
                coordinates,
                node_features,
                atom_mask,
                tau=0.5,
                dt=0.25,
                condition_features=conditions,
            )
            residual = float(projector.residual(output.event_velocity).abs().max())
            finite = bool(
                torch.isfinite(output.event_velocity).all()
                and torch.isfinite(output.geometry_velocity).all()
            )
            if not finite or residual > 1e-8:
                raise ValueError(f"software control {mode} failed finite/conservation acceptance")
            learned_execution[mode] = {
                "event_velocity_finite": finite,
                "geometry_velocity_finite": finite,
                "conservation_residual_max_abs": residual,
            }
    measured_manifest = coupled_control_manifest(
        modes=runtime.control_modes,
        models={mode: model for mode in runtime.control_modes},
        physical_call_budgets={mode: 0 for mode in runtime.control_modes},
        coupling_strength=1.0,
        guidance_strength=0.0,
        dataset_fingerprint="not-executed-paired-corpus-blocked-by-issue-58",
        split_fingerprint="not-executed",
        input_view_fingerprint="reactant-side-contract-v0.1",
    )
    if measured_manifest["same_generation_weights_for_controls"] is not True:
        raise ValueError("registered learned controls do not share a measured model state")
    if measured_manifest["same_physical_budget_for_controls"] is not True:
        raise ValueError("software acceptance controls do not share a measured zero-call budget")

    return {
        "schema": "xtbflow-workflow-b-software-acceptance/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": _sha256(PROTOCOL),
        "protocol_schema_sha256": _sha256(PROTOCOL_SCHEMA),
        "runtime_config_sha256": _sha256(RUNTIME_CONFIG),
        "git": _git_identity(),
        "runtime_constructor": runtime.constructor_record(),
        "model_identity": model_identity,
        "software_arm_execution": {
            "strong_rule": {
                "attempted": rules.n_attempted,
                "accepted_candidates": len(rules.candidates),
                "truncated": rules.truncated,
            },
            "learned_controls": learned_execution,
        },
        "measured_control_manifest": measured_manifest,
        "checks": [
            "five-arm first scientific matrix is frozen",
            "learned arms include the current-state joint_unidirectional ablation",
            "learned software controls share one measured parameter state",
            "software acceptance assigns zero calculator calls to every learned control",
            "issue 58 remains the explicit paired event/TS supervision blocker",
            "SPICE energy/force labels are not admitted as event/TS supervision",
        ],
        "limits": [
            "This report is software acceptance only.",
            "No real event/TS training or held-out scientific comparison was run.",
            "No calculator or quantum-chemistry call was made.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    report = validate_protocol()
    if args.require_clean and report["git"]["dirty"]:
        raise SystemExit("workflow B acceptance requires a clean committed worktree")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
