#!/usr/bin/env python3
"""Validate the frozen joint architecture/scale/data planning matrix.

This command only checks configuration and emits no training or chemistry
result.  It is intentionally usable on a machine without PyTorch or n2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SCALE_CONFIG = ROOT / "configs/models/joint_flow_scale_ladder_v0.1.json"
MATRIX_CONFIG = ROOT / "configs/runs/joint_scale_data_matrix_v0.1.json"
CONTROL_MODES = {"both_off", "serial_independent", "joint_unidirectional", "joint_bidirectional"}


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _parameter_count(dimensions: Mapping[str, Any], hidden_dim: int) -> int:
    n = int(dimensions["node_feature_dim"])
    c = int(dimensions["condition_feature_dim"])
    h = int(hidden_dim)
    r = int(dimensions["radial_features"])
    count = (n + 1) * h + h * h + 3 * h + (n + 2) * h + n
    if c:
        count += h * h + (c + 2) * h + (c + n + 1) * h + n
    count += 2 * h * h + 4 * h + 1
    count += (r + 2) * h + 1
    count += (n + 3) * h + n
    count += (2 * n + r + 3) * h + 1
    return count


def validate(scale: Mapping[str, Any], matrix: Mapping[str, Any]) -> dict[str, Any]:
    if scale.get("schema") != "xtbflow-joint-scale-ladder/v1":
        raise ValueError("unsupported scale ladder schema")
    if matrix.get("schema") != "xtbflow-joint-scale-data-matrix/v1":
        raise ValueError("unsupported scale/data matrix schema")
    dimensions = scale.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise ValueError("scale ladder dimensions are required")
    scales = scale.get("scales")
    if not isinstance(scales, list) or not scales:
        raise ValueError("scale ladder must declare at least one scale")
    scale_ids: set[str] = set()
    widths: list[int] = []
    measured_counts: dict[str, int] = {}
    for item in scales:
        if not isinstance(item, Mapping):
            raise ValueError("each scale must be an object")
        scale_id = item.get("scale_id")
        width = item.get("hidden_dim")
        expected = item.get("expected_trainable_parameters")
        if not isinstance(scale_id, str) or not scale_id or scale_id in scale_ids:
            raise ValueError("scale IDs must be unique non-empty strings")
        if type(width) is not int or width < 1:
            raise ValueError("scale hidden_dim must be a positive integer")
        if type(expected) is not int or expected != _parameter_count(dimensions, width):
            raise ValueError(f"parameter count mismatch for scale {scale_id}")
        scale_ids.add(scale_id)
        widths.append(width)
        measured_counts[scale_id] = expected
    if widths != sorted(widths):
        raise ValueError("scale widths must be increasing")

    architectures = matrix.get("primary_architectures")
    if not isinstance(architectures, list) or len(architectures) != 3:
        raise ValueError("the primary matrix must contain exactly three architecture arms")
    architecture_ids: set[str] = set()
    for arm in architectures:
        if not isinstance(arm, Mapping):
            raise ValueError("architecture arms must be objects")
        arm_id = arm.get("architecture_id")
        mode = arm.get("control_mode")
        if not isinstance(arm_id, str) or not arm_id or arm_id in architecture_ids:
            raise ValueError("architecture IDs must be unique non-empty strings")
        if mode not in CONTROL_MODES:
            raise ValueError(f"unsupported control mode: {mode}")
        architecture_ids.add(arm_id)

    model_scales = matrix.get("model_scales")
    if not isinstance(model_scales, Mapping):
        raise ValueError("model_scales is required")
    primary_scale_ids = model_scales.get("primary_scale_ids")
    if not isinstance(primary_scale_ids, list) or len(primary_scale_ids) != 3:
        raise ValueError("the primary matrix must contain exactly three model scales")
    if len(set(primary_scale_ids)) != len(primary_scale_ids) or not set(primary_scale_ids).issubset(scale_ids):
        raise ValueError("primary model scales must reference declared scale IDs")

    data_scales = matrix.get("data_scales")
    if not isinstance(data_scales, list) or len(data_scales) != 3:
        raise ValueError("the primary matrix must contain exactly three data scales")
    parent_counts: list[int] = []
    data_ids: set[str] = set()
    for item in data_scales:
        if not isinstance(item, Mapping):
            raise ValueError("data scales must be objects")
        data_id = item.get("data_scale_id")
        count = item.get("train_parent_count")
        if not isinstance(data_id, str) or not data_id or data_id in data_ids:
            raise ValueError("data scale IDs must be unique non-empty strings")
        if type(count) is not int or count < 1:
            raise ValueError("train_parent_count must be a positive integer")
        data_ids.add(data_id)
        parent_counts.append(count)
    if parent_counts != sorted(parent_counts):
        raise ValueError("data parent counts must be increasing")

    training = matrix.get("training")
    if not isinstance(training, Mapping) or training.get("seeds") != [11, 17, 23]:
        raise ValueError("the primary matrix requires the three frozen training seeds")
    expected_runs = len(architectures) * len(primary_scale_ids) * len(data_scales) * len(training["seeds"])
    if training.get("primary_run_count") != expected_runs:
        raise ValueError("primary_run_count does not match the declared factorial matrix")
    evaluation = matrix.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("evaluation contract is required")
    if evaluation.get("generator_only_is_primary") is not True:
        raise ValueError("generator-only metrics must be primary")
    if evaluation.get("label_assisted_event_and_geometry_metrics_are_diagnostic_only") is not True:
        raise ValueError("label-assisted metrics must be diagnostic only")
    blocker = matrix.get("scientific_blocker")
    if not isinstance(blocker, Mapping) or blocker.get("energy_force_only_data_is_insufficient") is not True:
        raise ValueError("the data admission blocker must remain explicit")

    return {
        "schema": "xtbflow-joint-scale-data-matrix-validation/v1",
        "status": "PASS",
        "primary_run_count": expected_runs,
        "primary_architecture_ids": sorted(architecture_ids),
        "primary_scale_ids": list(primary_scale_ids),
        "train_parent_counts": parent_counts,
        "parameter_counts": measured_counts,
        "scientific_execution": "blocked_until_admitted_event_ts_manifest",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale-config", type=Path, default=SCALE_CONFIG)
    parser.add_argument("--matrix-config", type=Path, default=MATRIX_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = validate(_load(args.scale_config), _load(args.matrix_config))
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
