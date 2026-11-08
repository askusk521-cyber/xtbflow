from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _planner_module():
    path = ROOT / "scripts" / "plan_joint_scale_runs.py"
    spec = importlib.util.spec_from_file_location("xtbflow_scale_planner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parent(parent_id: str, admission: str) -> dict[str, object]:
    return {
        "parent_reaction_id": parent_id,
        "family_id": f"family-{parent_id}",
        "split_group": f"split-{parent_id}",
        "input_fingerprint": f"fingerprint-{parent_id}",
        "charge": 0,
        "multiplicity": 1,
        "overlap_audit": "pass",
        "admission": admission,
    }


def test_scale_planner_is_durable_and_blocked_without_manifest(tmp_path: Path):
    planner = _planner_module()
    report = planner.build_plan(None)
    assert report["status"] == "blocked"
    assert report["primary_run_count"] == 0
    assert "manifest" in report["blockers"][0]


def test_scale_planner_materializes_the_frozen_81_runs(tmp_path: Path):
    planner = _planner_module()
    manifest = tmp_path / "parents.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for index in range(400):
            handle.write(json.dumps(_parent(f"train-{index}", "train")) + "\n")
        for index in range(100):
            handle.write(json.dumps(_parent(f"validation-{index}", "validation")) + "\n")
        for index in range(100):
            handle.write(json.dumps(_parent(f"test-{index}", "test")) + "\n")
    report = planner.build_plan(manifest)
    assert report["status"] == "ready"
    assert report["primary_run_count"] == 81
    assert len(report["runs"]) == 81
    assert set(report["nested_parent_subsets"]) == {"25", "100", "400"}


def test_scale_planner_rejects_target_fields_before_run_materialization(tmp_path: Path):
    planner = _planner_module()
    manifest = tmp_path / "parents.jsonl"
    row = _parent("train-1", "train")
    row["ts_geometry"] = "forbidden"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    report = planner.build_plan(manifest)
    assert report["status"] == "blocked"
    assert report["primary_run_count"] == 0
    assert "target-derived" in report["blockers"][0]
