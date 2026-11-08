#!/usr/bin/env python3
"""Materialize the frozen architecture/data scale matrix without training.

The planner consumes an identity-only parent manifest.  It refuses rows that
are quarantined or that expose product/transition-state fields, then writes
the 81 bounded run records only when the admission gate is complete.  With no
manifest it emits a durable blocked plan, which is useful for CI and for
recording why a GPU submission was not made.

The parent-row shape is documented in ``schemas/joint_scale_parent.schema.json``;
this planner performs the additional result-blind and split-count checks that a
JSON Schema cannot express.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SCALE_CONFIG = ROOT / "configs/models/joint_flow_scale_ladder_v0.1.json"
MATRIX_CONFIG = ROOT / "configs/runs/joint_scale_data_matrix_v0.1.json"
FORBIDDEN_KEYS = frozenset(
    {"product", "event", "ts", "transition_state", "reference", "target", "sealed", "label"}
)
REQUIRED_PARENT_FIELDS = (
    "parent_reaction_id",
    "family_id",
    "split_group",
    "input_fingerprint",
    "charge",
    "multiplicity",
    "overlap_audit",
    "admission",
)


def _nested_parent_subsets(parent_ids: list[str], sizes: list[int], *, seed: str) -> dict[int, tuple[str, ...]]:
    """Local stdlib-only copy of the result-blind parent ordering contract."""

    if not parent_ids or len(set(parent_ids)) != len(parent_ids):
        raise ValueError("training parent identities must be unique and nonempty")
    if not sizes or sizes != sorted(set(sizes)):
        raise ValueError("training parent sizes must be strictly increasing")
    if sizes[-1] > len(parent_ids):
        raise ValueError("largest requested size exceeds available train parents")
    ordered = sorted(parent_ids, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest())
    return {size: tuple(ordered[:size]) for size in sizes}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_identity() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {"commit": commit}


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain an object")
    return value


def _contains_forbidden(value: Any, path: str = "root") -> str | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                return f"{path} has a non-string key"
            if any(token in key.lower() for token in FORBIDDEN_KEYS):
                return f"{path}.{key} is a target-derived field"
            reason = _contains_forbidden(nested, f"{path}.{key}")
            if reason:
                return reason
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            reason = _contains_forbidden(nested, f"{path}[{index}]")
            if reason:
                return reason
    return None


def _load_parent_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    seen_parent: dict[str, tuple[str, str]] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            reasons.append(f"line {line_number}: invalid JSON ({exc.msg})")
            continue
        if not isinstance(value, dict):
            reasons.append(f"line {line_number}: row must be an object")
            continue
        forbidden = _contains_forbidden(value)
        if forbidden:
            reasons.append(f"line {line_number}: {forbidden}")
            continue
        missing = [field for field in REQUIRED_PARENT_FIELDS if field not in value]
        if missing:
            reasons.append(f"line {line_number}: missing required fields {missing}")
            continue
        parent = value["parent_reaction_id"]
        family = value["family_id"]
        split_group = value["split_group"]
        if any(not isinstance(item, str) or not item.strip() for item in (parent, family, split_group)):
            reasons.append(f"line {line_number}: parent, family, and split_group must be nonempty strings")
            continue
        if type(value["charge"]) is not int or type(value["multiplicity"]) is not int:
            reasons.append(f"line {line_number}: charge and multiplicity must be explicit integers")
            continue
        if value["admission"] not in {"train", "validation", "test"}:
            reasons.append(f"line {line_number}: admission must be train, validation, or test")
            continue
        if value["overlap_audit"] != "pass":
            reasons.append(f"line {line_number}: overlap_audit is not pass")
            continue
        previous = seen_parent.get(parent)
        identity = (family, split_group)
        if previous is not None and previous != identity:
            reasons.append(f"line {line_number}: parent {parent!r} has inconsistent family/split identity")
            continue
        seen_parent[parent] = identity
        rows.append(value)
    return rows, reasons


def build_plan(parent_path: Path | None) -> dict[str, Any]:
    scale = _load_json(SCALE_CONFIG)
    matrix = _load_json(MATRIX_CONFIG)
    status = "blocked"
    reasons: list[str] = []
    rows: list[dict[str, Any]] = []
    manifest_hash: str | None = None
    parent_subsets: dict[int, tuple[str, ...]] = {}
    if parent_path is None:
        reasons.append("no admitted sample-level parent manifest was supplied")
    elif not parent_path.is_file():
        reasons.append(f"parent manifest does not exist: {parent_path}")
    else:
        manifest_hash = _sha256(parent_path)
        rows, reasons = _load_parent_rows(parent_path)
        by_parent = {row["parent_reaction_id"]: row for row in rows}
        if len(by_parent) != len(rows):
            reasons.append("parent manifest contains duplicate parent rows")
        requested = [int(item["train_parent_count"]) for item in matrix["data_scales"]]
        try:
            train_parent_ids = [
                parent for parent, row in by_parent.items() if row["admission"] == "train"
            ]
            parent_subsets = _nested_parent_subsets(
                train_parent_ids,
                requested,
                seed=str(matrix["nested_parent_selection"]["ranking_seed"]),
            )
        except ValueError as exc:
            reasons.append(str(exc))
        validation_target = int(matrix["nested_parent_selection"]["validation_parent_target"][0])
        test_target = int(matrix["nested_parent_selection"]["test_parent_target"][0])
        if sum(row["admission"] == "validation" for row in by_parent.values()) < validation_target:
            reasons.append("validation parent count is below the frozen minimum")
        if sum(row["admission"] == "test" for row in by_parent.values()) < test_target:
            reasons.append("test parent count is below the frozen minimum")
        if not reasons:
            status = "ready"

    architectures = matrix["primary_architectures"]
    scales = {item["scale_id"]: item for item in scale["scales"]}
    primary_scale_ids = matrix["model_scales"]["primary_scale_ids"]
    seeds = matrix["training"]["seeds"]
    run_records: list[dict[str, Any]] = []
    if status == "ready":
        for architecture in architectures:
            for scale_id in primary_scale_ids:
                for data_scale in matrix["data_scales"]:
                    for seed in seeds:
                        run_records.append(
                            {
                                "run_id": f"{architecture['architecture_id']}__{scale_id}__{data_scale['data_scale_id']}__seed{seed}",
                                "architecture_id": architecture["architecture_id"],
                                "control_mode": architecture["control_mode"],
                                "scale_id": scale_id,
                                "hidden_dim": scales[scale_id]["hidden_dim"],
                                "trainable_parameters": scales[scale_id]["expected_trainable_parameters"],
                                "data_scale_id": data_scale["data_scale_id"],
                                "train_parent_count": data_scale["train_parent_count"],
                                "seed": seed,
                                "status": "planned",
                            }
                        )
    return {
        "schema": "xtbflow-joint-scale-run-plan/v1",
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_host": platform.node(),
        "git": _git_identity(),
        "configuration": {
            "scale_ladder_sha256": _sha256(SCALE_CONFIG),
            "matrix_sha256": _sha256(MATRIX_CONFIG),
            "parent_manifest_sha256": manifest_hash,
        },
        "parent_row_count": len(rows),
        "parent_count": len({row.get("parent_reaction_id") for row in rows}),
        "nested_parent_subsets": {str(size): list(values) for size, values in parent_subsets.items()},
        "blockers": reasons,
        "primary_run_count": len(run_records),
        "runs": run_records,
        "claim_limit": "planning and admission evidence only; no model was trained and no chemistry result is implied",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parents", type=Path, default=None, help="identity-only admitted parent JSONL")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    report = build_plan(args.parents)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "primary_run_count": report["primary_run_count"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
