#!/usr/bin/env python3
"""Run the preregistered grouped SPICE2 Direct/Delta learning-curve matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRIC_KEYS = (
    "energy_mae_kcal_per_mol",
    "energy_rmse_kcal_per_mol",
    "energy_max_abs_kcal_per_mol",
    "relative_energy_mae_kcal_per_mol",
    "force_component_mae_hartree_per_angstrom",
    "force_component_rmse_hartree_per_angstrom",
    "force_component_p95_abs_hartree_per_angstrom",
    "force_vector_mean_angle_degrees",
    "force_vector_p95_angle_degrees",
)
PARENT_BOOTSTRAP_KEYS = (
    "energy_mae_kcal_per_mol",
    "relative_energy_mae_kcal_per_mol",
    "force_component_mae_hartree_per_angstrom",
    "force_vector_mean_angle_degrees",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compact_metrics(value: dict) -> dict:
    return {key: item for key, item in value.items() if key != "parent_metrics"}


def _mean_std(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std_across_training_seeds": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _aggregate_metrics(reports: list[dict], arm: str, split: str) -> dict:
    if arm == "bare_gfn2":
        payloads = [report["bare_gfn2"]["metrics"][split] for report in reports]
    else:
        payloads = [
            next(item for item in report["results"] if item["model_kind"] == arm)["metrics"][split]
            for report in reports
        ]
    output = {}
    for key in METRIC_KEYS:
        values = [payload[key] for payload in payloads if payload.get(key) is not None]
        if values:
            output[key] = _mean_std(values)
    return output


def _parent_map(report: dict, arm: str) -> dict[str, dict]:
    if arm == "bare_gfn2":
        rows = report["bare_gfn2"]["metrics"]["test"]["parent_metrics"]
    else:
        learned = next(item for item in report["results"] if item["model_kind"] == arm)
        rows = learned["metrics"]["test"]["parent_metrics"]
    return {row["parent_record_id"]: row for row in rows}


def _bootstrap_improvement(reports: list[dict], arm: str, *, resamples: int, seed: int) -> dict:
    bare = _parent_map(reports[0], "bare_gfn2")
    learned_maps = [_parent_map(report, arm) for report in reports]
    parent_ids = sorted(bare)
    if len(parent_ids) != 8 or any(sorted(mapping) != parent_ids for mapping in learned_maps):
        raise RuntimeError("bootstrap requires the same eight independent test parents in every seed")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(parent_ids), size=(resamples, len(parent_ids)))
    output = {}
    for key in PARENT_BOOTSTRAP_KEYS:
        bare_values = np.asarray([bare[parent][key] for parent in parent_ids], dtype=np.float64)
        learned_values = np.asarray([
            np.mean([mapping[parent][key] for mapping in learned_maps])
            for parent in parent_ids
        ], dtype=np.float64)
        difference = bare_values - learned_values
        boot = difference[indices].mean(axis=1)
        bare_mean = float(bare_values.mean())
        learned_mean = float(learned_values.mean())
        output[key] = {
            "positive_means_learned_arm_is_better": True,
            "bare_parent_equal_weight_mean": bare_mean,
            "learned_parent_equal_weight_mean_across_training_seeds": learned_mean,
            "mean_difference_bare_minus_learned": float(difference.mean()),
            "fraction_improvement": float((bare_mean - learned_mean) / bare_mean) if bare_mean else None,
            "parents_improved_fraction": float(np.mean(difference > 0)),
            "parent_bootstrap_95pct_ci_difference": [
                float(np.quantile(boot, 0.025)),
                float(np.quantile(boot, 0.975)),
            ],
            "independent_test_parents": len(parent_ids),
            "bootstrap_resamples": resamples,
        }
    return output


def _compact_run(report: dict, path: Path) -> dict:
    learned = {item["model_kind"]: item for item in report["results"]}
    return {
        "report_path": str(path),
        "report_sha256": _sha256(path),
        "seed": report["protocol"]["seed"],
        "train_parent_count": report["protocol"]["train_parent_count"],
        "selected_training_parents": report["data_identity"]["selected_training_parents"],
        "bare_gfn2": {
            "validation": _compact_metrics(report["bare_gfn2"]["metrics"]["validation"]),
            "test": _compact_metrics(report["bare_gfn2"]["metrics"]["test"]),
            "test_slices": {
                key: _compact_metrics(value)
                for key, value in report["bare_gfn2"]["test_slices"].items()
            },
        },
        "direct": {
            "best_epoch": learned["direct"]["training"]["best_epoch"],
            "training_wall_seconds": learned["direct"]["training"]["wall_seconds"],
            "validation": _compact_metrics(learned["direct"]["metrics"]["validation"]),
            "test": _compact_metrics(learned["direct"]["metrics"]["test"]),
            "test_slices": {
                key: _compact_metrics(value) for key, value in learned["direct"]["test_slices"].items()
            },
        },
        "delta": {
            "best_epoch": learned["delta"]["training"]["best_epoch"],
            "training_wall_seconds": learned["delta"]["training"]["wall_seconds"],
            "validation": _compact_metrics(learned["delta"]["metrics"]["validation"]),
            "test": _compact_metrics(learned["delta"]["metrics"]["test"]),
            "test_slices": {
                key: _compact_metrics(value) for key, value in learned["delta"]["test_slices"].items()
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema") != "xtbflow-spice2-energy-learning-curve/v1":
        parser.error("unexpected learning-curve config schema")
    pairs = ROOT / config["pairs"]["path"]
    manifest = ROOT / config["manifest"]["path"]
    if _sha256(pairs) != config["pairs"]["sha256"]:
        parser.error("pair cache hash does not match frozen config")
    if _sha256(manifest) != config["manifest"]["sha256"]:
        parser.error("manifest hash does not match frozen config")
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status.strip():
        parser.error("learning-curve execution requires a clean committed worktree")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    max_seconds = float(config["execution"]["max_single_gpu_hours"]) * 3600.0
    run_records = []
    reports_by_size: dict[int, list[dict]] = {}

    for train_parent_count in config["train_parent_counts"]:
        reports_by_size[train_parent_count] = []
        for seed in config["training_seeds"]:
            elapsed = time.perf_counter() - started
            if elapsed >= max_seconds:
                raise SystemExit(f"execution cap reached before size={train_parent_count}, seed={seed}")
            report_path = args.output_dir / f"size{train_parent_count}_seed{seed}.json"
            log_path = args.output_dir / f"size{train_parent_count}_seed{seed}.log"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_energy_baselines.py"),
                "--pairs", str(pairs),
                "--manifest", str(manifest),
                "--output", str(report_path),
                "--device", args.device,
                "--seed", str(seed),
                "--train-parent-count", str(train_parent_count),
            ]
            completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
            log_path.write_text(
                "COMMAND: " + " ".join(command) + "\n\nSTDOUT:\n" + completed.stdout
                + "\nSTDERR:\n" + completed.stderr,
                encoding="utf-8",
            )
            if completed.returncode != 0:
                raise SystemExit(f"learning-curve run failed: size={train_parent_count}, seed={seed}; see {log_path}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report["protocol"]["seed"] != seed or report["protocol"]["train_parent_count"] != train_parent_count:
                raise RuntimeError("run report identity does not match requested matrix point")
            if report["protocol"]["training_parent_rank_seed"] != config["train_parent_rank_seed"]:
                raise RuntimeError("run report parent-ranking seed does not match frozen config")
            frozen_training = config["training"]
            observed_training = report["protocol"]
            for key in ("epochs", "batch_size", "learning_rate", "force_weight", "gradient_clip"):
                if observed_training[key] != frozen_training[key]:
                    raise RuntimeError(f"run training protocol drifted for {key}")
            if observed_training["validation_score_formula"] != frozen_training["validation_score"]:
                raise RuntimeError("validation checkpoint score drifted from frozen config")
            for learned in report["results"]:
                model_config = learned["model_config"]
                if model_config["hidden_dim"] != frozen_training["hidden_dim"]:
                    raise RuntimeError("hidden_dim drifted from frozen config")
                if model_config["radial_features"] != frozen_training["radial_features"]:
                    raise RuntimeError("radial_features drifted from frozen config")
                if model_config["distance_scale"] != frozen_training["distance_scale_angstrom"]:
                    raise RuntimeError("distance_scale drifted from frozen config")
            reports_by_size[train_parent_count].append(report)
            run_records.append(_compact_run(report, report_path))
            if time.perf_counter() - started > max_seconds:
                raise SystemExit("execution cap exceeded; stopping after completed bounded run")

    nested_parent_sets = {}
    previous: set[str] = set()
    for size in config["train_parent_counts"]:
        parent_lists = [report["data_identity"]["selected_training_parents"] for report in reports_by_size[size]]
        if any(value != parent_lists[0] for value in parent_lists[1:]):
            raise RuntimeError(f"training-parent subset changed across seeds at size {size}")
        current = set(parent_lists[0])
        if previous and not previous.issubset(current):
            raise RuntimeError("training-parent subsets are not nested")
        previous = current
        nested_parent_sets[str(size)] = {
            "parents": parent_lists[0],
            "parent_list_sha256": hashlib.sha256(
                json.dumps(parent_lists[0], separators=(",", ":")).encode()
            ).hexdigest(),
        }

    aggregate = {}
    bootstrap_config = config["bootstrap"]
    for size in config["train_parent_counts"]:
        reports = reports_by_size[size]
        aggregate[str(size)] = {
            "bare_gfn2": {
                "validation": _aggregate_metrics(reports, "bare_gfn2", "validation"),
                "test": _aggregate_metrics(reports, "bare_gfn2", "test"),
            },
            "direct": {
                "validation": _aggregate_metrics(reports, "direct", "validation"),
                "test": _aggregate_metrics(reports, "direct", "test"),
                "test_parent_bootstrap_vs_bare": _bootstrap_improvement(
                    reports, "direct",
                    resamples=int(bootstrap_config["resamples"]),
                    seed=int(bootstrap_config["seed"]),
                ),
            },
            "delta": {
                "validation": _aggregate_metrics(reports, "delta", "validation"),
                "test": _aggregate_metrics(reports, "delta", "test"),
                "test_parent_bootstrap_vs_bare": _bootstrap_improvement(
                    reports, "delta",
                    resamples=int(bootstrap_config["resamples"]),
                    seed=int(bootstrap_config["seed"]),
                ),
            },
        }

    elapsed = time.perf_counter() - started
    summary = {
        "schema": "xtbflow-spice2-energy-learning-curve-result/v1",
        "status": "complete",
        "source_commit": source_commit,
        "config_path": str(args.config),
        "config_sha256": _sha256(args.config),
        "data_identity": {
            "pairs": config["pairs"],
            "manifest": config["manifest"],
            "parent_split_counts": config["parent_split_counts"],
            "configs_per_parent": config["configs_per_parent"],
        },
        "matrix": {
            "train_parent_counts": config["train_parent_counts"],
            "training_seeds": config["training_seeds"],
            "learned_arms": config["learned_arms"],
            "runs": len(run_records),
        },
        "nested_training_parent_sets": nested_parent_sets,
        "aggregate": aggregate,
        "runs": run_records,
        "execution": {
            "device": args.device,
            "wall_seconds_total": elapsed,
            "single_gpu_hours_if_one_gpu_used": elapsed / 3600.0,
            "cap_single_gpu_hours": config["execution"]["max_single_gpu_hours"],
            "new_semiempirical_calls": 0,
            "new_dft_calls": 0,
        },
        "limits": [
            "Validation selects checkpoints; test metrics are never used to change the frozen matrix.",
            "The test split contains 32 configurations but only 8 independent parent molecules.",
            "Training-seed variance and parent-bootstrap uncertainty are reported separately; seeds and configurations are not treated as independent chemical samples.",
            "This is public E/F development evidence, not reaction-discovery, transition-state, mechanism, or experimental evidence.",
        ],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(args.summary),
        "runs": len(run_records),
        "wall_seconds_total": elapsed,
        "gpu_hours": elapsed / 3600.0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
