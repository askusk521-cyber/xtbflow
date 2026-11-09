#!/usr/bin/env python3
"""Run no-op, standard force-step, and adapter-guided baselines on test parents."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from xtbflow.calculators import CalculatorProtocol, MolecularSystem, XTBOracleAdapter
from xtbflow.data.paired_loader import load_paired_pilot


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _jitter(coordinates: tuple[tuple[float, float, float], ...], seed: int, parent_id: str) -> np.ndarray:
    token = int(hashlib.sha256(f"{parent_id}:{seed}".encode()).hexdigest()[:12], 16)
    rng = random.Random(token)
    return np.asarray(
        [[value + rng.uniform(-0.002, 0.002) for value in row] for row in coordinates],
        dtype=np.float64,
    )


def _protocol(payload: dict[str, Any]) -> CalculatorProtocol:
    return CalculatorProtocol(
        protocol_id="tblite-gfn2-qualified-baseline-v1",
        calculator="xtb_oracle",
        method="GFN2-xTB",
        backend="cpu",
        parameters={
            "accuracy": payload["calculator_protocol"]["accuracy"],
            "max_iterations": payload["calculator_protocol"]["max_iterations"],
            "electronic_temperature": payload["calculator_protocol"]["electronic_temperature"],
            "solvent": payload["calculator_protocol"]["solvent"],
            "implementation": payload["calculator_protocol"]["implementation"],
        },
    )


def _system(sample: Any, coordinates: np.ndarray, parent_id: str, seed: int) -> MolecularSystem:
    symbols = tuple({1: "H", 6: "C", 7: "N", 8: "O", 16: "S"}[int(value)] for value in sample.atomic_numbers)
    return MolecularSystem(
        symbols=symbols,
        coordinates=tuple(tuple(float(value) for value in row) for row in coordinates),
        charge=int(sample.record.reactant["charge"]),
        multiplicity=int(sample.record.reactant["multiplicity"]),
        environment={"periodic": False, "confirmation_parent": parent_id, "confirmation_seed": seed},
        system_id=f"{parent_id}:confirmation-seed-{seed}",
    )


def _clip_delta(delta: np.ndarray, radius: float) -> np.ndarray:
    norm = float(np.sqrt(np.sum(np.square(delta))))
    return delta if norm <= radius or norm == 0.0 else delta * (radius / norm)


def _evaluate(adapter: XTBOracleAdapter, system: MolecularSystem) -> tuple[Any, float]:
    start = time.perf_counter()
    result = adapter.evaluate(system, operation="energy_forces")
    return result, time.perf_counter() - start


def _run_row(sample: Any, parent_id: str, seed: int, arm: str, config: dict[str, Any], adapter: XTBOracleAdapter) -> dict[str, Any]:
    coordinates = _jitter(tuple(tuple(float(value) for value in row) for row in sample.reactant_coordinates), seed, parent_id)
    system = _system(sample, coordinates, parent_id, seed)
    initial, initial_wall = _evaluate(adapter, system)
    if initial.status != "success" or initial.forces is None or initial.energy is None:
        return {"parent_id": parent_id, "seed": seed, "arm": arm, "status": "initial_failed", "calculator_calls": 1, "wall_seconds": initial_wall}
    initial_forces = np.asarray(initial.forces, dtype=np.float64)
    trust_radius = float(config["step_policy"]["trust_radius_angstrom"])
    if arm == "no_op":
        delta = np.zeros_like(coordinates)
    elif arm == "standard_force_step":
        delta = _clip_delta(initial_forces * float(config["step_policy"]["standard_step_scale"]), trust_radius)
    elif arm == "normalized_force_step":
        force_norm = float(np.sqrt(np.sum(np.square(initial_forces))))
        delta = initial_forces * (trust_radius / max(force_norm, 1.0e-12))
    else:
        raise ValueError(f"unknown arm {arm}")
    final_system = system.with_coordinates(tuple(tuple(float(value) for value in row) for row in coordinates + delta))
    final, final_wall = _evaluate(adapter, final_system)
    calls = 2
    if final.status != "success" or final.forces is None or final.energy is None:
        return {
            "parent_id": parent_id,
            "seed": seed,
            "arm": arm,
            "status": "final_failed",
            "calculator_calls": calls,
            "wall_seconds": initial_wall + final_wall,
            "geometry_displacement_angstrom": float(np.sqrt(np.sum(np.square(delta)))),
        }
    final_forces = np.asarray(final.forces, dtype=np.float64)
    return {
        "parent_id": parent_id,
        "seed": seed,
        "arm": arm,
        "status": "success",
        "calculator_calls": calls,
        "wall_seconds": initial_wall + final_wall,
        "energy_initial_hartree": float(initial.energy),
        "energy_final_hartree": float(final.energy),
        "energy_delta_hartree": float(final.energy - initial.energy),
        "force_rms_initial": _rms(initial_forces),
        "force_rms_final": _rms(final_forces),
        "force_rms_reduction": _rms(initial_forces) - _rms(final_forces),
        "geometry_displacement_angstrom": float(np.sqrt(np.sum(np.square(delta)))),
        "trust_radius_angstrom": trust_radius,
        "delta_norm_within_trust_radius": float(np.sqrt(np.sum(np.square(delta)))) <= trust_radius + 1.0e-12,
        "input_fingerprint": system.input_hash,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.protocol.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_pre_run":
        raise ValueError("baseline protocol must be frozen_pre_run")
    loader = load_paired_pilot(args.manifest, args.cache_root, batch_size=1)
    by_parent = {sample.record.parent_reaction_id: sample for sample in loader.samples}
    parent_ids = list(config["parent_ids"])
    missing = sorted(set(parent_ids) - set(by_parent))
    if missing:
        raise ValueError(f"confirmation parents missing from manifest: {missing}")
    wrong_split = sorted(parent_id for parent_id in parent_ids if by_parent[parent_id].record.admission != "development_test")
    if wrong_split:
        raise ValueError(f"confirmation parents are not in development_test: {wrong_split}")
    adapter = XTBOracleAdapter(protocol=_protocol(config), implementation="tblite")
    rows = []
    start = time.perf_counter()
    for parent_id in parent_ids:
        for seed in config["seeds"]:
            for arm in config["arms"]:
                rows.append(_run_row(by_parent[parent_id], parent_id, int(seed), arm, config, adapter))
    elapsed = time.perf_counter() - start
    calls = sum(int(row.get("calculator_calls", 0)) for row in rows)
    report = {
        "schema": "xtbflow-qualified-baseline-confirmation-result/v1",
        "status": "completed" if all(row["status"] == "success" for row in rows) else "completed_with_failures",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": _sha256(args.manifest),
        "protocol": config,
        "parent_count": len(parent_ids),
        "seed_count": len(config["seeds"]),
        "row_count": len(rows),
        "calculator_calls": calls,
        "elapsed_wall_seconds": elapsed,
        "rows": rows,
        "claim_limits": config["claim_limits"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "rows": len(rows), "calculator_calls": calls}, sort_keys=True))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
