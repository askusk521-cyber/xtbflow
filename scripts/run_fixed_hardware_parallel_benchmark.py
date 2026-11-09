#!/usr/bin/env python3
"""Compare serial and threaded execution on one fixed n2 CPU workload."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
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


def _jitter(coordinates: tuple[tuple[float, float, float], ...], seed: int, parent_id: str) -> np.ndarray:
    token = int(hashlib.sha256(f"{parent_id}:{seed}".encode()).hexdigest()[:12], 16)
    rng = random.Random(token)
    return np.asarray([[value + rng.uniform(-0.002, 0.002) for value in row] for row in coordinates], dtype=np.float64)


def _system(sample: Any, coordinates: np.ndarray, parent_id: str, seed: int) -> MolecularSystem:
    symbols = tuple({1: "H", 6: "C", 7: "N", 8: "O", 16: "S"}[int(value)] for value in sample.atomic_numbers)
    return MolecularSystem(
        symbols=symbols,
        coordinates=tuple(tuple(float(value) for value in row) for row in coordinates),
        charge=int(sample.record.reactant["charge"]),
        multiplicity=int(sample.record.reactant["multiplicity"]),
        environment={"periodic": False, "benchmark_parent": parent_id, "benchmark_seed": seed},
        system_id=f"{parent_id}:benchmark-seed-{seed}",
    )


def _task(sample: Any, parent_id: str, seed: int, step_scale: float, trust_radius: float) -> dict[str, Any]:
    coordinates = _jitter(tuple(tuple(float(value) for value in row) for row in sample.reactant_coordinates), seed, parent_id)
    system = _system(sample, coordinates, parent_id, seed)
    adapter = XTBOracleAdapter(
        protocol=CalculatorProtocol(
            protocol_id="tblite-gfn2-fixed-hardware-benchmark-v1",
            calculator="xtb_oracle",
            method="GFN2-xTB",
            backend="cpu",
            parameters={"accuracy": 1.0, "max_iterations": 250, "electronic_temperature": 300.0, "solvent": None, "implementation": "tblite"},
        ),
        implementation="tblite",
    )
    start = time.perf_counter()
    initial = adapter.evaluate(system, operation="energy_forces")
    if initial.status != "success" or initial.forces is None:
        return {"parent_id": parent_id, "seed": seed, "status": "initial_failed", "calculator_calls": 1, "wall_seconds": time.perf_counter() - start}
    forces = np.asarray(initial.forces, dtype=np.float64)
    delta = forces * step_scale
    norm = float(np.sqrt(np.sum(np.square(delta))))
    if norm > trust_radius:
        delta *= trust_radius / norm
    final = adapter.evaluate(system.with_coordinates(tuple(tuple(float(value) for value in row) for row in coordinates + delta)), operation="energy_forces")
    elapsed = time.perf_counter() - start
    return {
        "parent_id": parent_id,
        "seed": seed,
        "status": "success" if final.status == "success" else "final_failed",
        "calculator_calls": 2,
        "wall_seconds": elapsed,
        "geometry_displacement_angstrom": float(np.sqrt(np.sum(np.square(delta)))),
    }


def _summary(rows: list[dict[str, Any]], elapsed: float, workers: int) -> dict[str, Any]:
    walls = [float(row["wall_seconds"]) for row in rows]
    return {
        "tasks": len(rows),
        "success_tasks": sum(row.get("status") == "success" for row in rows),
        "calculator_calls": sum(int(row.get("calculator_calls", 0)) for row in rows),
        "elapsed_wall_seconds": elapsed,
        "worker_count": workers,
        "p50_task_wall_seconds": statistics.median(walls) if walls else 0.0,
        "p95_task_wall_seconds": sorted(walls)[int(0.95 * (len(walls) - 1))] if walls else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.protocol.read_text(encoding="utf-8"))
    loader = load_paired_pilot(args.manifest, args.cache_root, batch_size=1)
    by_parent = {sample.record.parent_reaction_id: sample for sample in loader.samples}
    parent_ids = list(config["parent_ids"])
    if any(parent_id not in by_parent for parent_id in parent_ids):
        raise ValueError("benchmark parent missing from manifest")
    if any(by_parent[parent_id].record.admission != "development_test" for parent_id in parent_ids):
        raise ValueError("benchmark parents must all be development_test")
    tasks = [(parent_id, int(seed), repeat) for repeat in range(int(config["repeats"])) for parent_id in parent_ids for seed in config["seeds"]]
    step_scale = float(config["task"]["step_scale"])
    trust_radius = float(config["task"]["trust_radius_angstrom"])
    modes: dict[str, dict[str, Any]] = {}
    for mode in ("serial", "parallel"):
        start = time.perf_counter()
        if mode == "serial":
            rows = []
            for parent_id, seed, repeat in tasks:
                row = _task(by_parent[parent_id], parent_id, seed, step_scale, trust_radius)
                row["repeat"] = repeat
                rows.append(row)
        else:
            rows = []
            with ThreadPoolExecutor(max_workers=int(config["workers"])) as pool:
                futures = [(parent_id, seed, repeat, pool.submit(_task, by_parent[parent_id], parent_id, seed, step_scale, trust_radius)) for parent_id, seed, repeat in tasks]
                for parent_id, seed, repeat, future in futures:
                    row = future.result()
                    row["repeat"] = repeat
                    rows.append(row)
        elapsed = time.perf_counter() - start
        modes[mode] = {"summary": _summary(rows, elapsed, 1 if mode == "serial" else int(config["workers"])), "rows": rows}
    serial_elapsed = modes["serial"]["summary"]["elapsed_wall_seconds"]
    parallel_elapsed = modes["parallel"]["summary"]["elapsed_wall_seconds"]
    report = {
        "schema": "xtbflow-fixed-hardware-parallel-benchmark-result/v1",
        "status": "completed",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": _sha256(args.manifest),
        "protocol": config,
        "runtime": {"host": platform.node(), "python": platform.python_version(), "cpu_count": __import__("os").cpu_count()},
        "modes": modes,
        "comparison": {"parallel_over_serial_wall_ratio": parallel_elapsed / serial_elapsed if serial_elapsed else None, "serial_over_parallel_speedup": serial_elapsed / parallel_elapsed if parallel_elapsed else None},
        "claim_limits": config["claim_limits"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "serial": serial_elapsed, "parallel": parallel_elapsed, "speedup": report["comparison"]["serial_over_parallel_speedup"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
