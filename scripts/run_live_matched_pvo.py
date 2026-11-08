#!/usr/bin/env python3
"""Run the bounded live matched PVO against the frozen paired-pilot parents.

This entry point deliberately keeps the result diagnostic: the tblite protocol
is executed for real on n2, but its strict native/independent parity
qualification remains incomplete.  No result is promoted to a chemistry claim.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any, Mapping

from xtbflow.calculators import CalculatorProtocol, MolecularSystem, XTBOracleAdapter
from xtbflow.data.paired_loader import load_paired_pilot
from xtbflow.sampling import GuidanceBudget, GuidanceCost, GuidanceLane, GuidanceOutput, GenerationSnapshot, MergeConfig, ProviderRequest, AdapterRunner, merge_guidance, fingerprint_text


ARMS = ("none", "physical_only", "policy_only", "serial_full", "parallel_full", "policy_off")
BOHR_PROTOCOL = {
    "accuracy": 1.0,
    "max_iterations": 250,
    "electronic_temperature": 300.0,
    "solvent": None,
    "implementation": "tblite",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else 0.0


def _distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _covalent_radius(atomic_number: int) -> float:
    return {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 16: 1.05}.get(int(atomic_number), 0.8)


def bond_set(matrix: Any, map_ids: tuple[int, ...]) -> set[tuple[str, int, int]]:
    values = matrix.tolist() if hasattr(matrix, "tolist") else matrix
    result: set[tuple[str, int, int]] = set()
    for index in range(len(map_ids)):
        for other in range(index + 1, len(map_ids)):
            if float(values[index][other]) > 0.5:
                result.add(("bond", min(map_ids[index], map_ids[other]), max(map_ids[index], map_ids[other])))
    return result


def reference_event(sample: Any) -> set[tuple[str, int, int]]:
    reactant = bond_set(sample.reactant_bonds, tuple(sample.record.atoms["map_ids"]))
    product = bond_set(sample.product_bonds, tuple(sample.record.atoms["map_ids"]))
    return {("add", a, b) for _, a, b in product - reactant} | {("delete", a, b) for _, a, b in reactant - product}


def deterministic_jitter(coordinates: tuple[tuple[float, float, float], ...], seed: int, parent_id: str) -> tuple[tuple[float, float, float], ...]:
    token = int(hashlib.sha256(f"{parent_id}:{seed}".encode()).hexdigest()[:12], 16)
    rng = random.Random(token)
    return tuple(tuple(value + rng.uniform(-0.002, 0.002) for value in row) for row in coordinates)


def make_system(sample: Any, coordinates: tuple[tuple[float, float, float], ...], parent_id: str, seed: int) -> MolecularSystem:
    symbols = tuple({1: "H", 6: "C", 7: "N", 8: "O", 16: "S"}[int(value)] for value in sample.atomic_numbers)
    return MolecularSystem(
        symbols=symbols,
        coordinates=coordinates,
        charge=int(sample.record.reactant["charge"]),
        multiplicity=int(sample.record.reactant["multiplicity"]),
        environment={"periodic": False, "pvo_parent": parent_id, "pvo_seed": seed},
        system_id=f"{parent_id}:seed-{seed}",
    )


class LivePhysicalProvider:
    lane = GuidanceLane.GEOMETRY
    provider_revision = fingerprint_text("live-tblite-physical-pvo-v1")
    config_fingerprint = fingerprint_text(json.dumps(BOHR_PROTOCOL, sort_keys=True))
    protocol_fingerprint = fingerprint_text("tblite-gfn2-live-pvo-v1")

    def __init__(self, system: MolecularSystem, adapter_factory):
        self.system = system
        self.adapter_factory = adapter_factory
        self.last_result = None

    def estimate(self, request: ProviderRequest) -> GuidanceCost:
        return GuidanceCost(calculator_calls=1, wall_seconds=10.0)

    def invoke(self, request: ProviderRequest) -> GuidanceOutput:
        adapter = self.adapter_factory()
        result = adapter.evaluate(self.system, operation="energy_forces")
        self.last_result = result
        if result.status != "success" or result.forces is None:
            raise RuntimeError(result.error_message or result.error_category or "physical calculator failed")
        forces = result.forces
        norm = rms([component for row in forces for component in row])
        scale = min(0.05, 0.05 / max(norm, 1e-12))
        delta = tuple(tuple(float(component) * scale for component in row) for row in forces)
        return GuidanceOutput(
            lane=self.lane,
            provider="live-tblite-physical-pvo-v1",
            input_fingerprint=request.snapshot.snapshot_id,
            geometry_delta=delta,
            trust_radius=0.05,
            estimated_cost=self.estimate(request),
            measured_cost=GuidanceCost(calculator_calls=result.calculator_calls, wall_seconds=0.0),
            provenance={"physical": {"calculator": "tblite", "protocol": self.protocol_fingerprint}},
        )


class LivePolicyProvider:
    lane = GuidanceLane.EVENT
    provider_revision = fingerprint_text("live-heuristic-policy-pvo-v1")
    config_fingerprint = fingerprint_text("nearest-nonbonded-pair-policy-v1")
    protocol_fingerprint = fingerprint_text("recorded-policy-action-space-v1")

    def __init__(self, coordinates, atomic_numbers, map_ids):
        self.coordinates = coordinates
        self.atomic_numbers = atomic_numbers
        self.map_ids = map_ids
        self.action_id = "no-action"
        self.predicted_event: set[tuple[str, int, int]] = set()

    def estimate(self, request: ProviderRequest) -> GuidanceCost:
        return GuidanceCost(model_forwards=1, wall_seconds=0.01)

    def invoke(self, request: ProviderRequest) -> GuidanceOutput:
        candidates: list[tuple[float, int, int]] = []
        for index in range(len(self.coordinates)):
            for other in range(index + 1, len(self.coordinates)):
                distance = _distance(self.coordinates[index], self.coordinates[other])
                bonded_cutoff = 1.25 * (_covalent_radius(self.atomic_numbers[index]) + _covalent_radius(self.atomic_numbers[other]))
                if bonded_cutoff < distance < 2.2:
                    candidates.append((distance, index, other))
        candidates.sort()
        if candidates:
            _, left, right = candidates[0]
            a, b = sorted((self.map_ids[left], self.map_ids[right]))
            self.action_id = f"add:{a}-{b}"
            self.predicted_event = {("add", a, b)}
        else:
            self.action_id = "no-action"
            self.predicted_event = set()
        return GuidanceOutput(
            lane=self.lane,
            provider="live-heuristic-policy-pvo-v1",
            input_fingerprint=request.snapshot.snapshot_id,
            event_delta=(0.05,),
            trust_radius=0.2,
            estimated_cost=self.estimate(request),
            measured_cost=self.estimate(request),
            provenance={"policy": {"action_id": self.action_id, "policy_revision": self.provider_revision}},
        )


def calculator_protocol() -> CalculatorProtocol:
    return CalculatorProtocol(
        protocol_id="tblite-gfn2-live-pvo-v1",
        calculator="xtb_oracle",
        method="GFN2-xTB",
        backend="cpu",
        parameters=BOHR_PROTOCOL,
    )


def evaluate(adapter: XTBOracleAdapter, system: MolecularSystem) -> tuple[Any, float]:
    start = time.perf_counter()
    result = adapter.evaluate(system, operation="energy_forces")
    return result, time.perf_counter() - start


def run_row(sample: Any, parent_id: str, seed: int, arm: str, protocol: Mapping[str, Any]) -> dict[str, Any]:
    coordinates = deterministic_jitter(tuple(tuple(float(v) for v in row) for row in sample.reactant_coordinates), seed, parent_id)
    system = make_system(sample, coordinates, parent_id, seed)
    initial_snapshot = GenerationSnapshot(
        parent_id=parent_id,
        attempt_id=f"{parent_id}:seed-{seed}",
        generation_step=0,
        geometry=coordinates,
        event=(0.0,),
        event_mask=(True,),
        random_seed=seed,
        metadata={"candidate_pool_id": protocol["source_manifest_sha256"]},
        provenance={"pvo": "live-matched-v1"},
    )
    reference = reference_event(sample)
    adapter_factory = lambda: XTBOracleAdapter(protocol=calculator_protocol(), implementation="tblite")
    initial_adapter = adapter_factory()
    initial_result, initial_wall = evaluate(initial_adapter, system)
    calculator_calls = int(initial_result.calculator_calls)
    final_system = system
    predicted_event: set[tuple[str, int, int]] = set()
    provider_wall = 0.0
    queue_wait = 0.0
    merge_wall = 0.0
    rows_ledger: list[dict[str, Any]] = []
    if initial_result.status != "success" or initial_result.forces is None:
        return {"parent_id": parent_id, "seed": seed, "arm": arm, "status": "initial_calculator_failed", "calculator_calls": calculator_calls, "wall_seconds": initial_wall, "reference_event": sorted(reference)}
    budget = GuidanceBudget(GuidanceCost(model_forwards=2, calculator_calls=3, wall_seconds=120.0))
    physical = LivePhysicalProvider(system, adapter_factory)
    policy = LivePolicyProvider(coordinates, tuple(sample.atomic_numbers), tuple(sample.record.atoms["map_ids"]))

    def run_physical():
        runner = AdapterRunner(budget)
        request = ProviderRequest(snapshot=initial_snapshot, lane=GuidanceLane.GEOMETRY, provider_revision=physical.provider_revision, config_fingerprint=physical.config_fingerprint, protocol_fingerprint=physical.protocol_fingerprint, feedback={"force_feedback": "current_snapshot"})
        start = time.perf_counter(); run = runner.run(request, physical); return run, runner.ledger, time.perf_counter() - start

    def run_policy(current_snapshot):
        runner = AdapterRunner(budget)
        request = ProviderRequest(snapshot=current_snapshot, lane=GuidanceLane.EVENT, provider_revision=policy.provider_revision, config_fingerprint=policy.config_fingerprint, protocol_fingerprint=policy.protocol_fingerprint, feedback={"current_snapshot": current_snapshot.snapshot_id}, action_space={"action_id": "nearest-nonbonded-pair"})
        start = time.perf_counter(); run = runner.run(request, policy); return run, runner.ledger, time.perf_counter() - start

    physical_run = None
    policy_run = None
    if arm in {"physical_only", "serial_full", "parallel_full"}:
        if arm == "parallel_full":
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=2) as pool:
                physical_future = pool.submit(run_physical)
                policy_future = pool.submit(run_policy, initial_snapshot)
                physical_run, physical_ledger, physical_time = physical_future.result()
                policy_run, policy_ledger, policy_time = policy_future.result()
            provider_wall = time.perf_counter() - start
            rows_ledger.extend([entry.to_mapping() for entry in physical_ledger.entries + policy_ledger.entries])
            outputs = tuple(run.output for run in (physical_run, policy_run) if run.output is not None)
        else:
            physical_run, physical_ledger, physical_time = run_physical()
            rows_ledger.extend([entry.to_mapping() for entry in physical_ledger.entries])
            outputs = tuple(run.output for run in (physical_run,) if run.output is not None)
            provider_wall += physical_time
            if arm == "serial_full":
                merge_start = time.perf_counter()
                first_merge = merge_guidance(initial_snapshot, outputs, constraint_matrix=((1.0,),), config=MergeConfig(geometry_trust_radius=0.2, collision_distance=0.2), event_velocity_fn=lambda g,e,_: (g[0][0] + e[0],))
                merge_wall += time.perf_counter() - merge_start
                policy_run, policy_ledger, policy_time = run_policy(first_merge.snapshot)
                rows_ledger.extend([entry.to_mapping() for entry in policy_ledger.entries])
                provider_wall += policy_time
                outputs = tuple(run.output for run in (policy_run,) if run.output is not None)
                initial_snapshot = first_merge.snapshot
            elif arm == "physical_only":
                pass
    elif arm in {"policy_only"}:
        policy_run, policy_ledger, policy_time = run_policy(initial_snapshot)
        rows_ledger.extend([entry.to_mapping() for entry in policy_ledger.entries])
        outputs = tuple(run.output for run in (policy_run,) if run.output is not None)
        provider_wall += policy_time
    else:
        outputs = ()

    merge_start = time.perf_counter()
    merged = merge_guidance(initial_snapshot, outputs, constraint_matrix=((1.0,),), config=MergeConfig(geometry_trust_radius=0.2, collision_distance=0.2), event_velocity_fn=lambda g,e,_: (g[0][0] + e[0],))
    merge_wall += time.perf_counter() - merge_start
    final_system = system.with_coordinates(merged.snapshot.geometry)
    final_adapter = adapter_factory()
    final_result, final_wall = evaluate(final_adapter, final_system)
    calculator_calls += int(final_result.calculator_calls)
    if policy is not None and policy.predicted_event:
        predicted_event = set(policy.predicted_event)
    force_initial = rms([float(value) for row in initial_result.forces for value in row])
    force_final = rms([float(value) for row in final_result.forces for value in final_result.forces for value in []]) if False else None
    if final_result.forces is not None:
        force_final = rms([float(value) for row in final_result.forces for value in row])
    return {
        "parent_id": parent_id,
        "seed": seed,
        "arm": arm,
        "status": "success" if final_result.status == "success" else "final_calculator_failed",
        "snapshot_id": initial_snapshot.snapshot_id,
        "reference_event": sorted(reference),
        "predicted_event": sorted(predicted_event),
        "event_exact_match": predicted_event == reference,
        "geometry_changed": merged.snapshot.geometry != initial_snapshot.geometry,
        "event_changed": merged.snapshot.event != initial_snapshot.event,
        "energy_initial_hartree": initial_result.energy,
        "energy_final_hartree": final_result.energy,
        "force_rms_initial": force_initial,
        "force_rms_final": force_final,
        "force_rms_reduction": None if force_final is None else force_initial - force_final,
        "calculator_calls": calculator_calls,
        "provider_wall_seconds": provider_wall,
        "queue_wait_seconds": queue_wait,
        "merge_wall_seconds": merge_wall,
        "wall_seconds": initial_wall + provider_wall + merge_wall + final_wall,
        "adapter_ledger": rows_ledger,
        "qualification_status": "requalification_required",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parents", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=None)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_pre_run":
        raise RuntimeError("PVO protocol is not frozen_pre_run")
    loader = load_paired_pilot(args.manifest, args.cache_root, batch_size=1)
    by_parent = {}
    for sample in loader.samples:
        by_parent.setdefault(sample.record.parent_reaction_id, sample)
    parent_ids = list(protocol["primary_parent_ids"])
    if args.parents is not None:
        parent_ids = parent_ids[: args.parents]
    seeds = args.seeds or protocol["seeds"]
    arms = args.arms or protocol["arms"]
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for parent_id in parent_ids:
        sample = by_parent[parent_id]
        for seed in seeds:
            for arm in arms:
                rows.append(run_row(sample, parent_id, seed, arm, protocol))
    elapsed = time.perf_counter() - start
    summary = {
        "schema": "xtbflow-live-matched-pvo-result/v1",
        "status": "completed_with_qualification_limit",
        "protocol": protocol,
        "source_manifest_sha256": protocol["source_manifest_sha256"],
        "source_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip(),
        "row_count": len(rows),
        "parent_count": len(parent_ids),
        "seed_count": len(seeds),
        "arms": arms,
        "elapsed_wall_seconds": elapsed,
        "rows": rows,
        "claim_limits": protocol["claim_limits"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status":summary["status"],"rows":len(rows),"parents":len(parent_ids),"seeds":len(seeds),"elapsed_wall_seconds":elapsed},sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
