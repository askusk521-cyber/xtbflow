#!/usr/bin/env python3
"""Run the first bounded live-xTB feedback loop on one parent and one seed.

The experiment deliberately has two matched arms:

``no_feedback``
    calls xTB on the same current snapshot and charges the call, but does not
    apply its force feedback;

``physical_feedback``
    sends the same xTB forces through the immutable snapshot/provider/merge
    contract and applies bounded geometry plus force-derived event guidance.

The downstream TS/IRC validator is invoked fail-closed with no fabricated
Hessian/path evidence.  A successful loop therefore produces an auditable
``certification=blocked`` artifact until an independent TS/IRC backend is
provided.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from importlib.util import find_spec
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

from xtbflow.calculators import MolecularSystem, XTBOracleAdapter
from xtbflow.runtime import LedgerEvent, RunLedger, StageBudget
from xtbflow.sampling.adapters import AdapterRunner, GuidanceBudget, ProviderRequest
from xtbflow.sampling.live_physical import (
    LiveXTBEventProvider,
    LiveXTBGuidanceConfig,
    LiveXTBPhysicalProvider,
    xTB_protocol_from_mapping,
)
from xtbflow.sampling.parallel_guidance import (
    GenerationSnapshot,
    GuidanceCost,
    GuidanceLane,
    MergeConfig,
    merge_guidance,
)
from xtbflow.validation.ts_search import TSValidationConfig, validate_ts_evidence


SCHEMA = "xtbflow-real-feedback-loop/v1"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_identity(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        result = subprocess.run(["git", *args], cwd=root, check=False, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_text(encoding="utf-8")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("real feedback config must be a JSON object")
    return value, raw


def _snapshot(config: Mapping[str, Any], system: MolecularSystem, seed: int) -> GenerationSnapshot:
    event_size = int(config["generation"]["event_size"])
    return GenerationSnapshot(
        parent_id=str(config["parent_id"]),
        attempt_id=f"{config['parent_id']}-seed-{seed}",
        generation_step=0,
        geometry=system.coordinates,
        event=tuple(0.0 for _ in range(event_size)),
        event_mask=tuple(True for _ in range(event_size)),
        random_seed=seed,
        metadata={
            "input_hash": system.input_hash,
            "symbols": list(system.symbols),
            "charge": system.charge,
            "multiplicity": system.multiplicity,
            "environment": dict(system.environment),
        },
        provenance={"workflow": "one-parent-one-seed-live-xtb-v1"},
    )


def _system(config: Mapping[str, Any]) -> MolecularSystem:
    fixture = config["reactant"]
    return MolecularSystem(
        symbols=tuple(fixture["symbols"]),
        coordinates=tuple(tuple(float(value) for value in row) for row in fixture["coordinates_angstrom"]),
        charge=int(fixture["charge"]),
        multiplicity=int(fixture["multiplicity"]),
        environment=dict(fixture.get("environment", {})),
        system_id=str(config["parent_id"]),
    )


def _make_backend(config: Mapping[str, Any]):
    implementation = str(config["calculator"].get("implementation", "xtb"))
    if implementation != "xtb" or find_spec("xtb") is None:
        return None, None, "BLOCKED_BACKEND"
    protocol = xTB_protocol_from_mapping(config["calculator"])
    backend = XTBOracleAdapter(protocol=protocol, implementation=implementation)
    return backend, protocol, None


def _run_arm(
    arm: str,
    *,
    config: Mapping[str, Any],
    system: MolecularSystem,
    snapshot: GenerationSnapshot,
    backend: Any | None,
    protocol: Any | None,
    blocked_reason: str | None,
    output_dir: Path,
) -> dict[str, Any]:
    budget_cfg = config["budget"]
    guidance_budget = GuidanceBudget(
        GuidanceCost(
            model_forwards=int(budget_cfg["max_model_forwards"]),
            calculator_calls=int(budget_cfg["max_calculator_calls"]),
            wall_seconds=float(budget_cfg["max_wall_seconds"]),
        )
    )
    adapter_runner = AdapterRunner(guidance_budget)
    config_obj = LiveXTBGuidanceConfig(**dict(config["guidance"]))
    provider = None
    physical_run = None
    outputs = []
    feedback = None
    if backend is not None:
        provider = LiveXTBPhysicalProvider(backend, system, protocol, config=config_obj)
        physical_request = ProviderRequest(
            snapshot=snapshot,
            lane=GuidanceLane.GEOMETRY,
            provider_revision=provider.provider_revision,
            config_fingerprint=provider.config_fingerprint,
            protocol_fingerprint=provider.protocol_fingerprint,
            feedback={"arm": arm, "feedback_visibility": "current_snapshot_only"},
        )
        physical_run = adapter_runner.run(physical_request, provider)
        feedback = provider.last_feedback
        if arm == "physical_feedback" and physical_run.output is not None and feedback is not None:
            outputs.append(physical_run.output)
            event_provider = LiveXTBEventProvider(feedback, len(snapshot.event), config=config_obj)
            event_request = ProviderRequest(
                snapshot=snapshot,
                lane=GuidanceLane.EVENT,
                provider_revision=event_provider.provider_revision,
                config_fingerprint=event_provider.config_fingerprint,
                protocol_fingerprint=event_provider.protocol_fingerprint,
                feedback={"arm": arm, "source_feedback_fingerprint": feedback.feedback_fingerprint},
                action_space={"event_update": "force_derived_bounded_delta"},
            )
            event_run = adapter_runner.run(event_request, event_provider)
            if event_run.output is not None:
                outputs.append(event_run.output)
    if arm == "physical_feedback" and outputs:
        merged = merge_guidance(
            snapshot,
            outputs,
            config=MergeConfig(
                geometry_trust_radius=float(config["merge"]["geometry_trust_radius"]),
                collision_distance=float(config["merge"]["collision_distance"]),
                event_freeze_step=int(config["merge"]["event_freeze_step"]),
            ),
            constraint_matrix=tuple(tuple(float(item) for item in row) for row in config["merge"]["constraint_matrix"]),
            event_velocity_fn=lambda _geometry, event, _snapshot: event,
        )
        final_snapshot = merged.snapshot
        merge_mapping = {"decision": merged.decision.to_mapping(), "ledger": merged.ledger.to_mapping()}
    else:
        final_snapshot = replace(snapshot, generation_step=1, snapshot_id="")
        merge_mapping = {
            "decision": {
                "applied_lanes": [],
                "rejected": [] if blocked_reason is None else [["geometry", blocked_reason]],
                "geometry_before": [list(row) for row in snapshot.geometry],
                "geometry_after": [list(row) for row in snapshot.geometry],
                "event_before": list(snapshot.event),
                "event_after": list(snapshot.event),
                "event_frozen": snapshot.event_frozen,
            },
            "ledger": {"schema": "xtbflow-guidance-ledger/v1", "entries": []},
        }

    runtime_ledger = RunLedger(
        StageBudget(
            phase=f"real_feedback_{arm}",
            max_system_evaluations=1,
            max_calculator_calls=int(budget_cfg["max_calculator_calls"]),
            max_cpu_core_hours=float(budget_cfg["max_cpu_core_hours"]),
            max_gpu_hours=float(budget_cfg.get("max_gpu_hours", 0.0)),
            max_retries_per_job=0,
            max_wall_minutes_per_job=float(budget_cfg["max_wall_minutes_per_job"]),
        ),
        ledger_id=f"real-feedback-{arm}-{snapshot.snapshot_id[:16]}",
    )
    result_status = "success" if feedback is not None else "failure"
    calculator_calls = feedback.calculator_calls if feedback is not None else 0
    error = None if feedback is not None else (blocked_reason or getattr(provider, "last_error", "physical provider did not produce feedback"))
    runtime_event = LedgerEvent(
        run_id=f"real-feedback-{snapshot.snapshot_id[:16]}",
        phase=f"real_feedback_{arm}",
        calculator="xtb_oracle",
        protocol_id=protocol.protocol_id if protocol is not None else "unavailable",
        calculator_build_hash=(feedback.calculator_build_hash if feedback is not None else "unavailable"),
        system_id=f"{snapshot.parent_id}:{arm}",
        input_hash=system.with_coordinates(snapshot.geometry).input_hash,
        operation="energy_forces",
        status=result_status,
        calculator_calls=calculator_calls,
        cpu_seconds=feedback.cpu_seconds if feedback is not None else 0.0,
        gpu_seconds=0.0,
        error=error,
    )
    runtime_ledger.append(runtime_event, persist_path=output_dir / f"ledger_{arm}.json")
    certification_config = TSValidationConfig(
        require_connectivity=True,
        max_calls=1,
        reference_protocol_id=protocol.protocol_id if protocol is not None else "unavailable",
        reference_build_hash=(feedback.calculator_build_hash if feedback is not None else "unavailable"),
    )
    certification = validate_ts_evidence(
        f"{snapshot.parent_id}:{arm}",
        "live-xtb-real-feedback",
        {},
        certification_config,
        calls=0,
    )
    return {
        "arm": arm,
        "snapshot_before": snapshot.to_mapping(),
        "snapshot_after": final_snapshot.to_mapping(),
        "geometry_changed": final_snapshot.geometry != snapshot.geometry,
        "event_changed": final_snapshot.event != snapshot.event,
        "feedback_used": bool(arm == "physical_feedback" and feedback is not None and outputs),
        "feedback": None if feedback is None else feedback.to_mapping(),
        "physical_provider_status": None if physical_run is None else physical_run.status.value,
        "physical_provider_failure_stage": None if physical_run is None else physical_run.failure_stage,
        "physical_provider_error": None if physical_run is None else physical_run.error,
        "adapter_ledger": adapter_runner.ledger.to_mapping(),
        "merge": merge_mapping,
        "runtime_ledger": runtime_ledger.summary(),
        "certification": {
            "status": "blocked",
            "reason_code": "MISSING_TS_HESSIAN_IRC_EVIDENCE",
            "record": certification.to_dict(),
            "claim_limit": "No TS/IRC certification is claimed until an independent stationary-point, mode, endpoint, and bidirectional path backend supplies evidence.",
        },
    }


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config, raw = _load_config(config_path)
    config_sha256 = _sha256_text(raw)
    seed = int(config["seed"])
    system = _system(config)
    backend, protocol, blocked_reason = _make_backend(config)
    snapshot = _snapshot(config, system, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    arms = []
    for arm in ("no_feedback", "physical_feedback"):
        arms.append(_run_arm(arm, config=config, system=system, snapshot=snapshot, backend=backend, protocol=protocol, blocked_reason=blocked_reason, output_dir=output_dir))
    certification_blocked = any(row["certification"]["status"] != "success" for row in arms)
    manifest = {
        "schema": SCHEMA,
        "status": "blocked_backend" if blocked_reason else ("blocked_certification" if certification_blocked else "completed"),
        "source": _git_identity(Path(__file__).resolve().parents[1]),
        "execution_host": platform.node(),
        "python": sys.version.split()[0],
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "input": system.payload(),
        "input_hash": system.input_hash,
        "parent_id": config["parent_id"],
        "seed": seed,
        "protocol": None if protocol is None else {"protocol_id": protocol.protocol_id, "identity": protocol.identity, "parameters": dict(protocol.parameters)},
        "arms": arms,
        "matched_budget": dict(config["budget"]),
        "claim_levels": {
            "workflow": "software_demo",
            "calculator_feedback": "proxy_metric",
            "ts_irc": "strict_validation_required",
        },
        "claim_limit": "Live xTB force feedback and bounded state transition are demonstrated; no chemical pathway, TS, barrier, mechanism, or superiority claim is made.",
        "output_paths": {
            "manifest": str(output_dir / "manifest.json"),
            "ledger": str(output_dir / "ledger.json"),
            "feedback_jsonl": str(output_dir / "feedback.jsonl"),
            "summary": str(output_dir / "summary.md"),
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    ledger_payload = {"schema": "xtbflow-real-feedback-ledger/v1", "arms": [{"arm": row["arm"], "runtime_ledger": row["runtime_ledger"], "adapter_ledger": row["adapter_ledger"], "merge_ledger": row["merge"]["ledger"]} for row in arms]}
    _write_json(output_dir / "ledger.json", ledger_payload)
    with (output_dir / "feedback.jsonl").open("w", encoding="utf-8") as handle:
        for row in arms:
            handle.write(json.dumps({"arm": row["arm"], "feedback": row["feedback"], "certification": row["certification"]}, ensure_ascii=False, sort_keys=True) + "\n")
    summary = [
        "# Real physical-feedback loop summary",
        "",
        f"status: `{manifest['status']}`",
        f"source_commit: `{manifest['source']['commit']}`",
        f"source_dirty: `{manifest['source']['dirty']}`",
        f"parent_id: `{manifest['parent_id']}` seed: `{seed}`",
        "",
        "| arm | feedback_used | geometry_changed | event_changed | calculator_calls | certification |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for row in arms:
        summary.append(f"| {row['arm']} | {row['feedback_used']} | {row['geometry_changed']} | {row['event_changed']} | {row['runtime_ledger']['totals']['calculator_calls']} | `{row['certification']['status']}` |")
    summary.extend([
        "",
        "The xTB forces are current-snapshot proxy feedback. The certification gate is intentionally blocked because this run does not fabricate Hessian, endpoint, or bidirectional IRC evidence.",
    ])
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = run(args.config, args.output_dir)
    except Exception as exc:
        print(json.dumps({"status": "failed", "reason_code": "INTERNAL_ERROR", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    print(json.dumps({"status": manifest["status"], "output_dir": str(args.output_dir), "arms": [{"arm": row["arm"], "geometry_changed": row["geometry_changed"], "event_changed": row["event_changed"], "certification": row["certification"]["status"]} for row in manifest["arms"]]}, sort_keys=True))
    return 0 if manifest["status"] in {"completed", "blocked_certification", "blocked_backend"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
