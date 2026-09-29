#!/usr/bin/env python3
"""Run a bounded shared-control joint-flow software smoke test.

The target is deterministic, nonzero, time dependent, and synthetic.  This
checks train/save/restore/sample wiring and the registered control paths; it is
not event/TS supervision and produces no chemical performance claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import torch

from xtbflow.models import JointFlowRuntimeConfig, pack_be, total_electron_projector
from xtbflow.sampling import control_euler_step
from xtbflow.training import (
    coupled_control_manifest,
    joint_flow_loss,
    load_joint_checkpoint,
    model_state_identity,
    save_joint_checkpoint,
)


SYMBOLS = ("C", "O", "H")


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().item())


def _config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_identity() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    entries = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(entries), "dirty_entries": entries}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/models/joint_flow_runtime_v0.1.json"),
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--training-steps", type=int, default=8)
    parser.add_argument("--sample-steps", type=int, default=4)
    parser.add_argument("--dt", type=float, default=None)
    args = parser.parse_args()
    if args.training_steps < 1 or args.sample_steps < 1:
        raise SystemExit("training and sample steps must be positive")
    dt = 1.0 / args.sample_steps if args.dt is None else args.dt
    if not 0 < dt <= 1 or args.sample_steps * dt > 1 + 1e-12:
        raise SystemExit("dt must be positive and sample_steps*dt must be at most one")

    started = time.monotonic()
    torch.manual_seed(args.seed)
    dtype = torch.float64
    runtime = JointFlowRuntimeConfig.load(args.config)
    projector = total_electron_projector(SYMBOLS, dtype=dtype)
    event_matrix = torch.tensor(
        [[[4.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=dtype,
    )
    event = pack_be(event_matrix)
    coordinates = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.8, 0.1, 0.0], [-0.2, 0.7, 0.1]]],
        dtype=dtype,
    )
    node_features = torch.randn(1, len(SYMBOLS), runtime.node_feature_dim, dtype=dtype)
    atom_mask = torch.ones(1, len(SYMBOLS), dtype=torch.bool)
    condition_features = torch.randn(1, runtime.condition_feature_dim, dtype=dtype)

    model = runtime.build(projector).to(dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    tau_schedule = (0.2, 0.5, 0.8)
    losses: list[float] = []
    raw_event_target = torch.tensor([[0.3, -0.2, 0.1, -0.1, 0.15, -0.05]], dtype=dtype)
    for step in range(args.training_steps):
        tau = tau_schedule[step % len(tau_schedule)]
        target_event = projector.project(raw_event_target * tau)
        target_geometry = torch.zeros_like(coordinates)
        target_geometry[:, 1, 0] = 0.15 * tau
        target_geometry[:, 2, 1] = -0.1 * tau
        optimizer.zero_grad(set_to_none=True)
        output = model.forward_control(
            "joint_bidirectional",
            event,
            coordinates,
            node_features,
            atom_mask,
            tau=tau,
            dt=dt,
            condition_features=condition_features,
        )
        loss = joint_flow_loss(
            output,
            target_event,
            target_geometry,
            atom_mask=atom_mask,
            require_nonzero_target=True,
        )["total"]
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    model.eval()
    modes = runtime.control_modes
    with torch.no_grad():
        pre_save = {
            mode: model.forward_control(
                mode,
                event,
                coordinates,
                node_features,
                atom_mask,
                tau=0.5,
                dt=dt,
                condition_features=condition_features,
            )
            for mode in modes
        }
    identity_before = model_state_identity(model)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_joint_checkpoint(
        args.checkpoint,
        model,
        optimizer,
        loop={
            "global_step": args.training_steps,
            "tau_schedule": list(tau_schedule),
            "dt": dt,
            "data_order": list(range(args.training_steps)),
        },
        sampler_state={
            "event_state": event.clone(),
            "coordinates": coordinates.clone(),
            "node_features": node_features.clone(),
            "atom_mask": atom_mask.clone(),
            "condition_features": condition_features.clone(),
        },
        metadata={
            "target": "diagnostic_nonzero_time_dependent_synthetic_velocity",
            "runtime_config_sha256": _config_sha256(args.config),
        },
    )

    restored = runtime.build(projector).to(dtype=dtype)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-2)
    restore_info = load_joint_checkpoint(args.checkpoint, restored, restored_optimizer)
    restored.eval()
    with torch.no_grad():
        post_restore = {
            mode: restored.forward_control(
                mode,
                event,
                coordinates,
                node_features,
                atom_mask,
                tau=0.5,
                dt=dt,
                condition_features=condition_features,
            )
            for mode in modes
        }
    restore_differences: dict[str, float] = {}
    for mode in modes:
        restore_differences[f"{mode}.event_velocity"] = _max_abs(
            pre_save[mode].event_velocity, post_restore[mode].event_velocity
        )
        restore_differences[f"{mode}.geometry_velocity"] = _max_abs(
            pre_save[mode].geometry_velocity, post_restore[mode].geometry_velocity
        )
        restore_differences[f"{mode}.event_message"] = _max_abs(
            pre_save[mode].event_message, post_restore[mode].event_message
        )
        restore_differences[f"{mode}.geometry_message"] = _max_abs(
            pre_save[mode].geometry_message, post_restore[mode].geometry_message
        )

    rollouts: dict[str, dict[str, object]] = {}
    with torch.no_grad():
        for mode in modes:
            sample_event = event.clone()
            sample_coordinates = coordinates.clone()
            residuals: list[float] = []
            for step in range(args.sample_steps):
                tau = step * dt
                sample_event, sample_coordinates, sample_output = control_euler_step(
                    restored,
                    sample_event,
                    sample_coordinates,
                    node_features,
                    atom_mask,
                    mode=mode,
                    tau=tau,
                    dt=dt,
                    condition_features=condition_features,
                )
                residuals.append(float(projector.residual(sample_output.event_velocity).abs().max()))
            rollouts[mode] = {
                "event_finite": bool(torch.isfinite(sample_event).all()),
                "geometry_finite": bool(torch.isfinite(sample_coordinates).all()),
                "velocity_conservation_residual_max_abs": max(residuals),
            }

    control_manifest = coupled_control_manifest(
        modes=modes,
        models={mode: restored for mode in modes},
        physical_call_budgets={mode: 0 for mode in modes},
        coupling_strength=1.0,
        guidance_strength=0.0,
        dataset_fingerprint="synthetic-software-smoke-only",
        split_fingerprint="not-applicable",
        input_view_fingerprint="reactant-side-synthetic-v1",
    )
    if restore_info["resume_kind"] != "training" or not restore_info["complete_resume"]:
        raise RuntimeError("checkpoint did not provide a complete training resume")
    if max(restore_differences.values()) > 1e-12:
        raise RuntimeError("checkpoint restore changed a registered control output")
    if any(
        not rollout["event_finite"]
        or not rollout["geometry_finite"]
        or rollout["velocity_conservation_residual_max_abs"] > 1e-8
        for rollout in rollouts.values()
    ):
        raise RuntimeError("a registered control rollout violated the software contract")
    if control_manifest["same_generation_weights_for_controls"] is not True:
        raise RuntimeError("registered controls did not use the same measured model state")

    report = {
        "schema": "xtbflow-joint-flow-smoke/v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_host": platform.node(),
        "data_status": "diagnostic_synthetic_nonzero_time_dependent_velocity_target",
        "seed": args.seed,
        "training_steps": args.training_steps,
        "sample_steps": args.sample_steps,
        "dt": dt,
        "tau_domain": [0.0, min(1.0, args.sample_steps * dt)],
        "runtime_config": runtime.constructor_record(),
        "runtime_config_path": str(args.config),
        "runtime_config_sha256": _config_sha256(args.config),
        "git": _git_identity(),
        "checkpoint_schema": restore_info["schema"],
        "complete_resume": restore_info["complete_resume"],
        "resume_kind": restore_info["resume_kind"],
        "tau_schedule": list(tau_schedule),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "model_identity_before_save": identity_before,
        "model_identity_after_restore": model_state_identity(restored),
        "restore_max_abs_difference": max(restore_differences.values()),
        "restore_component_max_abs_difference": restore_differences,
        "rollouts": rollouts,
        "control_manifest": control_manifest,
        "elapsed_seconds": time.monotonic() - started,
        "limits": [
            "Synthetic targets only test the software chain.",
            "No real event or transition-state supervision was used.",
            "Zero calculator calls were made and no chemistry result is claimed.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
