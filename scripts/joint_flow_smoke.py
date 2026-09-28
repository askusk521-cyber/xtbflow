#!/usr/bin/env python3
"""Run a bounded joint-flow train/save/restore/sample smoke test.

This is a software-chain check using a deterministic nonzero synthetic
velocity target.  It is deliberately labelled as such: the run does not
establish chemical accuracy, physical time interpretation, or performance on
admitted reference data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import torch

from xtbflow.models import ConservationProjector, JointEventGeometryFlow
from xtbflow.sampling import euler_step
from xtbflow.training import joint_flow_loss, load_joint_checkpoint, save_joint_checkpoint


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--training-steps", type=int, default=8)
    parser.add_argument("--sample-steps", type=int, default=4)
    parser.add_argument("--dt", type=float, default=None, help="explicit normalized Euler step; defaults to 1/sample_steps")
    args = parser.parse_args()
    if args.training_steps < 1 or args.sample_steps < 1:
        raise SystemExit("training and sample steps must be positive")
    dt = 1.0 / args.sample_steps if args.dt is None else args.dt
    if not 0 < dt <= 1 or args.sample_steps * dt > 1 + 1e-12:
        raise SystemExit("dt must be positive and sample_steps*dt must be at most one")
    started = time.monotonic()
    torch.manual_seed(args.seed)
    dtype = torch.float64
    event = torch.tensor([[0.5, -0.2, 0.1]], dtype=dtype)
    coordinates = torch.tensor([[[0.0, 0.0, 0.0], [0.8, 0.1, 0.0], [-0.2, 0.7, 0.1]]], dtype=dtype)
    node_features = torch.randn(1, 3, 4, dtype=dtype)
    atom_mask = torch.ones(1, 3, dtype=torch.bool)
    projector = ConservationProjector(torch.tensor([[1.0, 1.0, 1.0]], dtype=dtype))

    model = JointEventGeometryFlow(projector, 4, hidden_dim=8, radial_features=4).to(dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    target_event = torch.tensor([[0.2, -0.1, -0.1]], dtype=dtype)
    target_geometry = torch.zeros_like(coordinates)
    target_geometry[:, 1, 0] = 0.15
    target_geometry[:, 2, 1] = -0.1
    losses: list[float] = []
    for _ in range(args.training_steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(event, coordinates, node_features, atom_mask, tau=0.5)
        loss = joint_flow_loss(output, target_event, target_geometry, atom_mask=atom_mask, require_nonzero_target=True)["total"]
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    model.eval()
    with torch.no_grad():
        pre_save = model(event, coordinates, node_features, atom_mask, tau=0.5)
    state_digest_before = _state_digest(model)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_joint_checkpoint(
        args.checkpoint,
        model,
        optimizer,
        loop={"global_step": args.training_steps, "tau": 0.5, "dt": dt, "data_order": list(range(args.training_steps))},
        sampler_state={"event_state": torch.zeros_like(event), "coordinates": coordinates.clone(), "node_features": node_features.clone(), "atom_mask": atom_mask.clone()},
        metadata={"target": "diagnostic_nonzero_synthetic_velocity"},
    )

    restored = JointEventGeometryFlow(projector, 4, hidden_dim=8, radial_features=4).to(dtype=dtype)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-2)
    restore_info = load_joint_checkpoint(args.checkpoint, restored, restored_optimizer)
    restored.eval()
    with torch.no_grad():
        post_restore = restored(event, coordinates, node_features, atom_mask, tau=0.5)
    restore_differences = {
        "event_velocity": _max_abs(pre_save.event_velocity, post_restore.event_velocity),
        "geometry_velocity": _max_abs(pre_save.geometry_velocity, post_restore.geometry_velocity),
        "event_message": _max_abs(pre_save.event_message, post_restore.event_message),
        "geometry_message": _max_abs(pre_save.geometry_message, post_restore.geometry_message),
    }

    # A short deterministic Euler rollout checks that the restored model can
    # produce finite states while the projected event velocity remains in the
    # declared conservation subspace.
    sample_event = torch.zeros_like(event)
    sample_coordinates = coordinates.clone()
    residuals: list[float] = []
    with torch.no_grad():
        for step in range(args.sample_steps):
            tau = step * dt
            sample_event, sample_coordinates, sample_output = euler_step(
                restored,
                sample_event,
                sample_coordinates,
                node_features,
                atom_mask,
                tau=tau,
                dt=dt,
            )
            residuals.append(float((sample_event @ projector.constraint_matrix.T).abs().max()))

    report = {
        "schema": "xtbflow-joint-flow-smoke/v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_host": platform.node(),
        "data_status": "diagnostic_synthetic_nonzero_velocity_target",
        "seed": args.seed,
        "training_steps": args.training_steps,
        "sample_steps": args.sample_steps,
        "dt": dt,
        "tau_domain": [0.0, min(1.0, args.sample_steps * dt)],
        "checkpoint_schema": restore_info["schema"],
        "complete_resume": restore_info["complete_resume"],
        "final_loss": losses[-1],
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "state_digest_before_save": state_digest_before,
        "state_digest_after_restore": _state_digest(restored),
        "restore_max_abs_difference": max(restore_differences.values()),
        "restore_component_max_abs_difference": restore_differences,
        "sampled_event_finite": bool(torch.isfinite(sample_event).all()),
        "sampled_geometry_finite": bool(torch.isfinite(sample_coordinates).all()),
        "conservation_residual_max_abs": max(residuals),
        "elapsed_seconds": time.monotonic() - started,
        "limits": [
            "Synthetic nonzero targets only test the software chain.",
            "The checkpoint is a local smoke artefact and is not a trained scientific model.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
