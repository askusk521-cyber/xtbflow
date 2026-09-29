"""Checkpoint helpers for reproducible joint-flow continuation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import random

import numpy as np
import torch


CHECKPOINT_SCHEMA = "xtbflow-joint-checkpoint/v1"
_TRAINING_LOOP_FIELDS = ("global_step", "data_order")


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"torch_cpu": torch.get_rng_state(), "python": random.getstate(), "numpy": np.random.get_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Mapping[str, Any]) -> None:
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_joint_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    scheduler: Any = None,
    loop: Mapping[str, Any] | None = None,
    sampler_state: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Atomically persist model, optimizer, loop cursor, dynamic state and RNG."""

    loop_payload = dict(loop or {})
    sampler_payload = dict(sampler_state or {})
    # A model-only checkpoint is useful for inference, but it must never be
    # reported as a resumable training checkpoint.  Mark the caller's intent
    # separately so a malformed training checkpoint is distinguishable from a
    # deliberate inference export during recovery.
    training_requested = optimizer is not None
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scheduler_required": scheduler is not None,
        "loop": loop_payload,
        "sampler_state": sampler_payload,
        "rng_state": _rng_state(),
        "model_training": model.training,
        "training_requested": training_requested,
        "metadata": dict(metadata or {}),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(target)


def load_joint_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    scheduler: Any = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore a checkpoint and return its loop/sampler metadata.

    Legacy ``{"model", "optimizer", "seed"}`` smoke checkpoints remain
    readable but are explicitly reported as incomplete resumes.
    """

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    checkpoint_optimizer = payload.get("optimizer")
    if optimizer is not None and checkpoint_optimizer is not None:
        optimizer.load_state_dict(checkpoint_optimizer)
    checkpoint_scheduler = payload.get("scheduler")
    if scheduler is not None and checkpoint_scheduler is not None:
        scheduler.load_state_dict(checkpoint_scheduler)
    if payload.get("model_training") is not None:
        model.train(bool(payload["model_training"]))
    if restore_rng and payload.get("rng_state") is not None:
        _restore_rng(payload["rng_state"])
    schema = payload.get("schema")
    legacy = schema != CHECKPOINT_SCHEMA
    loop = payload.get("loop")
    sampler_state = payload.get("sampler_state")
    valid_loop = isinstance(loop, Mapping) and all(field in loop for field in _TRAINING_LOOP_FIELDS)
    if valid_loop:
        valid_loop = type(loop["global_step"]) is int and loop["global_step"] >= 0 and isinstance(loop["data_order"], (list, tuple))
    valid_sampler = isinstance(sampler_state, Mapping) and bool(sampler_state)
    has_rng = isinstance(payload.get("rng_state"), Mapping) and "torch_cpu" in payload["rng_state"]
    scheduler_required = bool(payload.get("scheduler_required", False))
    scheduler_restored = not scheduler_required or (scheduler is not None and checkpoint_scheduler is not None)
    rng_restored = restore_rng and has_rng
    training_complete = (
        not legacy
        and bool(payload.get("training_requested", checkpoint_optimizer is not None))
        and optimizer is not None
        and checkpoint_optimizer is not None
        and valid_loop
        and valid_sampler
        and rng_restored
        and scheduler_restored
    )
    if legacy:
        resume_kind = "legacy_incomplete"
    elif training_complete:
        resume_kind = "training"
    elif bool(payload.get("training_requested", False)) or checkpoint_optimizer is not None:
        resume_kind = "incomplete_training"
    else:
        resume_kind = "inference"
    return {
        "schema": schema,
        "legacy": legacy,
        "loop": loop,
        "sampler_state": sampler_state,
        "metadata": payload.get("metadata", {}),
        "resume_kind": resume_kind,
        "complete_resume": training_complete,
    }
