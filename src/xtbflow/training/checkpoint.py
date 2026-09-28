"""Checkpoint helpers for reproducible joint-flow continuation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import random

import numpy as np
import torch


CHECKPOINT_SCHEMA = "xtbflow-joint-checkpoint/v1"


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

    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "loop": dict(loop or {}),
        "sampler_state": dict(sampler_state or {}),
        "rng_state": _rng_state(),
        "model_training": model.training,
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
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if payload.get("model_training") is not None:
        model.train(bool(payload["model_training"]))
    if restore_rng and payload.get("rng_state") is not None:
        _restore_rng(payload["rng_state"])
    return {
        "schema": payload.get("schema"),
        "legacy": payload.get("schema") != CHECKPOINT_SCHEMA,
        "loop": payload.get("loop"),
        "sampler_state": payload.get("sampler_state"),
        "metadata": payload.get("metadata", {}),
        "complete_resume": payload.get("schema") == CHECKPOINT_SCHEMA and payload.get("rng_state") is not None,
    }
