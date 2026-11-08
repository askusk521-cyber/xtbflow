"""Budgeted terminal candidate de-duplication independent of event labels."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable


def _canonical_hash(value: Any) -> str:
    """Keep terminal slot accounting independent of NumPy/Torch imports."""

    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


GENERATION_FAILURE_STAGES = (
    "generation_preflight",
    "initialization",
    "integration",
    "event_decode",
    "candidate_validation",
    "deduplication",
    "physical_refinement",
    "settlement",
    "report_write",
)
GENERATION_STATUSES = (
    "completed",
    "not_run",
    "execution_failed",
    "settlement_failed",
    "report_write_failed",
    "budget_exhausted",
)
INITIAL_STATE_POLICIES = ("deterministic_reactant", "event_noise", "geometry_noise", "event_geometry_noise", "latent")


@dataclass(frozen=True)
class GenerationProtocol:
    """Declare whether repeated candidates have a real stochastic source.

    A seed list alone does not make a deterministic model stochastic.  When a
    parent is attempted more than once, the protocol therefore requires an
    explicit non-deterministic initial-state or latent policy and a positive
    candidate cap.  Training and inference code can record this object before
    any candidates are generated.
    """

    candidate_cap: int
    attempt_seeds: tuple[int, ...]
    initial_state_policy: str = "deterministic_reactant"
    latent_dim: int = 0
    training_sampling_policy: str = "unrecorded"

    def validate(self) -> None:
        if type(self.candidate_cap) is not int or self.candidate_cap < 1:
            raise ValueError("candidate_cap must be a positive integer")
        if not self.attempt_seeds or any(type(seed) is not int for seed in self.attempt_seeds):
            raise ValueError("attempt_seeds must contain at least one integer seed")
        if len(set(self.attempt_seeds)) != len(self.attempt_seeds):
            raise ValueError("attempt_seeds must be unique")
        if len(self.attempt_seeds) > self.candidate_cap:
            raise ValueError("attempt_seeds cannot exceed the per-parent candidate cap")
        if self.initial_state_policy not in INITIAL_STATE_POLICIES:
            raise ValueError(f"unsupported initial_state_policy: {self.initial_state_policy}")
        if type(self.latent_dim) is not int or self.latent_dim < 0:
            raise ValueError("latent_dim must be a nonnegative integer")
        if self.initial_state_policy == "latent" and self.latent_dim < 1:
            raise ValueError("latent initialization requires a positive latent_dim")
        if len(self.attempt_seeds) > 1 and self.initial_state_policy == "deterministic_reactant":
            raise ValueError("repeated attempts require stochastic initial states or a latent variable")

    @property
    def stochastic(self) -> bool:
        return self.initial_state_policy != "deterministic_reactant"

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": "xtbflow-generation-protocol/v1",
            "candidate_cap": self.candidate_cap,
            "attempt_seeds": list(self.attempt_seeds),
            "initial_state_policy": self.initial_state_policy,
            "latent_dim": self.latent_dim,
            "training_sampling_policy": self.training_sampling_policy,
            "stochastic_source_declared": self.stochastic,
        }


@dataclass(frozen=True)
class GenerationAttemptRecord:
    """One settled generation attempt with stage-specific failure accounting."""

    attempt_id: str
    parent_id: str
    seed: int
    status: str
    failure_stage: str | None = None
    consumed_calculator_calls: int = 0
    candidates_produced: int = 0

    def validate(self) -> None:
        for name, value in (("attempt_id", self.attempt_id), ("parent_id", self.parent_id)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if self.status not in GENERATION_STATUSES:
            raise ValueError(f"unsupported generation status: {self.status}")
        if self.failure_stage is not None and self.failure_stage not in GENERATION_FAILURE_STAGES:
            raise ValueError(f"unsupported generation failure stage: {self.failure_stage}")
        if type(self.consumed_calculator_calls) is not int or self.consumed_calculator_calls < 0:
            raise ValueError("consumed_calculator_calls must be a nonnegative integer")
        if type(self.candidates_produced) is not int or self.candidates_produced < 0:
            raise ValueError("candidates_produced must be a nonnegative integer")
        if self.status == "not_run" and self.consumed_calculator_calls:
            raise ValueError("not_run attempts cannot consume calculator calls")
        if self.status == "completed" and self.failure_stage is not None:
            raise ValueError("completed attempts cannot have a failure stage")
        if self.status != "completed" and self.failure_stage is None:
            raise ValueError("failed or unrun attempts require an explicit failure stage")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "attempt_id": self.attempt_id,
            "parent_id": self.parent_id,
            "seed": self.seed,
            "status": self.status,
            "failure_stage": self.failure_stage,
            "consumed_calculator_calls": self.consumed_calculator_calls,
            "candidates_produced": self.candidates_produced,
        }


@dataclass(frozen=True)
class TerminalCandidate:
    """One raw candidate and optional score; event duplicates share geometry slots."""

    geometry: tuple[tuple[float, float, float], ...]
    event_key: str
    score: float = 0.0
    metadata: dict[str, Any] | None = None

    def geometry_key(self, *, tolerance: float = 1e-4) -> str:
        if tolerance <= 0 or not math.isfinite(tolerance):
            raise ValueError("tolerance must be finite and positive")
        rounded = tuple(tuple(round(float(value) / tolerance) for value in row) for row in self.geometry)
        return _canonical_hash({"geometry": rounded})


def deduplicate_terminal_candidates(candidates: Iterable[TerminalCandidate], *, max_slots: int, tolerance: float = 1e-4) -> tuple[TerminalCandidate, ...]:
    """Keep at most one best-scored candidate per geometry slot."""

    if type(max_slots) is not int or max_slots < 1:
        raise ValueError("max_slots must be a positive integer")
    best: dict[str, TerminalCandidate] = {}
    for candidate in candidates:
        if not math.isfinite(float(candidate.score)):
            raise ValueError("candidate scores must be finite")
        key = candidate.geometry_key(tolerance=tolerance)
        previous = best.get(key)
        if previous is None or (candidate.score, candidate.event_key) < (previous.score, previous.event_key):
            best[key] = candidate
    return tuple(sorted(best.values(), key=lambda item: (item.score, item.event_key))[:max_slots])
