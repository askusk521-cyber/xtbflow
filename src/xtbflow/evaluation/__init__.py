"""Evaluation helpers for bounded candidate outputs."""

from .terminal import TerminalCandidate, deduplicate_terminal_candidates
from .blind_rollout import (
    BlindRolloutConfig,
    BlindRolloutResult,
    CandidateAttempt,
    ReactantRolloutInput,
    benchmark_rollouts,
    blind_rollout,
    make_reactant_input,
)

__all__ = [
    "TerminalCandidate",
    "deduplicate_terminal_candidates",
    "BlindRolloutConfig",
    "BlindRolloutResult",
    "CandidateAttempt",
    "ReactantRolloutInput",
    "benchmark_rollouts",
    "blind_rollout",
    "make_reactant_input",
]
