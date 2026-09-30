"""Evaluation helpers for bounded candidate outputs."""

from .terminal import TerminalCandidate, deduplicate_terminal_candidates
from .transition1x_geometry import (
    GUESS_FIELDS,
    Transition1xGeometryError,
    evaluate_transition1x_guesses,
    geometry_metrics,
)

__all__ = [
    "TerminalCandidate",
    "deduplicate_terminal_candidates",
    "GUESS_FIELDS",
    "Transition1xGeometryError",
    "evaluate_transition1x_guesses",
    "geometry_metrics",
]
