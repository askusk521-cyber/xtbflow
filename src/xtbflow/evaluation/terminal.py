"""Budgeted terminal candidate de-duplication independent of event labels."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

from xtbflow.data.records import canonical_hash


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
        return canonical_hash({"geometry": rounded})


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

