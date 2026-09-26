"""Endpoint connectivity and observed-event extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def _matrix(value: Sequence[Sequence[int]], name: str) -> tuple[tuple[int, ...], ...]:
    rows = tuple(tuple(int(item) for item in row) for row in value)
    if not rows or any(len(row) != len(rows) for row in rows):
        raise ValueError(f"{name} must be a nonempty square matrix")
    if any(item < 0 for row in rows for item in row):
        raise ValueError(f"{name} cannot contain negative bond orders")
    if any(rows[i][j] != rows[j][i] for i in range(len(rows)) for j in range(len(rows))):
        raise ValueError(f"{name} must be symmetric")
    if any(rows[i][i] != 0 for i in range(len(rows))):
        raise ValueError(f"{name} diagonal must be zero")
    return rows


@dataclass(frozen=True)
class ConnectivityEvidence:
    reactant_bonds: tuple[tuple[int, ...], ...]
    product_bonds: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        reactant = _matrix(self.reactant_bonds, "reactant_bonds")
        product = _matrix(self.product_bonds, "product_bonds")
        if len(reactant) != len(product):
            raise ValueError("reactant and product atom inventories must match")
        object.__setattr__(self, "reactant_bonds", reactant)
        object.__setattr__(self, "product_bonds", product)


def observed_event(evidence: ConnectivityEvidence) -> tuple[tuple[int, int, int], ...]:
    """Return endpoint bond edits as ``(atom_i, atom_j, delta_order)``."""

    n = len(evidence.reactant_bonds)
    edits: list[tuple[int, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            delta = evidence.product_bonds[i][j] - evidence.reactant_bonds[i][j]
            if delta:
                edits.append((i, j, delta))
    return tuple(edits)


def compare_candidate_event(candidate: Sequence[Sequence[int]], evidence: ConnectivityEvidence) -> bool:
    """Compare a proposed event to observed endpoint edits without relabelling it."""

    normalized = tuple(tuple(int(item) for item in row) for row in candidate)
    return normalized == observed_event(evidence)

