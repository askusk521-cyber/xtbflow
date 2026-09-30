"""Endpoint connectivity and observed-event extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


_COVALENT_RADII_ANGSTROM = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}


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


def endpoint_connectivity_gate_pass(
    evidence: ConnectivityEvidence,
    *,
    require_same_bonds: bool,
) -> bool:
    """Evaluate the endpoint connectivity gate from the calculated graphs.

    The observed endpoint event is derived from the two supplied endpoint
    structures.  A caller requiring a conformational path therefore fails
    when any bond edit is observed; it cannot mark the path as validated by
    supplying an independent expected-event flag.
    """

    if type(require_same_bonds) is not bool:
        raise ValueError("require_same_bonds must be boolean")
    return not require_same_bonds or not observed_event(evidence)


def infer_binary_connectivity(
    symbols: Sequence[str],
    coordinates_angstrom: Sequence[Sequence[float]],
    *,
    scale: float = 1.25,
) -> tuple[tuple[int, ...], ...]:
    """Infer an endpoint bond graph from the supplied coordinates.

    This is a declared binary covalent-radius diagnostic. It is intentionally
    used only to compare the two calculated endpoints; it does not label a
    mechanism or replace a source-supplied bond order.
    """

    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not np.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("scale must be finite and positive")
    names = tuple(str(symbol) for symbol in symbols)
    if not names or any(symbol not in _COVALENT_RADII_ANGSTROM for symbol in names):
        raise ValueError("symbols contain unsupported elements")
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.shape != (len(names), 3) or not np.isfinite(coordinates).all():
        raise ValueError("coordinates must be finite [N,3]")
    radii = np.asarray(
        [_COVALENT_RADII_ANGSTROM[symbol] for symbol in names], dtype=float
    )
    distances = np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :], axis=-1
    )
    threshold = float(scale) * (radii[:, None] + radii[None, :])
    bonds = (distances > 0.4) & (distances <= threshold)
    np.fill_diagonal(bonds, False)
    return tuple(tuple(int(value) for value in row) for row in bonds)


def compare_candidate_event(candidate: Sequence[Sequence[int]], evidence: ConnectivityEvidence) -> bool:
    """Compare a proposed event to observed endpoint edits without relabelling it."""

    normalized = tuple(tuple(int(item) for item in row) for row in candidate)
    return normalized == observed_event(evidence)
