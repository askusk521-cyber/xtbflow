"""Deterministic candidate geometry rules that read reactants only.

The approach rule here is a bounded fixture generator, not a reaction model.
It preserves each component's internal geometry, places two declared reactant
components at a fixed centroid distance, and chooses a collision-safe
principal-axis orientation with a deterministic tie-break.  It never reads a
product, TS, event label, or endpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Sequence

import numpy as np

from xtbflow.calculators import MolecularSystem


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _basis(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered
    _, vectors = np.linalg.eigh(covariance)
    # Eigenvector signs are arbitrary; fix each sign by its largest component.
    for column in range(3):
        pivot = int(np.argmax(np.abs(vectors[:, column])))
        if vectors[pivot, column] < 0:
            vectors[:, column] *= -1.0
    # Independent eigenbases may have opposite parity even when each sign is
    # deterministic. Both bases must be right handed before B1 P B2.T is SO(3).
    if np.linalg.det(vectors) < 0:
        vectors[:, -1] *= -1.0
    return vectors


def _proper_signed_permutations() -> tuple[np.ndarray, ...]:
    candidates: list[np.ndarray] = []
    import itertools

    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=float)
            for column, row in enumerate(permutation):
                matrix[row, column] = signs[column]
            if np.linalg.det(matrix) > 0:
                candidates.append(matrix)
    return tuple(candidates)


_ORIENTATIONS = _proper_signed_permutations()


@dataclass(frozen=True)
class ReactantApproachResult:
    coordinates: tuple[tuple[float, float, float], ...]
    metadata: dict[str, Any]

    @property
    def geometry_hash(self) -> str:
        return _hash([list(row) for row in self.coordinates])


def _validate_components(components: Sequence[Sequence[int]], size: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if len(components) != 2:
        raise ValueError("reactant approach currently requires exactly two components")
    normalized = tuple(tuple(int(index) for index in component) for component in components)
    if any(not component for component in normalized):
        raise ValueError("reactant components must be nonempty")
    flattened = [index for component in normalized for index in component]
    if sorted(flattened) != list(range(size)):
        raise ValueError("reactant components must partition the atom inventory")
    return normalized


def build_reactant_approach(
    system: MolecularSystem,
    components: Sequence[Sequence[int]],
    *,
    target_centroid_distance_angstrom: float = 3.0,
    collision_distance_angstrom: float = 0.70,
) -> ReactantApproachResult:
    """Create a fixed candidate from two reactant-only components."""

    if not math.isfinite(float(target_centroid_distance_angstrom)) or target_centroid_distance_angstrom <= 0:
        raise ValueError("target_centroid_distance_angstrom must be positive and finite")
    if not math.isfinite(float(collision_distance_angstrom)) or collision_distance_angstrom <= 0:
        raise ValueError("collision_distance_angstrom must be positive and finite")
    normalized = _validate_components(components, len(system.symbols))
    original = np.asarray(system.coordinates, dtype=float)
    first, second = (original[list(indexes)] for indexes in normalized)
    center = original.mean(axis=0)
    first_center = first.mean(axis=0)
    second_center = second.mean(axis=0)
    first_centered = first - first_center
    second_centered = second - second_center
    first_basis = _basis(first_centered)
    second_basis = _basis(second_centered)

    choices: list[tuple[tuple[float, float, int, int], np.ndarray, float]] = []
    for orientation_index, orientation in enumerate(_ORIENTATIONS):
        rotation = first_basis @ orientation @ second_basis.T
        rotated_second = second_centered @ rotation.T
        for axis_index in range(3):
            axis = first_basis[:, axis_index]
            target_first = center - axis * (target_centroid_distance_angstrom / 2.0)
            target_second = center + axis * (target_centroid_distance_angstrom / 2.0)
            candidate = np.empty_like(original)
            candidate[list(normalized[0])] = first_centered + target_first
            candidate[list(normalized[1])] = rotated_second + target_second
            distances = [
                float(np.linalg.norm(candidate[i] - candidate[j]))
                for i in normalized[0]
                for j in normalized[1]
            ]
            minimum = min(distances)
            collision_penalty = sum(max(0.0, collision_distance_angstrom - value) ** 2 for value in distances)
            # Prefer collision-free candidates, then maximum clearance, then
            # stable axis/orientation order.  No calculator or target label is
            # consulted by this ranking.
            score = (1.0 if collision_penalty == 0.0 else 0.0, minimum, -axis_index, -orientation_index)
            choices.append((score, candidate, minimum))
    _, selected, minimum = max(choices, key=lambda item: item[0])
    coordinates = tuple(tuple(float(value) for value in row) for row in selected)
    metadata = {
        "generator": "reactant-principal-axis-approach-v2",
        "rotation_group": "SO(3)",
        "enumerated_placements": len(choices),
        "seed_used_for_pose_selection": False,
        "components": [list(component) for component in normalized],
        "approach_centroid_distance_angstrom": float(target_centroid_distance_angstrom),
        "collision_distance_angstrom": float(collision_distance_angstrom),
        "minimum_cross_component_distance_angstrom": minimum,
        "input_hash": system.input_hash,
    }
    return ReactantApproachResult(coordinates, metadata)


__all__ = ["ReactantApproachResult", "build_reactant_approach"]
