"""Deterministic parent-group plans for nested learning curves.

The helpers operate on identity strings only.  They never inspect labels or
scores, so selecting a smaller training set cannot leak target information.
Chemical samples remain the statistical unit; configurations belonging to one
parent are kept together by the caller's manifest split.
"""
from __future__ import annotations

import hashlib
import math
from typing import Iterable, Mapping


def deterministic_parent_order(parent_ids: Iterable[str], *, seed: str) -> tuple[str, ...]:
    """Return a result-blind stable order for unique parent identities."""

    if not isinstance(seed, str) or not seed.strip():
        raise ValueError("seed must be a non-empty string")
    values = tuple(parent_ids)
    if not values or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("parent_ids must contain non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError("parent_ids must be unique")
    return tuple(
        sorted(
            values,
            key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest(),
        )
    )


def nested_parent_subsets(
    parent_ids: Iterable[str],
    sizes: Iterable[int],
    *,
    seed: str,
) -> dict[int, tuple[str, ...]]:
    """Build nested result-blind parent subsets at the requested sizes."""

    ordered = deterministic_parent_order(parent_ids, seed=seed)
    requested = tuple(sizes)
    if not requested or any(type(size) is not int or size < 1 for size in requested):
        raise ValueError("sizes must contain positive integers")
    if tuple(sorted(set(requested))) != requested:
        raise ValueError("sizes must be strictly increasing")
    if requested[-1] > len(ordered):
        raise ValueError("largest requested size exceeds available parent count")
    return {size: ordered[:size] for size in requested}


def validate_nested_parent_subsets(subsets: Mapping[int, Iterable[str]]) -> None:
    """Validate that every larger training set contains each smaller set."""

    if not subsets:
        raise ValueError("subsets must not be empty")
    normalized: list[tuple[int, tuple[str, ...]]] = []
    for size, values in subsets.items():
        if type(size) is not int or size < 1:
            raise ValueError("subset sizes must be positive integers")
        current = tuple(values)
        if len(current) != size or len(set(current)) != len(current):
            raise ValueError(f"subset {size} must contain exactly {size} unique parents")
        if any(not isinstance(value, str) or not value.strip() for value in current):
            raise ValueError("subset parent identities must be non-empty strings")
        normalized.append((size, current))
    normalized.sort(key=lambda item: item[0])
    for (_, smaller), (_, larger) in zip(normalized, normalized[1:]):
        if not set(smaller).issubset(larger):
            raise ValueError("training parent subsets are not nested")


def parent_equal_mean(values_by_parent: Mapping[str, float]) -> float:
    """Average one scalar per parent, avoiding configuration-count weighting."""

    if not values_by_parent:
        raise ValueError("values_by_parent must not be empty")
    values = tuple(values_by_parent.values())
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError("parent metrics must be numeric")
    return float(sum(float(value) for value in values) / len(values))


__all__ = [
    "deterministic_parent_order",
    "nested_parent_subsets",
    "parent_equal_mean",
    "validate_nested_parent_subsets",
]
