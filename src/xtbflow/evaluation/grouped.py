"""Leakage-safe summaries for repeated records from the same parent reaction.

Reaction archives often contain several records for one parent reaction.  A
record-weighted mean therefore gives large parents more influence than small
parents.  The helpers in this module keep both views explicit and provide a
paired, deterministic bootstrap for comparing two arms on the same parent
groups.
"""
from __future__ import annotations

from collections import defaultdict
import math
from numbers import Real
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def _metric_names(metric_keys: Iterable[str]) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(metric_keys))
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("metric_keys must contain at least one non-empty string")
    return names


def _validated_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    group_key: str,
    metric_keys: Sequence[str],
) -> list[tuple[str, dict[str, float]]]:
    if not isinstance(group_key, str) or not group_key.strip():
        raise ValueError("group_key must be a non-empty string")
    names = _metric_names(metric_keys)
    validated: list[tuple[str, dict[str, float]]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"row {index} must be a mapping")
        group = row.get(group_key)
        if not isinstance(group, str) or not group.strip():
            raise ValueError(f"row {index} has no non-empty {group_key}")
        values: dict[str, float] = {}
        for metric in names:
            if metric not in row:
                raise ValueError(f"row {index} is missing metric {metric!r}")
            value = row[metric]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{metric} in row {index} must be numeric")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{metric} in row {index} must be finite")
            values[metric] = value
        validated.append((group, values))
    if not validated:
        raise ValueError("at least one row is required")
    return validated


def summarize_by_group(
    rows: Iterable[Mapping[str, Any]],
    *,
    group_key: str = "parent_reaction_id",
    metric_keys: Iterable[str],
) -> dict[str, Any]:
    """Return record-weighted and equal-parent summaries.

    ``groups`` contains one mean per parent reaction.  ``record_weighted`` is
    the ordinary mean over all records, while ``parent_equal`` gives each
    parent one weight regardless of its number of records.
    """

    names = _metric_names(metric_keys)
    validated = _validated_rows(rows, group_key=group_key, metric_keys=names)
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for group, values in validated:
        grouped[group].append(values)

    record_weighted = {
        metric: float(np.mean([values[metric] for _, values in validated]))
        for metric in names
    }
    groups: list[dict[str, Any]] = []
    for group in sorted(grouped):
        values = grouped[group]
        groups.append(
            {
                "group_id": group,
                "record_count": len(values),
                "metric_means": {
                    metric: float(np.mean([row[metric] for row in values]))
                    for metric in names
                },
            }
        )
    parent_equal = {
        metric: float(np.mean([group["metric_means"][metric] for group in groups]))
        for metric in names
    }
    return {
        "group_key": group_key,
        "record_count": len(validated),
        "group_count": len(groups),
        "record_weighted": record_weighted,
        "parent_equal": parent_equal,
        "groups": groups,
    }


def paired_group_comparison(
    left_rows: Iterable[Mapping[str, Any]],
    right_rows: Iterable[Mapping[str, Any]],
    *,
    metric_keys: Iterable[str],
    left_label: str = "left",
    right_label: str = "right",
    group_key: str = "parent_reaction_id",
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare two arms after equal-weight parent aggregation.

    For each metric, the reported differences are ``left - right``.  Parent
    groups present in only one arm are retained in ``unmatched_*`` and are not
    silently treated as zero.  Bootstrap resampling is over parent groups,
    never over repeated records, so its uncertainty reflects the independent
    chemistry units available to this evaluation.
    """

    names = _metric_names(metric_keys)
    if not isinstance(left_label, str) or not left_label.strip() or not isinstance(right_label, str) or not right_label.strip():
        raise ValueError("comparison labels must be non-empty strings")
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    left = summarize_by_group(left_rows, group_key=group_key, metric_keys=names)
    right = summarize_by_group(right_rows, group_key=group_key, metric_keys=names)
    left_groups = {row["group_id"]: row["metric_means"] for row in left["groups"]}
    right_groups = {row["group_id"]: row["metric_means"] for row in right["groups"]}
    common = sorted(set(left_groups).intersection(right_groups))
    if not common:
        raise ValueError("the two arms have no common parent groups")
    unmatched_left = sorted(set(left_groups) - set(right_groups))
    unmatched_right = sorted(set(right_groups) - set(left_groups))
    rng = np.random.default_rng(seed)
    metrics: dict[str, Any] = {}
    pair_rows: list[dict[str, Any]] = []
    for group in common:
        pair_rows.append(
            {
                "group_id": group,
                "left": {metric: float(left_groups[group][metric]) for metric in names},
                "right": {metric: float(right_groups[group][metric]) for metric in names},
                "difference": {
                    metric: float(left_groups[group][metric] - right_groups[group][metric])
                    for metric in names
                },
            }
        )
    for metric in names:
        differences = np.asarray([row["difference"][metric] for row in pair_rows], dtype=np.float64)
        mean_difference = float(np.mean(differences))
        std = float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0
        bootstrap_indices = rng.integers(0, len(differences), size=(bootstrap_samples, len(differences)))
        bootstrap_means = differences[bootstrap_indices].mean(axis=1)
        lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
        metrics[metric] = {
            "left_minus_right_mean": mean_difference,
            "paired_group_std": std,
            "standard_error": float(std / math.sqrt(len(differences))),
            "bootstrap_95ci": [float(lower), float(upper)],
        }
    return {
        "group_key": group_key,
        "left_label": left_label,
        "right_label": right_label,
        "paired_group_count": len(common),
        "unmatched_left": unmatched_left,
        "unmatched_right": unmatched_right,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "metrics": metrics,
        "groups": pair_rows,
    }
