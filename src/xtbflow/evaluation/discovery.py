"""Parent-paired discovery statistics with explicit unknown costs and censoring.

Certificate inputs must come from the artifact-verifying adapter. This module
does not infer certification from generated graphs, proxy hits or TS geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import mean
from typing import Mapping, Sequence


COUNT_AXES = (
    "raw_proposals", "joint_forwards", "event_forwards", "geometry_forwards",
    "energy_calls", "force_calls", "energy_force_calls", "hvp_calls", "hessian_calls",
    "reference_primitive_calls", "reference_attempts", "retries", "cache_hits", "storage_bytes",
)
TIME_AXES = ("wall_seconds", "cpu_seconds", "gpu_seconds")
COST_AXES = COUNT_AXES + TIME_AXES
TRAINING_AXES = ("training_gpu_seconds", "training_cpu_seconds", "tuning_gpu_seconds", "tuning_cpu_seconds")
CONNECTED = frozenset({"reference_ts_connected", "valid_alternative"})
EVIDENCE = CONNECTED | {"same_endpoint", "reactant_mismatch", "attempted_unresolved"}
TERMINAL = frozenset({"completed", "execution_failed", "execution_timeout", "execution_interrupted", "budget_exhausted",
                     "not_started_budget_exhausted", "not_started_error", "cancelled", "native_output_only"})


def check_hash(value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("explicit SHA256 identity required")


def check_cost(cost: Mapping, axes: Sequence[str] = COST_AXES) -> None:
    if set(cost) != set(axes):
        raise ValueError("every cost field must be explicit; use null for unknown")
    for axis, value in cost.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("cost must be nonnegative finite or unknown")
        if axis in COUNT_AXES and type(value) is not int:
            raise ValueError("discrete cost counters require integers")


def add_costs(rows: Sequence[Mapping]) -> dict:
    for row in rows:
        check_cost(row)
    return {axis: None if any(r[axis] is None for r in rows) else sum(r[axis] for r in rows)
            for axis in COST_AXES}


@dataclass(frozen=True)
class Certificate:
    result_sha256: str
    protocol_sha256: str
    evidence_status: str
    observed_event_id: str | None

    def __post_init__(self):
        check_hash(self.result_sha256)
        check_hash(self.protocol_sha256)
        if self.evidence_status not in EVIDENCE:
            raise ValueError("unknown certification evidence state")
        if self.evidence_status in CONNECTED and (not isinstance(self.observed_event_id, str) or not self.observed_event_id):
            raise ValueError("connected certificate requires an observed event")


@dataclass(frozen=True)
class DiscoveryAttempt:
    candidate_id: str
    status: str
    cost: Mapping
    certificate: Certificate | None

    def __post_init__(self):
        if not self.candidate_id or self.status not in TERMINAL:
            raise ValueError("explicit candidate and terminal execution state required")
        check_cost(self.cost)
        if self.certificate and self.certificate.evidence_status in CONNECTED and self.status != "completed":
            raise ValueError("failed execution cannot count a certified channel")


@dataclass(frozen=True)
class DiscoveryTrial:
    parent_id: str
    input_sha256: str
    formula_group: str
    family_group: str | None
    method: str
    seed: int
    information_condition: str
    protocol_sha256: str
    budget: Mapping
    status: str
    attempts: Sequence[DiscoveryAttempt]
    training_cost: Mapping

    def __post_init__(self):
        for value in (self.input_sha256, self.protocol_sha256):
            check_hash(value)
        if not self.parent_id or not self.formula_group or not self.method or type(self.seed) is not int or self.seed < 0:
            raise ValueError("parent/formula/method/seed identities required")
        if self.information_condition not in {"reactant_only", "declared_partial_condition", "known_product"}:
            raise ValueError("undeclared information condition")
        if self.status not in TERMINAL:
            raise ValueError("ongoing trials cannot enter terminal discovery statistics")
        check_cost(self.budget)
        check_cost(self.training_cost, TRAINING_AXES)
        for axis in ("raw_proposals", "reference_primitive_calls", "cpu_seconds", "gpu_seconds", "wall_seconds"):
            if self.budget[axis] is None:
                raise ValueError("finite proposal/reference/time budgets required")
        for attempt in self.attempts:
            if attempt.certificate and attempt.certificate.protocol_sha256 != self.protocol_sha256:
                raise ValueError("certificate protocol differs from trial")


def summarize_trial(trial: DiscoveryTrial, *, axis: str, grid: Sequence[float]) -> dict:
    if axis not in COST_AXES or not grid or list(grid) != sorted(set(grid)) or any(
            isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) or x < 0 for x in grid):
        raise ValueError("known cost axis and increasing finite grid required")
    if trial.budget[axis] is None or grid[-1] > trial.budget[axis]:
        raise ValueError("curve grid exceeds declared budget")
    running, observed, within_budget = [], set(), set()
    points = []
    first = None
    for attempt in trial.attempts:
        running.append(attempt.cost)
        spent = add_costs(running)
        capped = [k for k in COST_AXES if trial.budget[k] is not None]
        budget_known = all(spent[k] is not None for k in capped)
        eligible = budget_known and all(spent[k] <= trial.budget[k] for k in capped)
        certificate = attempt.certificate
        if certificate and certificate.evidence_status in CONNECTED:
            event = certificate.observed_event_id
            observed.add(event)
            if eligible:
                within_budget.add(event)
            if eligible and first is None:
                first = dict(candidate_id=attempt.candidate_id, cost=spent,
                             certificate_sha256=certificate.result_sha256)
        points.append((spent[axis], len(within_budget) if budget_known else None))
    totals = add_costs(running)
    curve = []
    for threshold in grid:
        count = 0
        for spent, channels in points:
            if spent is None:
                count = None  # Missing cost cannot become a free observation.
                break
            if spent > threshold:
                break
            count = channels
        curve.append(dict(budget=threshold, unique_certified_channels=count,
                          any_certified=None if count is None else bool(count)))
    return dict(parent_id=trial.parent_id, method=trial.method, seed=trial.seed, status=trial.status,
                inference_cost=totals, training_cost=dict(trial.training_cost),
                full_cost_known=all(v is not None for v in (*totals.values(), *trial.training_cost.values())),
                observed_unique_channels_all_spending=len(observed),
                unique_certified_channels=len(within_budget) if all(
                    totals[k] is not None for k in COST_AXES if trial.budget[k] is not None) else None,
                curve=curve,
                first_certification=first, first_certification_right_censored=first is None,
                censor_at_cost=totals if first is None else None,
                over_budget_axes=[k for k in COST_AXES if trial.budget[k] is not None
                                  and totals[k] is not None and totals[k] > trial.budget[k]])


def paired_comparison(left: Sequence[DiscoveryTrial], right: Sequence[DiscoveryTrial], *,
                      axis: str, threshold: float, cluster: str, bootstrap_samples: int,
                      bootstrap_seed: int, minimum_practical_gain: float) -> dict:
    """Pair seeds, average within parents, then resample whole independent groups."""
    if cluster not in {"parent_id", "formula_group", "family_group"} or bootstrap_samples < 100:
        raise ValueError("explicit grouping and at least 100 bootstrap draws required")
    if not math.isfinite(minimum_practical_gain) or minimum_practical_gain < 0:
        raise ValueError("finite nonnegative practical gain required")
    def index(trials):
        result = {(t.parent_id, t.seed): t for t in trials}
        if not result or len(result) != len(trials) or len({t.method for t in trials}) != 1:
            raise ValueError("nonempty single-method trials and unique parent/seed cells required")
        return result
    a, b = index(left), index(right)
    if set(a) != set(b):
        raise ValueError("planned paired cells differ; missing/failed trials must be explicit")
    differences, groups, unknown = {}, {}, []
    for key in sorted(a):
        x, y = a[key], b[key]
        if any(getattr(x, k) != getattr(y, k) for k in (
                "input_sha256", "information_condition", "protocol_sha256", "budget", "formula_group", "family_group")):
            raise ValueError("paired input permissions, protocol, group or budget mismatch")
        group = getattr(x, cluster)
        if group is None:
            raise ValueError("unknown independent grouping cannot support requested interval")
        if x.parent_id in groups and groups[x.parent_id] != group:
            raise ValueError("parent grouping changes between seeds")
        groups[x.parent_id] = group
        values = [summarize_trial(t, axis=axis, grid=[threshold])["curve"][0]["unique_certified_channels"] for t in (x, y)]
        if None in values:
            unknown.append(dict(parent_id=key[0], seed=key[1]))
        else:
            differences.setdefault(x.parent_id, []).append(values[0] - values[1])
    if unknown:
        return dict(status="cost_unresolved", planned_parent_seed_cells=len(a), unknown_cost_cells=unknown,
                    mean_difference=None, interval95=None)
    parent_differences = {p: mean(v) for p, v in differences.items()}
    buckets = {}
    for parent, difference in parent_differences.items():
        buckets.setdefault(groups[parent], []).append(difference)
    interval = None
    if len(buckets) >= 2:
        keys, rng = sorted(buckets), random.Random(bootstrap_seed)
        draws = sorted(mean(v for _ in keys for v in buckets[rng.choice(keys)]) for _ in range(bootstrap_samples))
        interval = [draws[math.floor(.025*(len(draws)-1))], draws[math.ceil(.975*(len(draws)-1))]]
    return dict(status="descriptive_only" if interval is None else "paired_interval_computed",
                axis=axis, threshold=threshold, difference_direction=f"{left[0].method} minus {right[0].method}",
                parents=len(parent_differences), independent_groups=len(buckets), cluster=cluster,
                mean_difference=mean(parent_differences.values()), interval95=interval,
                minimum_practical_gain=minimum_practical_gain,
                interval_exceeds_practical_gain=None if interval is None else interval[0] > minimum_practical_gain,
                parent_differences=parent_differences,
                claim="Development paired statistic only; frozen independent confirmation is separate")
