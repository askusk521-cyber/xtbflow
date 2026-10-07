"""V1b design-based statistics (guide v2.1 Section 11), standard library only.

Stage A: single-phase stratified SRSWOR with Horvitz-Thompson totals.
Stage B: paired parent HT for Delta^cert = Delta^proxy + D with the single
pre-registered decision. Unknown labels are carried as [0, 1] intervals and are
never collapsed to zero.
"""
from collections import defaultdict
from fractions import Fraction
from math import comb, isclose
from statistics import NormalDist, mean, variance


def stratified_total(N, n, values):
    if type(N) is not int or type(n) is not int or not 1 <= n <= N:
        raise ValueError("invalid stratum sizes")
    if len(values) != n:
        raise ValueError("sampled units lost or counted twice")
    total = N * mean(values)
    if n == N:
        v_total = 0.0
    elif n == 1:
        v_total = None
    else:
        v_total = N * N * (1 - n / N) * variance(values) / n
    return {"ht_total": total, "population": N,
            "ht_rate": total / N, "variance_total": v_total,
            "variance_status": "ESTIMABLE" if v_total is not None
                               else "NOT_ESTIMABLE"}


def bundle_interval(intervals, source_count, stream_complete=True):
    """any() over a bundle's candidate intervals; unexamined candidates are [0, 1]."""
    if not stream_complete or source_count < len(intervals):
        raise ValueError("incomplete source or duplicate candidate verdicts")
    if any(not 0 <= lo <= hi <= 1 for lo, hi in intervals):
        raise ValueError("invalid candidate interval")
    lo = max((x[0] for x in intervals), default=0)
    hi = max((x[1] for x in intervals), default=0)
    if len(intervals) < source_count:
        hi = 1
    return lo, hi


def paired_ht(rows, M):
    if type(M) is not int or M <= 0:
        raise ValueError("positive parent population required")
    by = defaultdict(list)
    for r in rows:
        by[r["stratum"]].append(r)
    lo_total = hi_total = weight_total = var_total = 0.0
    variance_known = True
    source_total = 0
    for h, rs in by.items():
        N, n = rs[0]["N_h"], rs[0]["n_h"]
        if (type(N) is not int or type(n) is not int or
                len(rs) != n or not 1 <= n <= N):
            raise ValueError("all sampled parents, including unknown, required")
        if any((r["N_h"], r["n_h"]) != (N, n) or
               not isclose(r["pi"], n / N) for r in rs):
            raise ValueError("inconsistent inclusion probability")
        source_total += N
        ds = []
        for r in rs:
            al, au = r["A2"]
            bl, bu = r["B1"]
            if (any(x not in (0, 1) for x in (al, au, bl, bu)) or
                    not (al <= au and bl <= bu)):
                raise ValueError("invalid parent interval")
            dl, du = bl - au, bu - al
            lo_total += (N / n) * dl
            hi_total += (N / n) * du
            weight_total += N / n
            if dl != du:
                variance_known = False
            ds.append(dl)
        if n < N:
            if n == 1:
                variance_known = False
            else:
                var_total += N * N * (1 - n / N) * variance(ds) / n
    if source_total != M or not isclose(weight_total, M):
        raise ValueError("population stratum missing")
    return {"ht_lower": lo_total / M, "ht_upper": hi_total / M,
            "hajek_lower": lo_total / weight_total,
            "hajek_upper": hi_total / weight_total,
            "variance": var_total / (M * M) if variance_known else None}


def hypergeom_count_interval(N, n, observed, tail):
    if (any(type(x) is not int for x in (N, n, observed)) or
            not (0 <= observed <= n <= N and 0 < tail < 0.5)):
        raise ValueError("invalid hypergeometric interval arguments")
    denom = comb(N, n)
    tail_fraction = Fraction(str(tail))
    accepted = []
    for K in range(N + 1):
        left = right = 0
        for k in range(max(0, n - (N - K)), min(n, K) + 1):
            numerator = comb(K, k) * comb(N - K, n - k)
            if k <= observed:
                left += numerator
            if k >= observed:
                right += numerator
        cutoff = tail_fraction.numerator * denom
        if (left * tail_fraction.denominator >= cutoff and
                right * tail_fraction.denominator >= cutoff):
            accepted.append(K)
    if not accepted:
        raise ArithmeticError("empty confidence set")
    return min(accepted), max(accepted)


def paired_finite_population_ci(rows, M, alpha=0.05):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    paired_ht(rows, M)  # validate coverage and inclusion metadata
    by = defaultdict(list)
    for r in rows:
        by[r["stratum"]].append(r)
    tail = Fraction(str(alpha)) / (4 * len(by))
    total_lo = total_hi = 0
    for rs in by.values():
        N, n = rs[0]["N_h"], len(rs)
        bounds = [(r["B1"][0] - r["A2"][1],
                   r["B1"][1] - r["A2"][0]) for r in rs]
        plus_certain = sum(lo == 1 for lo, hi in bounds)
        plus_possible = sum(hi == 1 for lo, hi in bounds)
        minus_certain = sum(hi == -1 for lo, hi in bounds)
        minus_possible = sum(lo == -1 for lo, hi in bounds)
        pl = hypergeom_count_interval(N, n, plus_certain, tail)[0]
        pu = hypergeom_count_interval(N, n, plus_possible, tail)[1]
        ml = hypergeom_count_interval(N, n, minus_certain, tail)[0]
        mu = hypergeom_count_interval(N, n, minus_possible, tail)[1]
        total_lo += pl - mu
        total_hi += pu - ml
    return {"lower": max(-1.0, total_lo / M),
            "upper": min(1.0, total_hi / M), "alpha": alpha,
            "method": "stratified_hypergeom_bonferroni"}


def delta_cert_decision(rows, M, delta_proxy, complete, alpha=0.05):
    """The single pre-registered V1b decision (Section 10.5); depends only on Delta^cert."""
    est = paired_ht(rows, M)
    if not complete or est["ht_lower"] != est["ht_upper"]:
        return {"decision": "INCOMPLETE_BOUNDS", "bounds": est}
    point = est["ht_lower"]
    by = defaultdict(list)
    for r in rows:
        by[r["stratum"]].append(r)
    normal_ok = est["variance"] is not None and all(
        rs[0]["n_h"] == rs[0]["N_h"] or
        (len(rs) >= 2 and
         variance([r["B1"][0] - r["A2"][0] for r in rs]) > 0)
        for rs in by.values())
    if normal_ok:
        z = NormalDist().inv_cdf(1 - alpha)
        se = est["variance"] ** 0.5
        lower, upper = point - z * se, point + z * se
        method = "stratified_ht_normal"
    else:
        # Simultaneous two-sided interval at 2*alpha: each end is a one-sided bound.
        ci = paired_finite_population_ci(rows, M, alpha=2 * alpha)
        lower, upper, method = ci["lower"], ci["upper"], ci["method"]
    if lower > 0:
        decision = "V1A_CONFIRMED"
    elif upper <= 0:
        decision = "V1A_OVERTURNED"
    else:
        decision = "V1B_INCONCLUSIVE"
    return {"decision": decision, "delta_cert": point,
            "D_hat": point - delta_proxy,
            "lower_one95": lower, "upper_one95": upper,
            "interval_method": method}
