from itertools import combinations
from math import isclose

import pytest

from xtbflow.v1.audit_sampling import build_populations,parent_stratum,proxy_class,stratified_srswor
from xtbflow.v1.audit_statistics import (bundle_interval,delta_cert_decision,hypergeom_count_interval,
                                         paired_ht,stratified_total)


def test_v1b_statistics_contract():
    # Guide Section 15.1, verbatim semantics.
    assert bundle_interval([], 0) == (0, 0)
    assert bundle_interval([(1, 1)], 9) == (1, 1)
    assert bundle_interval([(0, 0)], 2) == (0, 1)
    result = stratified_total(4, 2, [0, 1])
    assert isclose(result["ht_rate"], 0.5)
    assert isclose(result["variance_total"], 2.0)
    assert stratified_total(3, 1, [1])["variance_total"] is None
    assert stratified_total(1, 1, [1])["variance_total"] == 0
    assert hypergeom_count_interval(4, 4, 2, 0.025) == (2, 2)
    assert hypergeom_count_interval(10, 2, 2, 0.025)[0] < 10
    census = (
        [{"stratum": "B1_only", "N_h": 2, "n_h": 2, "pi": 1.0,
          "A2": (0, 0), "B1": (1, 1)}] * 2 +
        [{"stratum": "neither_empty", "N_h": 2, "n_h": 2, "pi": 1.0,
          "A2": (0, 0), "B1": (0, 0)}] * 2)
    out = delta_cert_decision(census, 4, delta_proxy=0.5, complete=True)
    assert out["decision"] == "V1A_CONFIRMED" and isclose(out["D_hat"], 0.0)
    partial = [dict(r) for r in census]
    for r in partial[:2]:
        r.update({"N_h": 4, "n_h": 2, "pi": 0.5})
    out = delta_cert_decision(partial, 6, delta_proxy=4 / 6, complete=True)
    assert out["interval_method"] == "stratified_hypergeom_bonferroni"


def test_srswor_reproducible_positive_probability_and_independent_streams():
    frame = [dict(unit_id=f"u{i}", stratum="a" if i < 6 else "b") for i in range(10)]
    s1 = stratified_srswor(frame, {"a": 3, "b": 2}, "seedA")
    assert s1 == stratified_srswor(frame, {"a": 3, "b": 2}, "seedA")
    assert all(isclose(r["pi"], r["n_h"] / r["N_h"]) for r in s1)
    with pytest.raises(ValueError):
        stratified_srswor(frame, {"a": 0, "b": 2}, "s")
    with pytest.raises(ValueError):
        stratified_srswor(frame, {"a": 3}, "s")
    with pytest.raises(ValueError):
        stratified_srswor(frame + frame[:1], {"a": 3, "b": 2}, "s")


def test_stratified_ht_unbiased_by_enumeration():
    values = {"a": [0, 1, 1, 0, 1], "b": [1, 0, 0]}
    n = {"a": 2, "b": 2}
    truth = sum(sum(v) for v in values.values())
    estimates = []
    for sa in combinations(values["a"], n["a"]):
        for sb in combinations(values["b"], n["b"]):
            estimates.append(stratified_total(5, 2, list(sa))["ht_total"] +
                             stratified_total(3, 2, list(sb))["ht_total"])
    assert isclose(sum(estimates) / len(estimates), truth)


def test_weighted_parent_difference_differs_from_unweighted():
    rows = ([{"stratum": "x", "N_h": 10, "n_h": 2, "pi": .2, "A2": (0, 0), "B1": (1, 1)}] * 2 +
            [{"stratum": "y", "N_h": 2, "n_h": 2, "pi": 1., "A2": (1, 1), "B1": (0, 0)}] * 2)
    est = paired_ht(rows, 12)
    assert isclose(est["ht_lower"], (10 - 2) / 12) and est["ht_lower"] == est["ht_upper"]


def test_unknown_labels_widen_and_force_incomplete():
    rows = [{"stratum": "s", "N_h": 2, "n_h": 2, "pi": 1., "A2": (0, 0), "B1": (0, 1)},
            {"stratum": "s", "N_h": 2, "n_h": 2, "pi": 1., "A2": (0, 0), "B1": (1, 1)}]
    out = delta_cert_decision(rows, 2, .5, complete=True)
    assert out["decision"] == "INCOMPLETE_BOUNDS"
    assert out["bounds"]["ht_lower"] == .5 and out["bounds"]["ht_upper"] == 1.


def test_hypergeometric_interval_keeps_width_for_identical_small_samples():
    lo, hi = hypergeom_count_interval(20, 3, 3, .0125)
    assert lo < 20 and hi == 20


def candidate(cid, arm, pid, status="PROXY_MATCH", best=False, event=False, units=100.):
    return dict(candidate_id=cid, arm=arm, parent_id=pid, proxy_status=status, hits_best_reference=best,
                hits_best_event=event, completion_units=units, is_fallback=False)


def bundle(pid, arm, cids, hit, best_events):
    return dict(parent_id=pid, arm=arm, query_id=pid + "_q", bundle_id=f"{pid}_{arm}", training_seed=0,
                sampling_seed=0, source_budget_units=800, stream_complete=True, candidate_ids=cids,
                proxy_best_reference_hit=hit, best_event_candidate_ids=best_events)


def test_populations_keep_empty_bundles_and_strata_match_best_event_ids():
    cands = {}; bundles = []
    for pid, (a2hit, b1hit, b1event) in {"p1": (0, 1, True), "p2": (0, 0, False), "p3": (0, 0, True)}.items():
        for arm in ("A0", "A1", "A2", "B0", "B1"):
            cid = f"{pid}/{arm}/c0"
            hit = (arm == "A2" and a2hit) or (arm == "B1" and b1hit)
            ev = arm == "B1" and b1event
            cands[cid] = candidate(cid, arm, pid, best=bool(hit), event=ev or bool(hit))
            if arm == "A0" and pid == "p2":
                bundles.append(bundle(pid, arm, [], False, []))
                continue
            bundles.append(bundle(pid, arm, [cid], bool(hit), [cid] if (ev or hit) else []))
    frame, parents, dp, summary = build_populations(bundles, cands)
    strata = {p["unit_id"]: p["stratum"] for p in parents}
    assert strata == {"p1": "B1_only", "p2": "neither_empty", "p3": "neither_with_best_event"}
    assert isclose(dp, 1 / 3) and summary["n_candidates"] == 14
    cands["p1/B1/c0"]["completion_units"] = 900
    with pytest.raises(ValueError):
        build_populations(bundles, cands)


def test_proxy_classes_are_exclusive():
    assert proxy_class(candidate("c", "A0", "p", best=True)) == "MATCH_BEST"
    assert proxy_class(candidate("c", "A0", "p")) == "MATCH_NONBEST"
    assert proxy_class(candidate("c", "A0", "p", "INVALID_OUTPUT")) == "INVALID_OUTPUT"
    assert proxy_class(candidate("c", "A0", "p", "MATCH_UNRESOLVED")) == "PROXY_UNRESOLVED"
    assert parent_stratum(dict(proxy_best_reference_hit=1, best_event_candidate_ids=["x"]),
                          dict(proxy_best_reference_hit=1, best_event_candidate_ids=["y"])) == "both"
