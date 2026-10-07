"""V1b D0: finite populations, mutually exclusive strata and stratified SRSWOR.

Reads only the frozen V1a budget-16, training/sampling seed-0 source. No H
result may exist when strata or samples are built (guide Sections 3.1-3.3).
"""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from random import Random

ARMS = ('A0', 'A1', 'A2', 'B0', 'B1')
A_CLASSES = ('MATCH_BEST', 'MATCH_NONBEST', 'OUTSIDE_CATALOGUE_EVENT',
             'KNOWN_EVENT_GEOMETRY_MISS', 'PROXY_UNRESOLVED', 'INVALID_OUTPUT')
B_STRATA = ('B1_only', 'A2_only', 'both', 'neither_with_best_event', 'neither_empty')


def stratified_srswor(frame, allocation, seed):
    by = defaultdict(list)
    ids = [r["unit_id"] for r in frame]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sampling unit")
    for row in frame:
        by[row["stratum"]].append(row)
    if set(by) != set(allocation):
        raise ValueError("allocation must cover every nonempty stratum")
    selected = []
    for h in sorted(by):
        rows = sorted(by[h], key=lambda r: r["unit_id"])
        N, n = len(rows), allocation[h]
        if type(n) is not int or not 1 <= n <= N:
            raise ValueError("positive inclusion probability required")
        key = sha256(f"{seed}|{h}".encode()).digest()
        rng = Random(int.from_bytes(key, "big"))
        for row in sorted(rng.sample(rows, n), key=lambda r: r["unit_id"]):
            selected.append({**row, "N_h": N, "n_h": n,
                             "pi": n / N, "sampling_seed": str(seed)})
    return selected


def proxy_class(c):
    """Unique Stage-A class of one evaluated V1a candidate record."""
    status = c['proxy_status']
    if c.get('is_fallback') or status == 'INVALID_OUTPUT':
        return 'INVALID_OUTPUT'
    if status == 'PROXY_MATCH':
        return 'MATCH_BEST' if c['hits_best_reference'] else 'MATCH_NONBEST'
    if status in ('OUTSIDE_CATALOGUE_EVENT', 'KNOWN_EVENT_GEOMETRY_MISS'):
        return status
    return 'PROXY_UNRESOLVED'


def parent_stratum(a2, b1):
    """Stage-B stratum from the two frozen budget-16 bundles of one parent."""
    ya, yb = bool(a2['proxy_best_reference_hit']), bool(b1['proxy_best_reference_hit'])
    if yb and not ya:
        return 'B1_only'
    if ya and not yb:
        return 'A2_only'
    if ya and yb:
        return 'both'
    if a2['best_event_candidate_ids'] or b1['best_event_candidate_ids']:
        return 'neither_with_best_event'
    return 'neither_empty'


def build_populations(bundles, candidates):
    """Return (candidate frame, parent frame, delta_proxy, summary).

    `bundles` are the V1a v1b_source_population rows; `candidates` the matched
    seed-0 candidate records keyed by candidate_id. Every bundle candidate must be
    present and inside the 800-unit source budget; empty bundles are retained.
    """
    seen = set()
    by_parent = defaultdict(dict)
    frame = []
    for b in bundles:
        if (b['training_seed'], b['sampling_seed'], b['source_budget_units']) != (0, 0, 800):
            raise ValueError('bundle outside the budget-16 seed-0 source')
        if not b['stream_complete']:
            raise ValueError('incomplete source stream')
        if b['arm'] in by_parent[b['parent_id']]:
            raise ValueError('duplicate bundle')
        by_parent[b['parent_id']][b['arm']] = b
        if not set(b['best_event_candidate_ids']) <= set(b['candidate_ids']):
            raise ValueError('best-event candidates outside bundle')
        for cid in b['candidate_ids']:
            if cid in seen:
                raise ValueError('candidate in two bundles')
            seen.add(cid)
            c = candidates[cid]
            if c['completion_units'] > 800 + 1e-8 or c['arm'] != b['arm'] or c['parent_id'] != b['parent_id']:
                raise ValueError('candidate does not belong to its bundle budget')
            if (cid in b['best_event_candidate_ids']) != bool(c['hits_best_event']):
                raise ValueError('best-event list disagrees with candidate labels')
            cls = proxy_class(c)
            frame.append(dict(unit_id=cid, stratum=f"{b['arm']}|{cls}", arm=b['arm'], proxy_class=cls,
                              parent_id=b['parent_id'], query_id=b['query_id'], bundle_id=b['bundle_id'],
                              is_best_event_candidate=bool(c['hits_best_event'])))
    for pid, arms in by_parent.items():
        if set(arms) != set(ARMS):
            raise ValueError(f'parent {pid} lacks a bundle for some arm')
    parents = []
    for pid in sorted(by_parent):
        a2, b1 = by_parent[pid]['A2'], by_parent[pid]['B1']
        parents.append(dict(unit_id=pid, stratum=parent_stratum(a2, b1), query_id=a2['query_id'],
                            proxy_A2=int(a2['proxy_best_reference_hit']),
                            proxy_B1=int(b1['proxy_best_reference_hit']),
                            C_star_A2=list(a2['best_event_candidate_ids']),
                            C_star_B1=list(b1['best_event_candidate_ids'])))
    M = len(parents)
    delta_proxy = sum(p['proxy_B1'] - p['proxy_A2'] for p in parents) / M
    summary = dict(M=M, n_candidates=len(frame), delta_proxy=delta_proxy,
                   candidate_strata=dict(sorted(Counter(r['stratum'] for r in frame).items())),
                   parent_strata=dict(sorted(Counter(p['stratum'] for p in parents).items())),
                   C_star_sizes={arm: dict(sorted(Counter(len(p['C_star_' + arm]) for p in parents).items()))
                                 for arm in ('A2', 'B1')})
    return frame, parents, delta_proxy, summary


def frame_hash(rows):
    return sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
