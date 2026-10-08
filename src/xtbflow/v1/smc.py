"""Deterministic exploratory SMC selection; no catalogue-dependent selection."""
import hashlib

import torch

from xtbflow.m0.flow import geometry_noise, symmetric_noise
from .guidance import centered
from .interfaces import State
from .rng import addressed_seed


def tie_hash(*parts):
    return hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()


def score_key(score, scorer='relaxed', curvature_layers=True):
    if score['status'] != 'ok':
        return (2 if scorer == 'relaxed' and curvature_layers else 1, float('inf'))
    layer = int(score['curvature'] >= 0) if scorer == 'relaxed' and curvature_layers else 0
    return layer, score['barrier']


def selection_plan(parent_id, seed, scores, step, scorer, retain=16, curvature_layers=True):
    """Return survivor slots and (destination, ancestor) clone assignments."""
    if len(scores) != 2 * retain:
        raise ValueError('exactly one clone per survivor required')
    def key(j):
        if scorer == 'random':
            return (tie_hash(parent_id, seed, j, step),)
        return (*score_key(scores[j], scorer, curvature_layers), tie_hash(parent_id, seed, j))
    ranked = sorted(range(len(scores)), key=key)
    survivors = ranked[:retain]
    return survivors, list(zip(ranked[retain:], survivors))


def restart_state(query, b_hat, x_hat, proposals, seed, step, tb, tx, sigma_b=1., sigma_x=.5):
    """Conditional restart; addressed noise is independent of selection arm."""
    if len(proposals) != len(query.query_id):
        raise ValueError('proposal count mismatch')
    b = query.b_r.clone()
    x = query.x_r.clone()
    for i, (qid, proposal) in enumerate(zip(query.query_id, proposals)):
        n = int(query.atom_mask[i].sum())
        mask = torch.ones(1, n, dtype=torch.bool)
        keys = dict(query=qid, training_seed=seed, proposal=int(proposal), checkpoint_step=step)
        gb = torch.Generator().manual_seed(addressed_seed('explore_smc_clone', noise_role='joint_b', **keys))
        gx = torch.Generator().manual_seed(addressed_seed('explore_smc_clone', noise_role='joint_x', **keys))
        b[i, :n, :n] += sigma_b * symmetric_noise(mask, gb)[0].to(b)
        x[i, :n] += sigma_x * geometry_noise(mask, gx)[0].to(x)
    b = (1 - tb) * b + tb * b_hat
    x = centered((1 - tx) * x + tx * x_hat, query.atom_mask)
    return State(b, x, torch.full((len(b),), float(tb), device=b.device),
                 torch.full((len(b),), float(tx), device=b.device))


def population_metrics(rows, best_channel_ids, curvature_layers=True):
    """Rank distinct legal events by their best candidate, not multiplicity."""
    events = {}
    for row in rows:
        channel = row['channel_id']
        if channel is not None:
            key = score_key(row['score'], curvature_layers=curvature_layers)
            events[channel] = min(events.get(channel, (float('inf'), float('inf'))), key)
    ranked = sorted(events, key=lambda c: (*events[c], tie_hash(c)))
    best = set(best_channel_ids)
    hits = {f'hit@{k}': float(bool(best.intersection(ranked[:k]))) for k in (1, 2, 4)}
    return dict(hits, H=sum(hits.values()) / 3, hit_infinity=float(bool(best.intersection(events))),
                event_utility=sum(r['event_utility'] for r in rows) / len(rows),
                distinct_legal_events=len(events), legal_rate=sum(r['channel_id'] is not None for r in rows) / len(rows),
                best_reference_rate=sum(bool(r['hits_best_reference']) for r in rows) / len(rows))
