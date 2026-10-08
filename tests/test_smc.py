"""Selection contracts and force-unit tests for the exploratory SMC protocol."""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from xtbflow.v1.interfaces import query_from_parents
from xtbflow.v1.smc import population_metrics, restart_state, score_key, selection_plan
from xtbflow.v1.xtb_score import gradient_to_force, score_candidate
from xtbflow.calculators.xtb_oracle import BOHR_IN_ANGSTROM
from xtbflow.v1.geometry_information import HARTREE_TO_KCAL, xtb_energy_kcal


def query():
    return query_from_parents([dict(query_id='h2', atomic_numbers=[1, 1], charge=0,
                                   multiplicity=1, x_r=[[-.4, 0, 0], [.4, 0, 0]],
                                   b_r=[[0, 1], [1, 0]])])


def test_plan_exactly_one_clone_and_deterministic_ties():
    scores = [dict(status='ok', barrier=1., curvature=-1.) for _ in range(32)]
    for scorer in ('raw', 'relaxed', 'random'):
        s, c = selection_plan('p', 0, scores, 35, scorer)
        assert len(s) == len(set(s)) == 16
        assert sorted(a for _, a in c) == sorted(s)
        assert set(d for d, _ in c) == set(range(32)) - set(s)
        assert (s, c) == selection_plan('p', 0, scores, 35, scorer)
    with pytest.raises(ValueError):
        selection_plan('p', 0, scores[:30], 35, 'raw')


def test_score_layers_and_fallback():
    a = dict(status='ok', barrier=99., curvature=-1.)
    b = dict(status='ok', barrier=-99., curvature=1.)
    assert score_key(a) < score_key(b) < score_key(dict(status='failed'))
    assert score_key(b, curvature_layers=False) < score_key(a, curvature_layers=False)


def test_restart_endpoints_and_noise_pairing():
    q = query()
    bh, xh = q.b_r + 2, q.x_r * 2
    one = restart_state(q, bh, xh, [7], 2, 35, 1., 1.)
    assert torch.equal(one.b, bh)
    assert torch.equal(one.x, xh)
    zero = restart_state(q, bh, xh, [7], 2, 35, 0., 0.)
    other = restart_state(q, bh * 8, xh * 8, [7], 2, 35, 0., 0.)
    assert torch.equal(zero.b, other.b) and torch.equal(zero.x, other.x)
    assert not torch.equal(zero.b, q.b_r)
    assert torch.allclose(zero.b, zero.b.transpose(1, 2))
    assert torch.allclose(zero.x.sum(1), torch.zeros(1, 3), atol=1e-6)
    half = restart_state(q, bh, xh, [7], 2, 35, .3, .7)
    assert torch.allclose(half.b, .7 * zero.b + .3 * bh)
    assert torch.allclose(half.x, .3 * zero.x + .7 * xh, atol=1e-6)


def test_hit_at_k_deduplicates_and_keeps_legal_unknowns():
    def row(c, energy):
        return dict(channel_id=c, score=dict(status='ok', barrier=energy, curvature=-1.),
                    event_utility=0., hits_best_reference=False)
    m = population_metrics([row('a', 1), row('a', 0), row('best', 2), row(None, -10)], ['best'])
    assert m['hit@1'] == 0 and m['hit@2'] == m['hit@4'] == 1
    assert m['H'] == 2 / 3 and m['distinct_legal_events'] == 2 and m['legal_rate'] == .75
    assert population_metrics([row(None, 0)], ['best'])['H'] == 0


def test_force_conversion_and_saddle_sign():
    assert gradient_to_force(np.array([1.]))[0] == -HARTREE_TO_KCAL / BOHR_IN_ANGSTROM
    q = query()
    def harmonic(args):
        # E = |x|^2 / 2 kcal/mol; force=-x; reflected force pushes H2 apart.
        return np.asarray(args[1]) * BOHR_IN_ANGSTROM / HARTREE_TO_KCAL, 'ok'
    def energy(z, x, cfg):
        return float((x * x).sum() / 2), 'ok'
    with patch('xtbflow.v1.xtb_score.xtb_energy_kcal', energy):
        r = score_candidate([1, 1], q.x_r[0], q.b_r[0] * 2, q.b_r[0], 0., {}, gradient_job=harmonic)
    assert r['status'] == 'ok' and r['energy_drop'] < 0
    assert r['curvature'] == pytest.approx(1., abs=1e-10)
    assert r['calls'] == 24


def test_real_h2_gradient_units():
    import json
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from explore_geometry_information import _force_job
    cfg = json.loads((Path(__file__).resolve().parents[1] / 'configs/explore/geometry_information.json').read_text())['xtb']
    x = np.array([[-.4, 0, 0], [.4, 0, 0]])
    g, status = _force_job(([1, 1], x, cfg))
    assert status == 'ok'
    h = 1e-4
    xp, xm = x.copy(), x.copy()
    xp[1, 0] += h
    xm[1, 0] -= h
    ep, sp = xtb_energy_kcal([1, 1], xp, cfg)
    em, sm = xtb_energy_kcal([1, 1], xm, cfg)
    assert sp == sm == 'ok'
    assert gradient_to_force(g)[1, 0] == pytest.approx(-(ep - em) / (2 * h), abs=.02, rel=1e-3)
