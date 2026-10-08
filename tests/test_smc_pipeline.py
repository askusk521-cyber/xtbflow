"""Additional pre-analysis contracts: stage layout, clocks, scoring failure paths."""
from unittest.mock import patch

import numpy as np
import torch

from xtbflow.v1.clocks import clock_grid, observation_index
from xtbflow.v1.xtb_score import score_candidate
from xtbflow.v1.smc import score_key, selection_plan


def test_checkpoint_grid_and_population_batch_layout():
    tb, tx = clock_grid('geometry_lead3', 50)
    assert [observation_index(tx, s) for s in (.6, .7, .8, .9)] == [30, 35, 40, 45]
    assert np.allclose([tb[k] for k in (30, 35, 40, 45)], [.216, .343, .512, .729])
    pairs = [(p, j) for p in range(62) for j in range(32)]
    chunks = [pairs[k:k+256] for k in range(0, len(pairs), 256)]
    assert len(chunks) == 8
    for chunk in chunks:
        for p in {p for p, j in chunk}:
            assert [j for q, j in chunk if q == p] == list(range(32))


def test_raw_independent_of_relaxation_direction_failure():
    x = np.array([[-.4, 0, 0], [.4, 0, 0]])
    b = np.array([[0, 1], [1, 0]])
    with patch('xtbflow.v1.xtb_score.xtb_energy_kcal', return_value=(3., 'ok')):
        r = score_candidate([1, 1], x, b, b, 1., {}, gradient_job=lambda args: (None, 'unused'))
        raw = score_candidate([1, 1], x, b, b, 1., {}, relaxed=False)
    assert r['status'] == 'invalid_direction' and r['barrier'] is None
    assert r['raw_barrier'] == raw['barrier'] == 2.
    assert raw['status'] == 'ok'
    assert score_key(r) == (2, float('inf'))


def test_collapsed_candidate_fails_before_quantum_call():
    x = np.zeros((2, 3)); b = np.zeros((2, 2))
    with patch('xtbflow.v1.xtb_score.xtb_energy_kcal') as energy:
        r = score_candidate([1, 1], x, b, b, 0., {}, relaxed=False)
    energy.assert_not_called()
    assert r['status'] == 'invalid_geometry' and r['calls'] == 0


def test_force_failure_preserves_raw_but_does_not_rank_as_success():
    x = np.array([[-.4, 0, 0], [.4, 0, 0]])
    br = np.array([[0, 1], [1, 0]])
    with patch('xtbflow.v1.xtb_score.xtb_energy_kcal', return_value=(3., 'ok')):
        r = score_candidate([1, 1], x, br * 2, br, 1., {}, gradient_job=lambda args: (None, 'scf_failed:RuntimeError'))
    assert r['status'] == 'scf_failed:RuntimeError' and r['raw_barrier'] == 2.
    assert r['barrier'] is None and r['calls'] == 2
    good = dict(status='ok', barrier=1e6, curvature=1.)
    scores = [r] * 16 + [good] * 16
    s, clones = selection_plan('p', 0, scores, 35, 'relaxed', curvature_layers=False)
    assert set(s) == set(range(16, 32))
