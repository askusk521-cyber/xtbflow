"""Pure selection/mapping and execution guards; no quantum calculations."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('o2', Path(__file__).parents[1] / 'scripts/review_open_world_dft.py')
o2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(o2)


def fixture_rows():
    delta, verdicts = [], []
    for i in range(12):
        pid, cid = f'p{i}', f'c{i}'
        delta.append(dict(parent_id=pid, candidate_id=cid, delta_ea_kcal=float(i)))
        verdicts.extend([dict(group='BASELINE', parent_id=pid, id=f'r{i}', status='STRICT_JOINT_GRAPH_VALID'),
                         dict(group='OUTSIDE_CATALOGUE_EVENT', parent_id=pid, id=cid, status='STRICT_JOINT_GRAPH_VALID')])
    return dict(delta_ea=dict(rows=delta)), verdicts


def test_selection_order_and_independence():
    results, verdicts = fixture_rows()
    expected = o2.select_sample(results, verdicts)
    assert len(expected) == 10
    assert [r['parent_id'] for r in expected] == [f'p{i}' for i in range(10)]
    results['delta_ea']['rows'].reverse()
    assert o2.select_sample(results, list(reversed(verdicts))) == expected


def test_selection_rejects_gate_failure():
    results, verdicts = fixture_rows()
    verdicts[0]['status'] = 'IRC_LIMIT'
    with pytest.raises(ValueError, match='strict paired'):
        o2.select_sample(results, verdicts)


def test_selection_ties_sha256():
    results, verdicts = fixture_rows()
    for row in results['delta_ea']['rows']:
        row['delta_ea_kcal'] = 0
    expected = sorted((f'c{i}' for i in range(12)), key=lambda s: o2.sha256(s.encode()).hexdigest())[:10]
    assert [r['candidate_id'] for r in o2.select_sample(results, verdicts)] == expected


def test_reference_mapping():
    p = dict(permutations=[[2, 0, 1]])
    r = dict(x_ts=np.arange(9).reshape(3, 3).tolist(), b_p=np.arange(9).reshape(3, 3).tolist())
    x, b = o2.reference_start(p, r)
    assert x == [r['x_ts'][i] for i in (2, 0, 1)]
    assert b == [[r['b_p'][i][j] for j in (2, 0, 1)] for i in (2, 0, 1)]


def test_digest(tmp_path):
    p = tmp_path / 'input'
    p.write_bytes(b'abc')
    assert o2.digest(p) == o2.sha256(b'abc').hexdigest()


def test_no_gpu_outside_slurm(monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(RuntimeError, match='Slurm'):
        o2.run(SimpleNamespace())


def test_dirty_source_guard(monkeypatch):
    monkeypatch.setattr(o2.subprocess, 'check_output', lambda args, **kw: 'sha' if 'rev-parse' in args else ' M tracked')
    with pytest.raises(RuntimeError, match='clean'):
        o2.source_state()
