import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('smc_oracle',Path(__file__).resolve().parents[1]/'scripts/followup_smc_oracle.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def records():
    return [dict(batch='g1-0-0000.pt',start=0,parent_id='p',seed=0,step=35,method='aimnet',proposal=j,
                 score=dict(status='ok',barrier=float(j))) for j in range(32)]


def test_plans_reuse_selection_and_preserve_slots():
    r=records();p=m.make_plans(r)[0]
    assert p['survivors']==list(range(16))
    assert p['clones']==list(zip(range(16,32),range(16)))
    assert p['arm']=='aimnet_0.7'
    assert m.make_plans(list(reversed(r)))==[p]


def test_plan_rejects_duplicate_or_missing():
    with pytest.raises(ValueError):m.make_plans(records()[:-1])
    with pytest.raises(ValueError):m.make_plans(records()+records()[:1])


def test_failures_last_and_tie_hash():
    r=records()
    for row in r:row['score']=dict(status='ok',barrier=0.)
    r[0]['score']=dict(status='failed',barrier=None)
    p=m.make_plans(r)[0]
    assert 0 not in p['survivors']
    from xtbflow.v1.smc import tie_hash
    assert p['survivors']==sorted(range(1,32),key=lambda j:tie_hash('p',0,j))[:16]
