import numpy as np
import pytest

from xtbflow.v1 import streams
from xtbflow.v1.costs import prefixes
from xtbflow.v1.interfaces import query_from_parents


def run(arm,cap,monkeypatch,invalid=False):
    q=query_from_parents([dict(query_id='test',atomic_numbers=[1,1],charge=0,multiplicity=1,
                              x_r=[[0,0,0],[1,0,0]],b_r=[[0,1],[1,0]])])
    monkeypatch.setattr(streams,'decode_event',lambda q,s:None if invalid else np.array([[1,0],[0,1]]))
    cfg=dict(path='sync',alpha_x=.15,alpha_b=.1,guidance_start=.5,guidance_stop=.95,
             sigma_b=1.,sigma_x=.5,a2_times=[.4,.7],cap_units=cap)
    out={};p=streams.logical_stream(q,arm,0,{k:1. for k in ('f','g','h','X','X_back','E','E_back')},cfg,out)
    value=None;requests=[]
    while True:
        try:r=p.send(value)
        except StopIteration:break
        requests.append((r.role,r.start,r.end,r.proposal.number))
        value=float(r.proposal.number) if r.role=='rank' else None
    return out,requests


def test_b0_cost_prefix_and_partial(monkeypatch):
    out,requests=run('B0',325.,monkeypatch)
    assert len(out['candidates'])==6
    assert [c['completion_units'] for c in out['candidates']]==[50,100,150,200,250,300]
    assert out['spent_units']==325 and out['attempts'][-1]['status']=='BUDGET_PARTIAL'
    assert len(prefixes(out['candidates'])['4'])==4


def test_a2_breadth_first_promotions_and_new_round(monkeypatch):
    out,requests=run('A2',3200.,monkeypatch)
    assert requests[:4]==[('event',0,50,j) for j in range(4)]
    assert requests[4:8]==[('geometry',0,20,j) for j in range(4)]
    assert requests[8:12]==[('rank',20,20,j) for j in range(4)]
    first_complete=[r for r in requests if r[0]=='geometry' and r[2]==50 and r[3]<4]
    assert len(first_complete)==2
    assert len([r for r in requests if r[0]=='event' and 4<=r[3]<12])==8
    assert out['spent_units']<=3200
    assert sum(r['cost_units'] for r in out['operations'])==out['spent_units']
    assert len({r['attempt_id'] for r in out['attempts']})==len(out['attempts'])
    assert any(r['status']=='ELIMINATED_STAGE_0' for r in out['attempts'])


@pytest.mark.parametrize('arm',['A0','A1','A2'])
def test_illegal_proposals_cost_but_never_become_candidates(arm,monkeypatch):
    out,requests=run(arm,1200.,monkeypatch,True)
    assert not out['candidates'] and out['spent_units']>0
    assert all(r[0]=='event' for r in requests)
    assert any(r['status']=='INVALID_EVENT' for r in out['attempts'])


def test_guidance_backward_cost_and_no_tail_overspend(monkeypatch):
    out,requests=run('B1',200.,monkeypatch)
    assert len(out['candidates'])==1
    assert out['spent_units']<=200
    assert any(r['operation']=='X_back' and r['count']==3 for r in out['operations'])
    # 25 un-guided steps + 23 * (1+3+3) + two un-guided = 188.
    # One complete candidate then an affordable partial second candidate.
    assert out['attempts'][-1]['status']=='BUDGET_PARTIAL'
