import numpy as np
import pytest
import torch

from xtbflow.v1 import streams
from xtbflow.v1.clocks import increments
from xtbflow.v1.interfaces import State,query_from_parents
from xtbflow.v1.sampler import rollout

WEIGHTS={'f':1.,'g':.9367,'h':.9084,'X':.65,'X_back':.66,'E':.63,'E_back':.58}


def query():
    return query_from_parents([dict(query_id='test',atomic_numbers=[1,1],charge=0,multiplicity=1,
                                    x_r=[[0,0,0],[1,0,0]],b_r=[[0,1],[1,0]])])


def run(arm,cap,monkeypatch,**nfe):
    monkeypatch.setattr(streams,'decode_event',lambda q,s:np.array([[1,0],[0,1]]))
    cfg=dict(path='sync',alpha_x=.3,alpha_b=.25,guidance_start=.5,guidance_stop=.95,
             sigma_b=1.,sigma_x=.5,a2_times=[.4,.7],cap_units=cap,**nfe)
    out={};p=streams.logical_stream(query(),arm,0,WEIGHTS,cfg,out)
    value=None;requests=[]
    while True:
        try:r=p.send(value)
        except StopIteration:break
        requests.append(r);value=float(r.proposal.number) if r.role=='rank' else None
    return out,requests


def test_default_programs_keep_fifty_steps_and_output_format(monkeypatch):
    for arm in ('A0','B0','B1'):
        out,requests=run(arm,400.,monkeypatch)
        assert 'nfe_steps' not in out
        assert all(r.n_steps==50 for r in requests) and requests[0].end==50


def test_b0_twenty_five_joint_steps(monkeypatch):
    out,requests=run('B0',3200.,monkeypatch,nfe_joint=25)
    assert out['nfe_steps']==dict(event=50,geometry=50,joint=25)
    assert [(r.role,r.start,r.end,r.n_steps) for r in requests[:2]]==[('joint',0,25,25)]*2
    assert len(out['candidates'])==128
    assert [c['completion_units'] for c in out['candidates'][:3]]==[25.,50.,75.]


def test_a0_twenty_five_plus_twenty_five(monkeypatch):
    out,requests=run('A0',3200.,monkeypatch,nfe_event=25,nfe_geometry=25)
    assert [(r.role,r.end,r.n_steps) for r in requests[:2]]==[('event',25,25),('geometry',25,25)]
    unit=25*(WEIGHTS['g']+WEIGHTS['h'])
    assert out['candidates'][0]['completion_units']==pytest.approx(unit)
    assert len(out['candidates'])==int(3200//unit)
    # Same addressed starting noise as the 50+50 program: proposal j, cascade group.
    assert out['candidates'][1]['attempt_id'].endswith('/A0/s0/p1')


@pytest.mark.parametrize('arm',['A1','A2','B1'])
def test_overrides_refused_for_guided_and_serial_arms(arm,monkeypatch):
    with pytest.raises(ValueError):run(arm,400.,monkeypatch,nfe_joint=25,nfe_event=25)


class ConstantVelocity(torch.nn.Module):
    """Opposite unit x velocities on two atoms, so centering keeps the displacement."""

    def __init__(self):
        super().__init__();self.calls=0

    def forward(self,**kw):
        self.calls+=1
        v=torch.zeros_like(kw['x_cur']);v[:,0,0]=1.;v[:,1,0]=-1.
        return dict(b_vel=torch.zeros_like(kw['b_cur']),x_vel=v)


@pytest.mark.parametrize('n',[25,50])
def test_rollout_integrates_actual_clock_increments(n):
    q=query();net=ConstantVelocity()
    zeros=torch.zeros(1)
    state=State(q.b_r.clone(),q.x_r.clone(),zeros,zeros)
    out=rollout(net,q,state,role='joint',path='sync',n_steps=n)
    tb,tx=increments('sync',n)
    assert net.calls==n and len(tb)==n
    assert tb.sum()==pytest.approx(1.) and tx.sum()==pytest.approx(1.)
    assert float(out.state.t_x[0])==pytest.approx(1.)
    # Separation along x moves by -2 * sum(dt) = -2 regardless of re-centering.
    gap=lambda x:(x[0,1,0]-x[0,0,0]).item()
    assert gap(out.state.x)-gap(q.x_r)==pytest.approx(-2.,abs=1e-5)
