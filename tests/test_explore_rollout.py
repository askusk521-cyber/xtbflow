import numpy as np
import pytest
import torch

from xtbflow.m0.model import ReactionFlowNet
from xtbflow.physics.saddle import saddle_guidance
from xtbflow.v1.explore_rollout import event_direction,recorded_rollout,saddle_reflect
from xtbflow.v1.geometry_information import predicted_endpoints
from xtbflow.v1.interfaces import Query,State
from xtbflow.v1.sampler import rollout


def tiny_case(n=4,atoms=3,seed=0):
    torch.manual_seed(seed)
    net=ReactionFlowNet('joint',dual_time=True,scalar_dim=16,vector_dim=4,edge_dim=8,n_layers=1).eval()
    z=torch.tensor([[1,2,3]]*n);mask=torch.ones(n,atoms,dtype=torch.bool)
    x_r=torch.randn(n,atoms,3);b_r=torch.zeros(n,atoms,atoms);b_r[:,0,1]=b_r[:,1,0]=1
    q=Query(tuple(range(n)),z,z,mask,x_r,b_r)
    s=State(b_r+.3*torch.randn(n,atoms,atoms),x_r+.2*torch.randn(n,atoms,3),torch.zeros(n),torch.zeros(n))
    return net,q,s


@pytest.mark.parametrize('path',['sync','geometry_lead3','event_delay50'])
def test_recorded_rollout_reproduces_v1a_rollout_without_push(path):
    net,q,s=tiny_case()
    ref=rollout(net,q,s,role='joint',path=path,n_steps=10)
    out=recorded_rollout(net,q,s,path=path,n_steps=10)
    assert torch.equal(out['b_trace'],ref.b_trace) and torch.equal(out['x_trace'],ref.x_trace)
    assert torch.equal(out['b_hat'][-1],ref.state.b) and out['applied'].sum()==0
    if path=='sync':  # direct velocities agree with trace-difference recovery
        tb,tx=out['tb'],out['tx']
        assert torch.allclose(out['b_hat'],predicted_endpoints(ref.b_trace,tb),atol=1e-4)


def test_push_moves_geometry_only_inside_window():
    net,q,s=tiny_case()
    calls=[]
    def push(k,bh,xh,x,dt):
        calls.append(k);d=torch.zeros_like(x);d[:,0,0]=.01;d[:,1,0]=-.01
        return d,torch.ones(len(x),dtype=torch.bool)
    out=recorded_rollout(net,q,s,path='sync',n_steps=10,push=push,window=(.5,.8))
    assert calls==[5,6,7] and out['applied'].tolist()==[3.]*4
    ref=rollout(net,q,s,role='joint',path='sync',n_steps=10)
    assert torch.equal(out['x_trace'][:6],ref.x_trace[:6]) and not torch.equal(out['x_trace'][6],ref.x_trace[6])


def test_event_direction_signs_and_degenerate_case():
    x=torch.tensor([[[0.,0,0],[2.,0,0],[0,5.,0]]]);mask=torch.ones(1,3,dtype=torch.bool)
    b_r=torch.zeros(1,3,3)
    b=b_r.clone();b[0,0,1]=b[0,1,0]=1  # forming 0-1: atoms approach along x
    u,ok=event_direction(b,b_r,x,mask)
    assert ok.item() and u[0,0,0]>0 and u[0,1,0]<0 and abs(u[0,2]).max()<1e-6
    assert np.isclose(float(u.flatten(1).norm()),1.) and torch.allclose(u.sum(1),torch.zeros(1,3),atol=1e-6)
    u,_=event_direction(b_r-b,b_r,x,mask)  # breaking the same pair: atoms separate
    assert u[0,0,0]<0 and u[0,1,0]>0
    _,ok=event_direction(b_r,b_r,x,mask)
    assert not ok.item()


def test_saddle_reflect_matches_shared_guidance():
    f=torch.randn(2,3,3,dtype=torch.float64);u=torch.randn(2,3,3,dtype=torch.float64)
    u=u/u.flatten(1).norm(dim=1)[:,None,None]
    g=saddle_reflect(f,u)
    for k in range(2):
        assert torch.allclose(g[k],saddle_guidance(f[k],u[k]))
    along=(g*u).sum((1,2));assert torch.allclose(along,-(f*u).sum((1,2)))
