import numpy as np
import torch
from xtbflow.m0.model import ReactionFlowNet, count_parameters
from xtbflow.v1.clocks import clock_grid,increments,observation_index
from xtbflow.v1.training import generator_inputs


def test_clock_integrals_and_grid_alignment():
    for path in ['sync','event_lead2','geometry_lead2','geometry_lead3','geometry_lead4','event_delay30','event_delay50']:
        b,x=clock_grid(path)
        db,dx=increments(path)
        assert np.isclose(sum(db),1) and np.isclose(sum(dx),1)
        k=observation_index(x,.35)
        assert x[k]>=.35 and x[k-1]<.35
        assert observation_index(x,1)==50
        assert b[0]==x[0]==0 and b[-1]==x[-1]==1 and np.all(np.diff(b)>=0) and np.all(np.diff(x)>0)
        if not path.startswith('event_delay'):assert np.all(np.diff(b)>0)
    b,x=clock_grid('event_delay50')
    assert np.all(b[x<=.5]==0) and b[-1]==1 and np.allclose(b[x>.5],(x[x>.5]-.5)/.5)
    b,x=clock_grid('geometry_lead2')
    assert np.all(x[1:-1]>b[1:-1])


def test_independent_time_training_and_old_baseline():
    n=8
    batch=dict(z=torch.ones(n,2,dtype=torch.long),atom_mask=torch.ones(n,2,dtype=torch.bool),
               x_r=torch.zeros(n,2,3),x_ts=torch.ones(n,2,3),
               b_r=torch.zeros(n,2,2),b_p=torch.ones(n,2,2))
    g=torch.Generator().manual_seed(10)
    inputs,targets=generator_inputs('joint',batch,1.,.5,g)
    assert not torch.equal(inputs['t'],inputs['t_x'])
    base,_=generator_inputs('baseline',batch,1.,.5,g)
    assert 't_x' not in base
    model=ReactionFlowNet('joint',dual_time=True,scalar_dim=16,vector_dim=4,edge_dim=8,n_layers=1)
    a=model(**inputs)
    b=model(**dict(inputs,t_x=1-inputs['t_x']))
    assert not torch.allclose(a['b_vel'],b['b_vel'])
    assert torch.allclose(a['b_vel'].sum((1,2)),torch.zeros(n),atol=1e-6)


def test_parameter_matching():
    f=ReactionFlowNet('joint',dual_time=True)
    g=ReactionFlowNet('event',scalar_dim=92,vector_dim=23)
    h=ReactionFlowNet('geometry',scalar_dim=92,vector_dim=23)
    assert abs(count_parameters(g)+count_parameters(h)-count_parameters(f))/count_parameters(f)<.05
