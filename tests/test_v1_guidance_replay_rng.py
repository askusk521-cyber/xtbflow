import torch

from xtbflow.v1.guidance import paired_pulses,rms_atom,unit_geometry_direction
from xtbflow.v1.interfaces import Query,State
from xtbflow.v1.sampler import initial_state,rollout,subset_query


class Coupled(torch.nn.Module):
    def forward(self,z,atom_mask,x_cur,x_r,b_cur,b_r,t,t_x=None):
        # Geometry-dependent event velocity with zero sum across the whole BE block.
        v=x_cur.square().sum(-1)
        matrix=v[:,:,None]+v[:,None,:]
        matrix-=matrix.mean((1,2),keepdim=True)
        return {'b_vel':matrix,'x_vel':.1*(x_r-x_cur)}


class Score:
    def components(self,q,b,x,tb,tx):
        phi=x.square().sum((1,2))
        return dict(phi=phi,supported=torch.ones(len(x),dtype=torch.bool),
                    mean_kcal=phi,sd_kcal=phi*0)


def query():
    x=torch.tensor([[[0.,0,0],[1,0,0],[0,2,0]],[[0.,0,0],[1,0,0],[0,0,0]]])
    mask=torch.tensor([[True]*3,[True,True,False]])
    x=(x-(x*mask[...,None]).sum(1,keepdim=True)/mask.sum(1)[:,None,None])*mask[...,None]
    z=torch.ones(2,3,dtype=torch.long)*mask
    return Query(('q1','q2'),z,z,mask,x,torch.zeros(2,3,3))


def test_addressed_noise_survives_batch_changes_and_extra_diagnostics():
    q=query();a=initial_state(q,[0,7],0)
    _=initial_state(q,[12,11],1,namespace='rarity_pilot')
    b=initial_state(subset_query(q,[1]),[7],0)
    assert torch.equal(a.b[1],b.b[0]) and torch.equal(a.x[1],b.x[0])


def test_alpha_zero_and_every_step_replay():
    q=query();s=initial_state(q,[0,1],0);net=Coupled()
    b0=rollout(net,q,s,n_steps=5)
    b1=rollout(net,q,s,n_steps=5,score=Score(),alpha=0)
    b2=rollout(net,q,s,n_steps=5,score=Score(),alpha=0,replay=b0.b_trace)
    assert torch.equal(b0.b_trace,b1.b_trace) and torch.equal(b0.x_trace,b1.x_trace)
    assert torch.equal(b0.b_trace,b2.b_trace) and torch.equal(b0.x_trace,b2.x_trace)
    shadow=b0.b_trace.clone()
    guided=rollout(net,q,s,n_steps=5,score=Score(),alpha=.3,replay=b0.b_trace)
    assert torch.equal(guided.b_trace,shadow) and torch.equal(b0.b_trace,shadow)
    assert not torch.equal(guided.x_trace,b0.x_trace)


def test_geometry_guidance_changes_event_only_on_later_step():
    q=query();s=initial_state(q,[0,1],0);net=Coupled()
    b0=rollout(net,q,s,n_steps=5)
    b1=rollout(net,q,s,n_steps=5,score=Score(),alpha=.3)
    assert torch.equal(b0.b_trace[1],b1.b_trace[1])
    assert not torch.equal(b0.b_trace[2],b1.b_trace[2])


def test_pulse_common_amplitude_and_zero_direction_not_dropped():
    mask=torch.ones(2,12,dtype=torch.bool)
    g=torch.zeros(2,12,3);g[0,0,0]=100
    unit,valid=unit_geometry_direction(g,mask)
    random=torch.randn(2,12,3)
    ds,dr,actual=paired_pulses(unit,random,mask,.2,valid)
    assert torch.allclose(rms_atom(ds,mask),actual,atol=1e-7)
    assert torch.allclose(rms_atom(dr,mask),actual,atol=1e-7)
    assert actual[0]<.2 and actual[1]==0
    assert ds.norm(dim=-1).max()<=.6+1e-7
    assert dr.norm(dim=-1).max()<=.6+1e-7
