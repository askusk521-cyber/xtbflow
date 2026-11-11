import pytest
import torch

from xtbflow.sampling.amplitude_backtracking import bounded_backtrack, geometry_step
from xtbflow.sampling.guidance_fixture import fixture_query
from xtbflow.sampling.integrated_guidance import prepare_injection, model_values


class Linear(torch.nn.Module):
    def __init__(self):
        super().__init__();self.calls=[]
    def forward(self, **kw):
        self.calls.append((kw['b_cur'].clone(),kw['t'].clone(),kw['t_x'].clone()))
        x=kw['x_cur'];a=x[:,1,0]-x[:,0,0]
        return dict(x_vel=-.1*x,b_vel=a[:,None,None]*x.new_tensor([[1.,-1.],[-1.,1.]]))


def prepared():
    q,_,s=fixture_query();net=Linear()
    item=prepare_injection(net,q,s,.5,.25)
    y,vb,_=model_values(net,q,item.base)
    g=s.x/s.x.norm()
    return q,net,item,g,dict(y0=y,vb0=vb,jx=.925*g),dict(qualified=True,endpoint_response_resolved=True,endpoint_projection=.925)


@pytest.mark.parametrize('norms,selected,attempts', [([1.,1.1,.8],.5,2),([1.,.8],1.,1),([1.,1.1,1.1,1.1],None,3)])
def test_first_trial_is_shared_fixed_baseline_and_base_never_advances(norms,selected,attempts):
    q,net,item,g,derivative,qualified=prepared();calls=[];before=len(net.calls)
    def physics(y,label):
        calls.append(label)
        return dict(status='completed',force=(g[0]*norms[len(calls)-1]).tolist())
    state,row=bounded_backtrack(net,q,item,g,derivative,qualified,physics,context='fixture')
    assert row['selected_lambda']==selected
    assert len(calls)==1+attempts and len(net.calls)-before==attempts
    assert [t['lambda_value'] for t in row['trials']]==[1.,.5,.25][:attempts]
    assert torch.equal(state.b,item.base.b)
    for b,tb,tx in net.calls[before:]:
        assert torch.equal(b,item.base.b) and torch.equal(tb,item.base.t_b) and torch.equal(tx,item.base.t_x)
    if selected is None:
        assert torch.equal(state.x,item.base.x)
    else:
        assert row['accepted_phi']<=.99*row['baseline_phi']


def test_nonpositive_transport_spends_nothing_and_does_not_flip_direction():
    q,net,item,g,derivative,qualified=prepared()
    def forbidden(*args):raise AssertionError('must_not_call')
    state,row=bounded_backtrack(net,q,item,g,derivative,qualified|dict(endpoint_projection=-.1),forbidden,context='fixture')
    assert state is item.base and row['reason']=='amplitude_only_repair_not_supported'


def test_unknown_cost_stops_before_any_trial_and_is_not_a_zero_cost_rejection():
    q,net,item,g,derivative,qualified=prepared();before=len(net.calls)
    def unknown(*args):raise RuntimeError('backend_termination_unknown')
    with pytest.raises(RuntimeError,match='termination_unknown'):
        bounded_backtrack(net,q,item,g,derivative,qualified,unknown,context='fixture')
    assert len(net.calls)==before


def test_failed_trial_is_preserved_with_no_substitute_or_extra_lambda():
    q,net,item,g,derivative,qualified=prepared();calls=[]
    def physics(y,label):
        calls.append(label)
        return dict(status='completed',force=g[0].tolist()) if len(calls)==1 else dict(status='calculation_failed')
    state,row=bounded_backtrack(net,q,item,g,derivative,qualified,physics,context='fixture')
    assert len(calls)==4 and len(row['trials'])==3 and state is item.base
    assert all(t['physical_status']=='calculation_failed' for t in row['trials'])


def test_common_max_atom_scaling_is_retained():
    d=torch.zeros(1,16,3,dtype=torch.float64);d[0,0,0]=4.
    delta,actual=geometry_step(d,torch.ones(1,16,dtype=torch.bool),.001)
    assert actual.item()==pytest.approx(.00075) and delta.norm().item()==pytest.approx(.003)
