from dataclasses import replace
import json

import pytest
import torch

from xtbflow.sampling.guidance_fixture import fixture_query
from xtbflow.sampling.integrated_guidance import prepare_injection
from xtbflow.sampling.physical_diagnostic import StageMeter, measure_direction, physical_fields, rms_atom


class LinearCoupled(torch.nn.Module):
    def forward(self, **kw):
        x=kw['x_cur'];a=x[:,1,0]-x[:,0,0]
        return dict(x_vel=-.1*x,b_vel=a[:,None,None]*x.new_tensor([[1.,-1.],[-1.,1.]]))


def caps():
    return dict(forward=45,ef=3,jvp=6,reverse=12,wall_seconds=300,cpu_seconds=300,output_bytes=1024**2)


def test_direction_measurement_uses_updated_BE_and_next_clocks(tmp_path):
    query,_,state=fixture_query()
    model=LinearCoupled();meter=StageMeter(tmp_path,caps());meter.install_model(model)
    with torch.no_grad():
        injection=prepare_injection(model,query,state,.5,.25)
    direction=state.x.clone()
    row,cache=measure_direction(model,query,injection.base,direction,meter,context='F',scales=[1e-4,3e-4,1e-3])
    assert row['qualified'] and row['event_response_resolved']
    assert row['endpoint_gain']==pytest.approx(.925)
    assert row['endpoint_cosine']==pytest.approx(1.)
    assert row['l2_unit_rms']==pytest.approx(1/2**.5)
    assert meter.summary()['counts']==dict(forward=8,ef=0,jvp=1)
    assert meter.reverse_completed==2
    torch.testing.assert_close(cache['y0'],.925*injection.base.x)


def test_reflected_field_has_same_units_norm_and_fixed_event_direction():
    query,_,state=fixture_query()
    force=state.x*2
    F,G,u,valid=physical_fields(force,state.b+torch.tensor([[[1.,-1.],[-1.,1.]]]),query.b_r,state.x,query.atom_mask)
    assert valid
    assert G.norm()==pytest.approx(F.norm())
    assert G.sum(1).abs().max()==0
    scaled=physical_fields(force*10,state.b+torch.tensor([[[1.,-1.],[-1.,1.]]]),query.b_r,state.x,query.atom_mask)[1]
    torch.testing.assert_close(scaled,10*G)
    assert rms_atom(G,query.atom_mask)[0]==pytest.approx(G.norm()/2**.5)


def test_failed_started_work_is_preserved_and_budget_rejects_before_new_call(tmp_path):
    limits=caps()|dict(forward=1,jvp=1,reverse=2)
    meter=StageMeter(tmp_path,limits)
    first=meter.begin('forward','failed_model')
    meter.end(first,'failed')
    with pytest.raises(RuntimeError,match='forward_budget'):
        meter.begin('forward','must_not_start')
    assert json.loads((tmp_path/'work.json').read_text())['counts']['forward']==1
    derivative=meter.begin('jvp','failed_derivative')
    meter.end(derivative,'failed')
    assert meter.reverse_reserved==2 and meter.reverse_completed==0
    assert len(meter.rows)==2


def test_zero_force_is_explicit_and_spends_no_model_work(tmp_path):
    query,_,state=fixture_query();meter=StageMeter(tmp_path,caps())
    row,cache=measure_direction(LinearCoupled(),query,state,torch.zeros_like(state.x),meter,context='G',scales=[1e-4,3e-4,1e-3])
    assert not row['qualified'] and cache=={}
    assert meter.summary()['counts']==dict(forward=0,ef=0,jvp=0)
