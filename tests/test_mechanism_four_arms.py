from dataclasses import replace
import json

import pytest
import torch

from xtbflow.sampling.guidance_fixture import fixture_query
from xtbflow.sampling.mechanism_four_arms import generate_four_arms
from xtbflow.sampling.physical_diagnostic import StageMeter


class Coupled(torch.nn.Module):
    def forward(self,**kw):
        x=kw['x_cur'];a=x[:,1,0]-x[:,0,0]
        return dict(x_vel=-.1*x,b_vel=a[:,None,None]*x.new_tensor([[1.,-1.],[-1.,1.]]))


@pytest.mark.parametrize('zero_force',[False,True])
def test_four_arm_real_entry_counts_replay_and_terminal_nonzero_clock_doses(tmp_path,zero_force):
    q,_,initial=fixture_query();q.x_r=q.x_r*.05;initial=replace(initial,x=initial.x*.05)
    meter=StageMeter(tmp_path,dict(forward=162,ef=27,jvp=3,reverse=6,wall_seconds=300,cpu_seconds=300,output_bytes=8*1024**2))
    net=Coupled();meter.install_model(net);calls=[]
    def physical(arm,y,label):
        assert arm in ('T','F');calls.append((arm,label))
        row=meter.begin('ef',label);meter.end(row)
        return dict(status='completed',force=(torch.zeros_like(y[0]) if zero_force else y[0]).tolist())
    result=generate_four_arms(net,q,initial,physical,meter)
    assert torch.equal(result['traces']['N']['b'],result['traces']['C']['b'])
    assert torch.equal(result['states']['N'].b,result['states']['T'].b)
    assert result['work']['T']['forward']==0 and result['work']['T']['jvp']==0
    assert result['work']['N']['ef']==result['work']['C']['ef']==0
    assert meter.summary()['counts']['forward']<=162 and meter.summary()['counts']['ef']<=27
    if zero_force:
        for a in ('F','C','T'):
            assert torch.equal(result['states'][a].x,result['states']['N'].x)
            assert torch.equal(result['states'][a].b,result['states']['N'].b)
        assert meter.summary()['counts']==dict(forward=150,ef=6,jvp=0)
    else:
        assert any(d.norm()>0 for d in result['pushes'].values())
        assert not torch.equal(result['states']['F'].b,result['states']['C'].b)
        assert meter.summary()['counts']['forward']==156 and meter.summary()['counts']['jvp']==3
        # T's identity map can hit the literal RMS <= cap boundary at roundoff;
        # retain those extra trials, and reconcile independent backend invocations.
        assert meter.summary()['counts']['ef']==len(calls)
        assert 15<=len(calls)<=21
        assert [r['nominal_displacement']['rms_atom_angstrom'] for r in result['slots']['T']]==pytest.approx([.00084,.00204,.00324])
        assert len([label for arm,label in calls if arm=='T' and label.endswith('baseline')])==3
        for k in (10,25,40):
            torch.testing.assert_close(result['signals'][f'F:{k}']['base_b'],result['traces']['F']['b'][k+1])
