import pytest
import torch
from xtbflow.m0.sampler import sample_joint, sample_cascade


def rv():
    return dict(z=torch.ones(1,4,dtype=torch.long),atom_mask=torch.ones(1,4,dtype=torch.bool),
                x_r=torch.zeros(1,4,3),b_r=torch.zeros(1,4,4),index=torch.tensor([0]))


class ZeroNet:
    def __call__(self,z,mask,x,xr,b,br,t):
        return {'b_vel':torch.zeros_like(b),'x_vel':torch.zeros_like(x)}


def test_candidate_noise_not_identical_and_paired():
    joint=sample_joint(ZeroNet(),rv(),4,2,.5,.5,torch.Generator().manual_seed(3))
    cascade=sample_cascade(ZeroNet(),ZeroNet(),rv(),4,2,.5,.5,torch.Generator().manual_seed(3))
    assert torch.equal(joint['b_raw'],cascade['b_raw'])
    assert torch.equal(joint['x'],cascade['x'])
    assert not torch.equal(joint['x'][0],joint['x'][1])
    assert not torch.equal(joint['b_raw'][0],joint['b_raw'][1])


@pytest.mark.parametrize('key',['x_ts','x_p','b_p'])
def test_labels_rejected(key):
    d=rv();d[key]=torch.zeros(1)
    with pytest.raises(ValueError):
        sample_joint(ZeroNet(),d,2,2,.5,.5,torch.Generator())
