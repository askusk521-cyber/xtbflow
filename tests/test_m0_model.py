import pytest
import torch
from xtbflow.m0.flow import active_pairs, project_conserved, symmetric_noise
from xtbflow.m0.model import ReactionFlowNet, match_cascade_width

CFG = dict(scalar_dim=32, vector_dim=8, n_layers=2)


def inputs():
    torch.manual_seed(0)
    mask = torch.tensor([[True]*6, [True]*4+[False]*2])
    r = torch.randn(2, 6, 3) * mask[..., None]
    b = torch.randn(2, 6, 6)
    b = (b+b.transpose(1, 2))/2 * active_pairs(mask)
    return dict(z=torch.tensor([[1,2,3,4,1,2],[1,2,3,4,0,0]]),
                atom_mask=mask, x_r=r, x_cur=r+torch.randn_like(r)*mask[...,None],
                b_r=b, b_cur=b+ symmetric_noise(mask), t=torch.tensor([.2,.8]))


def assert_close(a,b):
    torch.testing.assert_close(a,b,atol=1e-4,rtol=1e-4)


def test_project_conserved():
    d=inputs(); mask=d['atom_mask']; p=project_conserved(d['b_cur'],mask)
    assert_close(p.sum((1,2)),torch.zeros(2))
    assert_close(p,p.transpose(1,2))
    assert (p[~active_pairs(mask)]==0).all()
    assert_close(project_conserved(p,mask),p)


def test_symmetric_noise():
    mask=inputs()['atom_mask']; p=symmetric_noise(mask)
    assert_close(p,p.transpose(1,2))
    assert_close(p.sum((1,2)),torch.zeros(2))
    assert (p[~active_pairs(mask)]==0).all()


def test_rotation_translation():
    d=inputs(); net=ReactionFlowNet('joint',**CFG); a=net(**d)
    q,_=torch.linalg.qr(torch.randn(3,3))
    if torch.det(q)<0: q[:,0]*=-1
    moved=dict(d)
    for k in ('x_r','x_cur'): moved[k]=d[k]@q.T+torch.tensor([2.,3.,-1.])
    b=net(**moved)
    assert_close(a['b_vel'],b['b_vel']); assert_close(a['x_vel']@q.T,b['x_vel'])


def test_permutation():
    d=inputs(); net=ReactionFlowNet('joint',**CFG); a=net(**d)
    perms=[torch.tensor([3,0,5,2,1,4]),torch.tensor([3,1,0,2,4,5])]
    perm=dict(d)
    for k in ('z','atom_mask','x_r','x_cur'):
        perm[k]=torch.stack([d[k][i,p] for i,p in enumerate(perms)])
    for k in ('b_r','b_cur'):
        perm[k]=torch.stack([d[k][i][p][:,p] for i,p in enumerate(perms)])
    b=net(**perm)
    assert_close(b['x_vel'],torch.stack([a['x_vel'][i,p] for i,p in enumerate(perms)]))
    assert_close(b['b_vel'],torch.stack([a['b_vel'][i][p][:,p] for i,p in enumerate(perms)]))


def test_padding_invariance():
    d=inputs(); net=ReactionFlowNet('joint',**CFG); a=net(**d); padded=dict(d)
    for k in ('z','atom_mask'): padded[k]=torch.nn.functional.pad(d[k],(0,1))
    for k in ('x_r','x_cur'): padded[k]=torch.nn.functional.pad(d[k],(0,0,0,1))
    for k in ('b_r','b_cur'): padded[k]=torch.nn.functional.pad(d[k],(0,1,0,1))
    b=net(**padded)
    assert_close(a['b_vel'],b['b_vel'][:,:6,:6]); assert_close(a['x_vel'],b['x_vel'][:,:6])


def test_output_symmetry_and_com():
    d=inputs(); a=ReactionFlowNet('joint',**CFG)(**d)
    assert torch.equal(a['b_vel'],a['b_vel'].transpose(1,2))
    assert_close(a['x_vel'].sum(1),torch.zeros(2,3))


def test_role_guards():
    d=inputs()
    with pytest.raises(ValueError): ReactionFlowNet('event',**CFG)(**d)
    with pytest.raises(ValueError): ReactionFlowNet('geometry',**CFG)(**d)


def test_collision_gradients():
    d=inputs(); d['x_cur'][0,1]=d['x_cur'][0,0]; d['x_r'][0,1]=d['x_r'][0,0]
    d['x_cur'].requires_grad_();d['x_r'].requires_grad_()
    net=ReactionFlowNet('joint',**CFG); out=net(**d)
    sum(v.square().sum() for v in out.values()).backward()
    for p in net.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert torch.isfinite(d['x_cur'].grad).all(); assert torch.isfinite(d['x_r'].grad).all()


def test_parameter_matching():
    cfg=dict(scalar_dim=128,vector_dim=32,edge_dim=64,n_layers=6,n_rbf=32,n_rbf_reactant=16,r_cut=10.)
    assert match_cascade_width(cfg)['rel_diff']<=.05
