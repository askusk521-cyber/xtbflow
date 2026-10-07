import json
import numpy as np
import pytest
import torch

from xtbflow.v1.interfaces import Query
from xtbflow.v1.score_training import ScoreRows,score_states
from xtbflow.v1.scores import BarrierHead,BarrierEnsemble


def query(n=8):
    mask=torch.ones(n,3,dtype=torch.bool)
    mask[-1,-1]=False
    z=torch.ones(n,3,dtype=torch.long)*mask
    x=torch.randn(n,3,3)*mask[...,None]
    b=torch.zeros(n,3,3)
    return Query(tuple(str(i) for i in range(n)),z,z,mask,x,b)


def test_event_score_never_reads_generated_geometry():
    q=query();m=BarrierHead('E',scalar_dim=16,vector_dim=4,edge_dim=8,n_layers=1)
    t=torch.rand(8)
    a=m(q,q.b_r,q.x_r,t,t)
    b=m(q,q.b_r,q.x_r*1000,t,1-t)
    assert torch.equal(a,b)


def test_score_state_mixture_contains_cascade_boundary_and_endpoints():
    q=query(64);batch=dict(z=q.element_index,atom_mask=q.atom_mask,x_r=q.x_r,
                         b_r=q.b_r,b_p=q.b_r+1,x_ts=q.x_r+1)
    s,source=score_states('X',batch,1.,.5,torch.Generator().manual_seed(4))
    assert torch.bincount(source).tolist()==[32,16,8,8]
    assert torch.equal(s.b[source==1],batch['b_p'][source==1])
    assert torch.equal(s.x[source==2],batch['x_ts'][source==2])
    assert torch.all(s.t_x[source==3]>=.8)


def test_input_gradient_same_for_shared_ensemble_and_finite_with_padding():
    q=query();models=[BarrierHead('X',scalar_dim=16,vector_dim=4,edge_dim=8,n_layers=1) for _ in range(3)]
    ens=BarrierEnsemble(models,median=20,scale=10)
    t=torch.rand(8);x=q.x_r.clone().requires_grad_(True)
    a=ens.components(q,q.b_r,x,t,t)
    g=torch.autograd.grad(a['phi'].sum(),x)[0]
    assert torch.isfinite(g).all() and torch.equal(g[-1,-1],torch.zeros(3))
    assert torch.allclose(a['phi'],(a['mean_kcal']-20)/10+.25*torch.clamp(a['sd_kcal']/10,0,2))
    x2=q.x_r.clone().requires_grad_(True)
    g2=torch.autograd.grad(ens(q,q.b_r,x2,t,t).sum(),x2)[0]
    assert torch.equal(g,g2)


def test_normalization_and_labels_only_use_training_fold(tmp_path):
    rows=[dict(parent_id='train',cache_index=0,catalog_barrier_kcal=10.),
          dict(parent_id='train',cache_index=1,catalog_barrier_kcal=20.),
          dict(parent_id='screen',cache_index=2,catalog_barrier_kcal=-999999.)]
    p=tmp_path/'refs.jsonl';p.write_text('\n'.join(json.dumps(r) for r in rows))
    split={'train':{'parent_ids':['train'],'cache_indices':[0,1]}}
    source=ScoreRows(None,split,p)
    assert source.median==15 and source.scale==5
    assert 2 not in source.barriers
    rows[0]['cache_index']=2;p.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError,match='non-training'):
        ScoreRows(None,split,p)
