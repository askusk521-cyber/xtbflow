import numpy as np
import pytest
from xtbflow.m0.metrics import (automorphisms, batched_kabsch_rmsd, decide,
    evaluate_query, is_valid_product, paired_bootstrap)
from xtbflow.m0.t1x_data import kabsch_align


def test_rmsd():
    rng=np.random.default_rng(0);x=rng.normal(size=(10,3));q,_=np.linalg.qr(rng.normal(size=(3,3)))
    if np.linalg.det(q)<0:q[:,0]*=-1
    y=x@q+3
    assert batched_kabsch_rmsd(y[None],x)[0]<1e-6
    noisy=y+rng.normal(size=y.shape)*.1
    expected=np.sqrt(((kabsch_align(noisy,x)-x)**2).sum(1).mean())
    assert batched_kabsch_rmsd(noisy[None],x)[0]==pytest.approx(expected)


def test_water_automorphism():
    pytest.importorskip('rdkit')
    z=np.array([8,1,1]);r=np.array([[4,1,1],[1,0,0],[1,0,0]])
    p=np.array([[6,0,1],[0,0,0],[1,0,0]])
    swap=np.array([0,2,1]);c=p[swap][:,swap]
    x=np.array([[0.,0.,0.],[1.5,0.,0.],[-.3,.9,0.]])
    perms,cap=automorphisms(z,r)
    assert len(perms)==2 and not cap
    result=evaluate_query(z,r,x,p,x,[c],x[swap][None],perms)
    assert result['hit_event'] and result['hit']['0.5']
    assert result['best_rmsd']<1e-6


def test_validity():
    pytest.importorskip('rdkit')
    from rdkit import Chem
    from xtbflow.m0.t1x_data import be_matrix
    mol=Chem.AddHs(Chem.MolFromSmiles('C=C'));z=np.array([a.GetAtomicNum() for a in mol.GetAtoms()]);p=be_matrix(mol)
    r=p.copy();r[0,1]=r[1,0]=0
    assert not is_valid_product(z,r,None)
    assert not is_valid_product(z,p,p)
    assert is_valid_product(z,r,p)


def test_bootstrap():
    a={'a':.2,'b':.3,'c':.6}
    same=paired_bootstrap(a,a,10000,0)
    assert same['delta']==0 and same['ci95']==[0.,0.]
    higher=paired_bootstrap(a,{k:v+.1 for k,v in a.items()},10000,0)
    assert higher['delta']==pytest.approx(.1)
    assert higher['ci95'][1]-higher['ci95'][0]<1e-9


def test_decision():
    valid={'ci95':[-.01,.02]}
    assert decide({'ci95':[.01,.03],'delta':.02},[.01,.02,-.01],valid)=='GO'
    assert decide({'ci95':[-.02,.01],'delta':-.01},[0,0,0],valid)=='NO-GO'
    assert decide({'ci95':[-.01,.05],'delta':.02},[.02,.02,.02],valid)=='INCONCLUSIVE'
