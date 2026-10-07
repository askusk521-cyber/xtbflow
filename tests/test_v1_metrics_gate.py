import numpy as np
import pytest
from xtbflow.v1.gate import decide_v1a
from xtbflow.v1.metrics import cluster_summary,parent_auc_difference,parent_mechanism,prefix_auc


def bounds(low,high):return dict(lower_one95=low,upper_one95=high)


def test_gate_intersection_union_and_resource_priority():
    a=bounds(.01,.09);good=bounds(.005,.04);negative=bounds(-.02,.04)
    assert decide_v1a(True,True,True,a,good,good)['decision']=='GO_V1b'
    assert decide_v1a(True,True,True,a,good,negative)['decision']=='HOLD_INCONCLUSIVE'
    assert decide_v1a(True,True,True,a,negative,good)['decision']=='HOLD_INCONCLUSIVE'
    assert decide_v1a(True,True,True,bounds(.001,.049),good,good)['decision']=='NO_GO_RESOURCE_SMALL'
    assert decide_v1a(True,True,True,a,good,bounds(.001,.019))['decision']=='NO_GO_MECHANISM_SMALL'
    assert decide_v1a(True,False,True,a,good,good)['decision']=='HOLD_INCOMPLETE'
    assert decide_v1a(True,True,False,a,good,good)['decision']=='HOLD_POWER'


def test_auc_aggregates_seeds_before_inference_and_rejects_nonmonotone():
    a=np.zeros((2,3,5));b=a.copy();b[:,:,2:]=1
    assert np.allclose(parent_auc_difference(a,b),.625)
    with pytest.raises(ValueError):prefix_auc([0,1,0,1,1])


def test_mechanism_no_max_bias_and_equal_amplitudes():
    shape=(10,3,8,3);s=np.full(shape,.5);n=s.copy();r=s.copy()
    s[:,:,:,2]=.8
    m0,mr=parent_mechanism(s,n,r)
    assert np.allclose(m0,.1) and np.allclose(mr,.1)
    rng=np.random.default_rng(14)
    s,n,r=[rng.random((1000,3,8,3)) for _ in range(3)]
    m0,mr=parent_mechanism(s,n,r)
    assert abs(m0.mean())<.006 and abs(mr.mean())<.006
    assert len(m0)==1000


def test_cluster_degenerate_not_fictitious_certainty_and_unequal_groups():
    r=cluster_summary([0,0,0],['a','b','c'])
    assert r['status']=='DEGENERATE_VARIANCE' and r['upper_one95']==1
    d=np.array([0.,0.,.2,.4]);g=['a','a','b','c']
    r=cluster_summary(d,g)
    assert r['n_parents']==4 and r['n_formula_groups']==3
    assert r['estimate']==pytest.approx(.15)
    assert r['ci_two95'][0]<r['lower_one95']<r['estimate']
    assert r['se_cluster']==pytest.approx(np.sqrt(1.5*((-.3)**2+.05**2+.25**2))/4)
