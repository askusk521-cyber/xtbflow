import numpy as np

from xtbflow.v1.gate import decide_v1a
from xtbflow.v1.metrics import cluster_summary
from xtbflow.v1.power import complete_gate_batch,nuisance,simulate_gate


def test_vectorized_gate_matches_real_gate_with_unequal_groups():
    sizes=[2,4,1,3,2];groups=np.repeat(np.arange(len(sizes)),sizes)
    x=np.random.default_rng(7).normal([.05,.02,.02],[.15,.08,.08],(200,sum(sizes),3))
    result=complete_gate_batch(x,sizes)
    names=['GO_V1b','NO_GO_RESOURCE_SMALL','NO_GO_MECHANISM_SMALL','HOLD_INCONCLUSIVE']
    for sample,decision in zip(x,result):
        summaries=[cluster_summary(sample[:,j],groups) for j in range(3)]
        expected=decide_v1a(True,True,True,*summaries)['decision']
        assert names[int(np.flatnonzero(decision)[0])]==expected


def test_simulator_uses_development_and_no_seed_discount():
    rng=np.random.default_rng(8);groups=np.repeat(np.arange(20),3)
    x=rng.normal(size=(60,3))*[.2,.05,.06]
    nu=nuisance(x,groups)
    np.testing.assert_allclose(nu['sd'],x.std(0,ddof=1))
    assert nu['seed_averaging_discount']==1
    for distribution in ('normal','empirical_skew'):
        result=simulate_gate(x,groups,[3]*20,[0,0,0],repeats=2000,distribution=distribution)
        assert abs(sum(result[k] for k in ('GO','NO_GO_RESOURCE','NO_GO_MECHANISM','HOLD'))-1)<1e-10
        assert result['GO']<.06
