import numpy as np
import pytest

from xtbflow.v1.costs import BUDGETS,prefixes
from xtbflow.v1.metrics import prefix_auc
from xtbflow.v1.reanalysis import (budget_prefix,count_prefix,included_points,kappa_star,mode_key,
                                   opens_verification_unit,paired_difference,renormalized_auc,
                                   same_event_rmsd,unit_prefix,verification_flags)

D1={'mode':'D1'}
D2={c:{'mode':'D2','rmsd_cut_angstrom':c} for c in (.10,.30,.50)}


def cand(i,channel,status='KNOWN_EVENT_GEOMETRY_MISS'):
    return dict(i=i,predicted_channel_id=channel,proxy_status='INVALID_OUTPUT' if channel is None else status)


# Pairwise RMSD (Å) between same-channel candidates of the synthetic stream below.
DIST={(1,2):0.,(1,3):.40,(1,5):.60,(3,5):.20,(2,5):.60}
STREAM=[cand(0,None),cand(1,'a'),cand(2,'a'),cand(3,'a'),cand(4,'b'),cand(5,'a'),cand(6,None)]


def rmsd(u,c):
    return DIST[min(u['i'],c['i']),max(u['i'],c['i'])]


def test_mode_keys():
    assert [mode_key(s) for s in (D1,D2[.1],D2[.3],D2[.5])]==['D1','D2_0.10','D2_0.30','D2_0.50']


def test_d1_counts_first_appearance_of_each_channel_only():
    assert verification_flags(STREAM,D1)==[False,True,False,False,True,False,False]


def test_d2_opens_for_new_geometry_of_a_known_event():
    # Candidate 2 duplicates 1; 3 is 0.40 Å from 1; 5 is 0.20 Å from 3 but 0.60 Å from 1.
    assert verification_flags(STREAM,D2[.3],rmsd)==[False,True,False,True,True,False,False]
    assert verification_flags(STREAM,D2[.5],rmsd)==[False,True,False,False,True,True,False]
    assert verification_flags(STREAM,D2[.1],rmsd)==[False,True,False,True,True,True,False]


def test_invalid_and_fallback_never_open_a_unit():
    fallback=dict(i=9,predicted_channel_id='a',proxy_status='INVALID_OUTPUT')
    for spec in (D1,D2[.3]):
        assert not opens_verification_unit(cand(0,None),[],spec['mode'],rmsd=rmsd)
        assert not opens_verification_unit(fallback,[],spec['mode'],rmsd=rmsd)
    with pytest.raises(ValueError):opens_verification_unit(cand(1,'a'),[cand(2,'a')],'D2')
    with pytest.raises(ValueError):opens_verification_unit(cand(1,'a'),[],'D3')


def test_same_event_rmsd_uses_automorphisms_and_proper_rotations_only():
    x=np.array([[0.,0,0],[1.1,0,0],[0,1.3,0],[0,0,1.6]])
    b=np.array([[0,1,1,1],[1,0,0,0],[1,0,0,0],[1,0,0,0]])
    identity=np.arange(4)[None]
    theta=.7;rot=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    assert same_event_rmsd(identity,b,x,b,x@rot.T+3.)==pytest.approx(0.,abs=1e-9)
    mirror=x*np.array([-1.,1,1])
    assert same_event_rmsd(identity,b,x,b,mirror)>.1
    # Swapping two equivalent atoms is allowed only through a reactant automorphism,
    # and the same mapping must carry both the bond matrix and the coordinates.
    swap=np.array([0,2,1,3]);perms=np.stack([np.arange(4),swap])
    assert same_event_rmsd(perms,b,x,b,x[swap])==pytest.approx(0.,abs=1e-9)
    assert same_event_rmsd(identity,b,x,b,x[swap])>.1
    other=b.copy();other[1,2]=other[2,1]=1
    with pytest.raises(RuntimeError):same_event_rmsd(perms,b,x,other,x)


def test_count_and_unit_prefixes_censor_short_streams():
    assert count_prefix(5,4)==4 and count_prefix(3,4) is None
    opens=[False,True,False,True,True,False]
    assert [unit_prefix(opens,n) for n in (1,2,3,4)]==[2,4,5,None]


def stream(units,step):
    cu=[step*(j+1) for j in range(int(3200//step))]
    return [dict(completion_units=c) for c in cu],cu


@pytest.mark.parametrize('step',[50.,92.67399890405615,151.3,37.2])
def test_kappa_zero_prefix_matches_v1a_prefix_candidate_by_candidate(step):
    cands,cu=stream(None,step);opens=[j%3==0 for j in range(len(cu))]
    v1a=prefixes(cands)
    for k in BUDGETS:
        m=budget_prefix(cu,opens,0.,k*50.,3200.)
        assert m is not None and cands[:m]==v1a[str(k)]


def test_budget_censoring_horizon():
    cu=[50.*(j+1) for j in range(64)]
    assert budget_prefix(cu,[True]*64,0.,3200.,3200.)==64
    assert budget_prefix(cu,[True]*64,10.,64*60.,3200.)==64
    assert budget_prefix(cu,[True]*63+[False],10.,64*60.,3200.) is None
    assert budget_prefix(cu,[False]*64,10.,4*60.,3200.)==4
    # Total cost includes kappa for every unit opened up to and including j.
    assert budget_prefix(cu,[True]*64,100.,2*150.,3200.)==2
    assert budget_prefix(cu,[True,False]+[True]*62,100.,2*150.,3200.)==2
    assert budget_prefix(cu,[True,True,False]+[True]*61,100.,3*150.,3200.)==3


def test_weight_renormalization_and_inclusion_rule():
    hits=[0,1,1,1,1]
    assert renormalized_auc(hits,[True]*5)==pytest.approx(float(prefix_auc(hits)))
    assert renormalized_auc(hits,[True,True,True,False,False])==pytest.approx(.5/.625)
    with pytest.raises(ValueError):renormalized_auc(hits,[False]*5)
    frac={'A0':[0,0,.05,.06,1],'B0':[0,.051,0,0,0]}
    assert included_points(frac,['A0'])==[True,True,True,False,False]
    assert included_points(frac,['A0','B0'])==[True,False,True,False,False]


def test_paired_difference_uses_only_seeds_available_in_both_arms():
    order=['p1','p2','p3'];groups=['g1','g2','g2'];seeds=[0,1]
    x={('p1',0):1.,('p1',1):0.,('p2',0):1.,('p2',1):None,('p3',0):None,('p3',1):1.}
    y={('p1',0):0.,('p1',1):0.,('p2',0):0.,('p2',1):1.,('p3',0):1.,('p3',1):None}
    s,coverage,dropped=paired_difference(x,y,order,groups,seeds)
    assert dropped==['p3'] and coverage==pytest.approx(2/3)
    assert s['estimate']==pytest.approx((.5+1.)/2)


def test_kappa_star_first_sign_change_and_skips():
    est=lambda v:dict(estimate=v)
    star=kappa_star([(0,est(.2)),(1,est(.1)),(3,None),(10,est(-.05)),(30,est(-.1))])
    assert star['kappa_star']==10 and star['kappa_before']==1 and star['skipped_kappas']==[3]
    assert kappa_star([(0,est(-.2)),(1,est(-.1))])['kappa_star'] is None
