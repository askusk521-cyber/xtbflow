import numpy as np

from xtbflow.v1.data import canonical_event
from xtbflow.v1.proxy import CatalogueMatcher


def fixture():
    # Four hydrogens: two H2 molecules with a directed partner-exchange event.
    z=np.ones(4,dtype=int);br=np.zeros((4,4),dtype=int);bp=br.copy()
    br[0,1]=br[1,0]=br[2,3]=br[3,2]=1
    bp[0,2]=bp[2,0]=bp[1,3]=bp[3,1]=1
    perms=np.array([[0,1,2,3],[1,0,3,2]])
    x=np.array([[0.,0,0],[1,0,0],[0,2,0],[0,0,3.]])
    event=canonical_event(br,bp,perms)
    refs=[dict(reference_id='low',parent_id='p',b_p=bp.tolist(),x_ts=(x*3).tolist(),
               catalog_barrier_kcal=2.,channel_id=event),
          dict(reference_id='high',parent_id='p',b_p=bp.tolist(),x_ts=x.tolist(),
               catalog_barrier_kcal=8.,channel_id=event)]
    p=dict(parent_id='p',atomic_numbers=z.tolist(),b_r=br.tolist(),permutations=perms.tolist(),
           reference_ids=['low','high'],best_reference_ids=['low'],best_channel_ids=[event],
           best_barrier_kcal=2.)
    return CatalogueMatcher(p,refs),bp,x


def test_actual_matched_reference_energy_only():
    m,b,x=fixture();r=m.match(b,x)
    assert r['proxy_status']=='PROXY_MATCH'
    assert r['matched_reference_ids']==['high']
    assert r['catalog_barrier_kcal']==8.
    assert r['hits_best_event'] and not r['hits_best_reference']
    assert r['event_utility']==1.


def test_same_event_geometry_mapping_and_proper_rotation():
    m,b,x=fixture();pi=m.perms[1];rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    assert m.match(b[np.ix_(pi,pi)],x[pi]@rotation+5)['proxy_status']=='PROXY_MATCH'
    mirror=x.copy();mirror[:,0]*=-1
    assert m.match(b,mirror)['proxy_status']=='KNOWN_EVENT_GEOMETRY_MISS'
    bad=x[[2,1,0,3]]
    assert m.match(b,bad)['proxy_status']=='KNOWN_EVENT_GEOMETRY_MISS'


def test_fallback_nonfinite_empty_and_geometry_miss():
    m,b,x=fixture()
    assert m.match(b,x,is_fallback=True)['proxy_status']=='INVALID_OUTPUT'
    assert m.match(None,x)['proxy_status']=='INVALID_OUTPUT'
    assert m.match(b,x*np.nan)['proxy_status']=='INVALID_OUTPUT'
    r=m.match(b,x*10)
    assert r['proxy_status']=='KNOWN_EVENT_GEOMETRY_MISS'
    assert r['hits_best_event'] and r['catalog_barrier_kcal'] is None
