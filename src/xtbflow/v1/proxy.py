"""Offline joint event/structure matching. Never imported by generation code."""
import numpy as np

from mechai.data.events import LewisState
from xtbflow.m0.metrics import batched_kabsch_rmsd, be_to_mol
from xtbflow.m0.t1x_data import SYMBOL
from .data import canonical_event


def valid_endpoint(z, br, bp, charge=0, multiplicity=1):
    if bp is None:
        return False
    bp=np.asarray(bp)
    if bp.shape!=np.asarray(br).shape or not np.isfinite(bp).all():
        return False
    if not np.array_equal(bp,np.round(bp)) or np.array_equal(br,bp):
        return False
    try:
        LewisState(tuple(SYMBOL[int(v)] for v in z),
                   tuple(tuple(int(v) for v in row) for row in bp),charge,multiplicity).validate()
        be_to_mol(np.asarray(z),bp)
    except (ValueError,RuntimeError,KeyError):
        return False
    return True


class CatalogueMatcher:
    def __init__(self, parent, references, cutoff=0.50):
        self.parent=parent
        self.refs=references
        self.cutoff=cutoff
        self.perms=np.asarray(parent['permutations'],dtype=np.int64)
        self.br=np.asarray(parent['b_r'])
        self.z=np.asarray(parent['atomic_numbers'])
        if len(self.perms)==0:
            raise ValueError('unresolved empty mapping set')
        if any(r['parent_id']!=parent['parent_id'] for r in references):
            raise ValueError('reference parent mismatch')
        if {r['reference_id'] for r in references}!=set(parent['reference_ids']):
            raise ValueError('incomplete reference catalogue')

    def match(self,bp,x,is_fallback=False,decode_status='VALID'):
        result=dict(proxy_status='INVALID_OUTPUT',predicted_channel_id=None,
                    matched_reference_ids=[],best_match_rmsd_angstrom=None,
                    best_event_rmsd_angstrom=None,catalog_barrier_kcal=None,
                    actual_dft_barrier_kcal=None,hits_best_reference=False,
                    hits_near_best_reference=False,hits_best_event=False,
                    event_utility=0.0)
        x=np.asarray(x)
        if (is_fallback or decode_status!='VALID' or x.shape!=(len(self.z),3)
                or not np.isfinite(x).all() or not valid_endpoint(self.z,self.br,bp)):
            return result
        bp=np.asarray(bp)
        event=canonical_event(self.br,bp,self.perms)
        result['predicted_channel_id']=event
        event_refs=[r for r in self.refs if r['channel_id']==event]
        if not event_refs:
            result['proxy_status']='OUTSIDE_CATALOGUE_EVENT'
            return result
        result['hits_best_event']=event in self.parent['best_channel_ids']
        result['event_utility']=max(0.,1.-(min(r['catalog_barrier_kcal'] for r in event_refs)
                                          -self.parent['best_barrier_kcal'])/10.)
        hits=[]
        distances=[]
        for ref in event_refs:
            rbp=np.asarray(ref['b_p'])
            # The SAME query -> reference mapping indexes both BE and geometry.
            matching=np.all(rbp[self.perms[:,:,None],self.perms[:,None,:]]==bp,axis=(1,2))
            maps=self.perms[matching]
            if len(maps)==0:
                raise RuntimeError('channel quotient disagrees with exact atom mapping')
            d=float(batched_kabsch_rmsd(np.asarray(ref['x_ts'])[maps],x).min())
            distances.append(d)
            if d<=self.cutoff:
                hits.append((ref,d))
        result['best_event_rmsd_angstrom']=min(distances)
        result['proxy_status']='PROXY_MATCH' if hits else 'KNOWN_EVENT_GEOMETRY_MISS'
        if hits:
            result['matched_reference_ids']=[r['reference_id'] for r,d in hits]
            result['best_match_rmsd_angstrom']=min(d for r,d in hits)
            result['catalog_barrier_kcal']=min(r['catalog_barrier_kcal'] for r,d in hits)
            result['hits_best_reference']=bool(set(result['matched_reference_ids']) &
                                               set(self.parent['best_reference_ids']))
            result['hits_near_best_reference']=any(r['catalog_barrier_kcal']<=
                self.parent['best_barrier_kcal']+2.0 for r,d in hits)
        return result
