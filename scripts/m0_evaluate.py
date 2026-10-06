"""Evaluate saved candidates; labels are used only here, never by sampling."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from xtbflow.m0.metrics import automorphisms, evaluate_query
from xtbflow.m0.sampler import decode_be
from xtbflow.m0.t1x_data import SPLITS, T1xCache
from xtbflow.m0.batching import ELEMENT_INDEX


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--candidates',type=Path,required=True)
    ap.add_argument('--split',choices=('val','test'),required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists(): raise FileExistsError(a.out)
    cache=T1xCache(a.cache)
    with np.load(a.candidates,allow_pickle=False) as f:
        c={k:f[k] for k in f.files}
    qids=c['query_index'];sizes=c['n_atoms'];offs=c['atom_off']
    if len(qids)!=len(sizes) or len(offs)!=len(qids)+1:
        raise ValueError('Malformed candidate offsets')
    boffs=np.r_[0,np.cumsum(sizes**2)]
    if offs[-1]!=len(c['x']) or boffs[-1]!=len(c['b_raw']):
        raise ValueError('Malformed CSR lengths')
    actual=set(map(int,np.unique(qids)))
    expected=set(map(int,np.flatnonzero(cache.split==SPLITS[a.split])))
    if actual!=expected: raise ValueError('Candidates must cover entire selected split')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as out:
        for q in sorted(actual):
            r=cache.reaction(q);z=r['z'];n=len(z)
            idx=np.flatnonzero(qids==q)
            if sorted(c['sample_id'][idx].tolist())!=list(range(len(idx))):
                raise ValueError('Missing or duplicate sample IDs')
            zidx=torch.tensor([ELEMENT_INDEX[int(v)] for v in z])
            mats=[];xs=[]
            for k in idx:
                if sizes[k]!=n:raise ValueError('Candidate atom count mismatch')
                raw=torch.from_numpy(c['b_raw'][boffs[k]:boffs[k+1]].reshape(n,n))
                mat,_=decode_be(raw,zidx,n);mats.append(mat)
                xs.append(c['x'][offs[k]:offs[k+1]])
            perms,cap=automorphisms(z,r['b_r'])
            result=evaluate_query(z,r['b_r'],r['x_r'],r['b_p'],r['x_ts'],mats,np.stack(xs),perms)
            result.update(index=q,parent_id=r['parent_id'],automorphism_cap_hit=cap)
            out.write(json.dumps(result,allow_nan=False)+'\n')
    print(json.dumps(dict(queries=len(actual),out=str(a.out))))


if __name__=='__main__':main()
