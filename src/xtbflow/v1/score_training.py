"""Parent-balanced supervision using only isolated training reference records."""
import json
from pathlib import Path

import numpy as np
import torch

from xtbflow.m0.batching import collate
from xtbflow.m0.flow import geometry_noise,interpolate,symmetric_noise
from .interfaces import Query,State


class ScoreRows:
    def __init__(self,cache,split,catalogue):
        self.cache=cache
        self.parent_ids=tuple(sorted(split['train']['parent_ids']))
        allowed=set(split['train']['cache_indices'])
        self.rows={p:[] for p in self.parent_ids}
        self.barriers={}
        for line in Path(catalogue).read_text(encoding='utf-8').splitlines():
            r=json.loads(line)
            if r['parent_id'] in self.rows:
                if r['cache_index'] not in allowed:
                    raise ValueError('training parent includes non-training reference')
                self.rows[r['parent_id']].append(r['cache_index'])
                self.barriers[r['cache_index']]=r['catalog_barrier_kcal']
        if set(self.barriers)!=allowed or not all(self.rows.values()):
            raise ValueError('incomplete training labels')
        y=np.array(list(self.barriers.values()))
        if not np.isfinite(y).all():
            raise ValueError('nonfinite training label')
        self.median=float(np.median(y));self.scale=max(float(np.subtract(*np.percentile(y,[75,25]))),5.)
        self.quantiles=np.percentile(y,[1,99]).tolist()

    def sample(self,rng,n,device):
        parents=rng.integers(0,len(self.parent_ids),size=n)
        indices=[int(rng.choice(self.rows[self.parent_ids[int(i)]])) for i in parents]
        batch={k:v.to(device) for k,v in collate([self.cache.reaction(i) for i in indices]).items()}
        labels=torch.tensor([(self.barriers[i]-self.median)/self.scale for i in indices],
                            dtype=torch.float32,device=device)
        return batch,labels


def query_from_training_batch(batch):
    import torch
    lookup=torch.tensor([0,1,6,7,8],device=batch['z'].device)
    return Query(tuple(str(i) for i in batch['index'].tolist()),lookup[batch['z']],batch['z'],
                 batch['atom_mask'],batch['x_r'],batch['b_r'])


def score_states(kind,batch,sigma_b,sigma_x,generator):
    mask=batch['atom_mask'];n=len(mask)
    if n%8:
        raise ValueError('score batches must be divisible by 8 for fixed mixture')
    device=mask.device
    tb=torch.rand(n,device=device,generator=generator)
    tx=torch.rand(n,device=device,generator=generator)
    source=torch.zeros(n,dtype=torch.long,device=device)
    source[n//2:3*n//4]=1  # exact event, interpolated geometry
    source[3*n//4:7*n//8]=2  # exact endpoints
    source[7*n//8:]=3  # late independent interpolants
    source=source[torch.randperm(n,device=device,generator=generator)]
    if kind=='X':
        tb=torch.where((source==1)|(source==2),torch.ones_like(tb),tb)
        tx=torch.where(source==2,torch.ones_like(tx),tx)
        tb=torch.where(source==3,.8+.2*tb,tb)
        tx=torch.where(source==3,.8+.2*tx,tx)
    elif kind=='E':
        tb=torch.where(source>0,torch.ones_like(tb),tb)
        tx=torch.zeros_like(tx)
    else:
        raise ValueError(kind)
    b0=batch['b_r']+sigma_b*symmetric_noise(mask,generator)
    x0=batch['x_r']+sigma_x*geometry_noise(mask,generator)
    b=interpolate(b0,batch['b_p'],tb)
    x=interpolate(x0,batch['x_ts'],tx) if kind=='X' else batch['x_r']
    return State(b,x,tb,tx),source
