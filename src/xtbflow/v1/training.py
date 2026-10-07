"""Reuse M0 training primitives with explicitly isolated V1a rows and clocks."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from xtbflow.m0.flow import geometry_noise, interpolate, symmetric_noise, training_inputs
from .data import digest


class TrainingRows(Dataset):
    def __init__(self,cache,split_path):
        self.cache=cache
        self.split=json.loads(Path(split_path).read_text(encoding='utf-8'))
        saved=self.split['split_hash']
        if digest({k:v for k,v in self.split.items() if k!='split_hash'})!=saved:
            raise ValueError('split manifest hash mismatch')
        self.indices=np.array(self.split['train']['cache_indices'],dtype=int)
        if len(self.indices)==0 or np.any(cache.split[self.indices]!=0):
            raise ValueError('invalid train rows')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,index):
        return self.cache.reaction(int(self.indices[index]))


def generator_inputs(role,batch,sigma_b,sigma_x,generator,sync_fraction=0.):
    if role!='joint':
        return training_inputs('joint' if role=='baseline' else role,
                               batch,sigma_b,sigma_x,generator)
    mask=batch['atom_mask'];n=mask.shape[0]
    tb=torch.rand(n,device=mask.device,generator=generator)
    tx=torch.rand(n,device=mask.device,generator=generator)
    if sync_fraction:
        sync=torch.rand(n,device=mask.device,generator=generator)<sync_fraction
        tx=torch.where(sync,tb,tx)
    b0=batch['b_r']+sigma_b*symmetric_noise(mask,generator)
    x0=batch['x_r']+sigma_x*geometry_noise(mask,generator)
    inputs=dict(z=batch['z'],atom_mask=mask,x_r=batch['x_r'],b_r=batch['b_r'],t=tb,t_x=tx,
                b_cur=interpolate(b0,batch['b_p'],tb),x_cur=interpolate(x0,batch['x_ts'],tx))
    targets=dict(b_vel=batch['b_p']-b0,x_vel=batch['x_ts']-x0)
    return inputs,targets
