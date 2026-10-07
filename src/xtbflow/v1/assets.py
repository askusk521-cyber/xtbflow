"""Load only frozen model weights and their provenance (no reference catalogue)."""
import hashlib
import json
from pathlib import Path

import torch

from xtbflow.m0.model import ReactionFlowNet
from .scores import BarrierEnsemble,BarrierHead


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def load_generator(root,role,seed,device='cuda',step=60000):
    directory=Path(root)/'generators'/f'{role}_s{seed}'
    status=json.loads((directory/'status.json').read_text())
    if status['status']!='complete' or status['steps']!=60000:
        raise ValueError('generator training incomplete')
    path=directory/f'ckpt_{step}.pt'
    obj=torch.load(path,map_location=device,weights_only=False);meta=obj['meta']
    if meta['role']!=role or meta['seed']!=seed or obj['step']!=step:
        raise ValueError('checkpoint identity mismatch')
    model=ReactionFlowNet('joint' if role=='baseline' else role,
                          **meta['model_cfg'],dual_time=meta['dual_time']).to(device)
    model.load_state_dict(obj['ema']);model.eval().requires_grad_(False)
    return model,dict(path=str(path),sha256=sha256(path),**meta)


def load_scores(root,kind,device='cuda',support=None):
    members=[];metadata=[]
    for seed in (101,102,103):
        directory=Path(root)/'scores'/f'{kind}_{seed}'
        status=json.loads((directory/'status.json').read_text())
        if status['status']!='complete' or status['steps']!=20000:
            raise ValueError('score training incomplete')
        path=directory/'ckpt_20000.pt';obj=torch.load(path,map_location=device,weights_only=False)
        meta=obj['meta']
        if meta['kind']!=kind or meta['member']!=seed:raise ValueError('score member mismatch')
        model=BarrierHead(kind,**meta['score_config']).to(device)
        model.load_state_dict(obj['ema']);members.append(model)
        metadata.append(dict(path=str(path),sha256=sha256(path),**meta))
    for k in ['median','scale','split_hash','labels_sha256','source_cache_sha256']:
        if any(m[k]!=metadata[0][k] for m in metadata):
            raise ValueError('ensemble member provenance differs: '+k)
    return BarrierEnsemble(members,metadata[0]['median'],metadata[0]['scale'],support),metadata
