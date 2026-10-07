"""Bounded V1a generator training with exact replay of data/noise on resume."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

from xtbflow.m0.batching import collate
from xtbflow.m0.flow import flow_loss
from xtbflow.m0.model import ReactionFlowNet, count_parameters
from xtbflow.m0.t1x_data import T1xCache
from xtbflow.v1.data import file_hash, write_json
from xtbflow.v1.training import TrainingRows, generator_inputs


def lr_at(step,tc):
    if step<tc['warmup_steps']:
        return tc['lr']*(step+1)/tc['warmup_steps']
    p=(step-tc['warmup_steps'])/max(1,tc['steps']-tc['warmup_steps'])
    return tc['min_lr']+.5*(tc['lr']-tc['min_lr'])*(1+math.cos(math.pi*p))


def main():
    if len(sys.argv)>1 and sys.argv[1]=='scores':
        from v1a_train_scores import main as score_main
        return score_main()
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=['generators'])
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--role',choices=['joint','event','geometry','baseline'],required=True)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--stop-at',type=int)
    ap.add_argument('--resume',action='store_true')
    a=ap.parse_args()
    cfg=json.loads(a.config.read_text(encoding='utf-8'))
    tc=cfg['training'];sigma_b=cfg['flow']['sigma_b'];sigma_x=cfg['flow']['sigma_x']
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda' and not a.stop_at:
        raise RuntimeError('full training requires allocated GPU')
    if a.role=='baseline' and a.seed!=0:
        raise ValueError('B_M0re uses seed 0')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if device.type=='cuda':
        torch.cuda.manual_seed_all(a.seed)
    mc=cfg['model'] if a.role in ('joint','baseline') else cfg['cascade_model']
    model=ReactionFlowNet('joint' if a.role=='baseline' else a.role,
                           **mc,dual_time=a.role=='joint').to(device)
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=tc['lr'],weight_decay=tc['weight_decay'])
    cache_path=Path(os.path.expandvars(cfg['data']['cache']))
    split_path=Path(os.path.expandvars(cfg['data']['split_manifest']))
    ds=TrainingRows(T1xCache(cache_path),split_path)
    # Keep training in RAM, with an explicit permutation/cursor for exact resume.
    data_rng=np.random.default_rng(a.seed)
    order=data_rng.permutation(len(ds));cursor=0
    noise=torch.Generator(device=device).manual_seed(a.seed+1)
    sha=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    meta=dict(role=a.role,seed=a.seed,model_cfg=mc,dual_time=a.role=='joint',
              sigma_b=sigma_b,sigma_x=sigma_x,training=tc,params=count_parameters(model),
              source_commit=sha,config_sha256=file_hash(a.config),
              source_cache_sha256=file_hash(cache_path),split_hash=ds.split['split_hash'],
              training_indices=ds.indices.tolist(),training_parent_ids=ds.split['train']['parent_ids'],
              training_formula_groups=ds.split['train']['formula_groups'],
              device=torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu')
    a.out.mkdir(parents=True,exist_ok=True)
    last=a.out/'resume.pt';step=0;elapsed=0.
    if a.resume:
        saved=torch.load(last,map_location=device,weights_only=False)
        for k in ['source_commit','config_sha256','source_cache_sha256','split_hash','role','seed']:
            if saved['meta'][k]!=meta[k]:
                raise ValueError(f'resume provenance mismatch: {k}')
        model.load_state_dict(saved['model']);ema.load_state_dict(saved['ema'])
        opt.load_state_dict(saved['optimizer']);noise.set_state(saved['noise_state'].cpu())
        order=np.array(saved['order']);cursor=saved['cursor'];step=saved['step']
        data_rng.bit_generator.state=saved['data_rng'];elapsed=saved['elapsed_s']
    elif (a.out/'run_meta.json').exists():
        raise FileExistsError('run already exists; use --resume')
    write_json(a.out/'run_meta.json',meta)
    stop=min(a.stop_at or tc['steps'],tc['steps']);started=time.monotonic()
    log=(a.out/'train_log.jsonl').open('a',encoding='utf-8')
    while step<stop:
        if cursor+tc['batch_size']>len(ds):
            order=data_rng.permutation(len(ds));cursor=0
        selected=order[cursor:cursor+tc['batch_size']];cursor+=len(selected)
        batch={k:v.to(device) for k,v in collate([ds[int(i)] for i in selected]).items()}
        for g in opt.param_groups:
            g['lr']=lr_at(step,tc)
        inputs,targets=generator_inputs(a.role,batch,sigma_b,sigma_x,noise,
                                       cfg['generator'].get('sync_fraction',0.))
        losses=flow_loss(model(**inputs),targets,batch['atom_mask'],cfg['flow']['geometry_loss_weight'])
        if not torch.isfinite(losses['total']):
            write_json(a.out/'failure.json',dict(step=step,reason='nonfinite_loss',
                                                indices=batch['index'].tolist()))
            raise FloatingPointError('nonfinite training loss')
        opt.zero_grad(set_to_none=True);losses['total'].backward()
        gn=torch.nn.utils.clip_grad_norm_(model.parameters(),tc['grad_clip'],error_if_nonfinite=True)
        opt.step()
        with torch.no_grad():
            for pe,p in zip(ema.parameters(),model.parameters()):
                pe.lerp_(p,1-tc['ema_decay'])
        step+=1
        if step%100==0 or step==stop:
            row=dict(step=step,elapsed_s=elapsed+time.monotonic()-started,grad_norm=float(gn),
                     **{k:float(v.detach()) for k,v in losses.items()})
            log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            print(json.dumps(row),flush=True)
        if step%1000==0 or step==stop:
            payload=dict(meta=meta,model=model.state_dict(),ema=ema.state_dict(),step=step,
                         optimizer=opt.state_dict(),noise_state=noise.get_state(),
                         order=order.tolist(),cursor=cursor,data_rng=data_rng.bit_generator.state,
                         elapsed_s=elapsed+time.monotonic()-started)
            temporary=a.out/'resume.tmp.pt';torch.save(payload,temporary);os.replace(temporary,last)
        if step in (20000,30000,40000,50000,60000) or step==stop:
            torch.save(dict(meta=meta,ema=ema.state_dict(),step=step),a.out/f'ckpt_{step}.pt')
    log.close()
    write_json(a.out/'status.json',dict(status='complete' if step==tc['steps'] else 'bounded_probe',
                                       steps=step,elapsed_s=elapsed+time.monotonic()-started))


if __name__=='__main__':
    main()
