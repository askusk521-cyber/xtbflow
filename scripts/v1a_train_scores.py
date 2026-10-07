"""Train one of six fixed score members, with traceable train-only labels."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from xtbflow.m0.t1x_data import T1xCache
from xtbflow.v1.data import digest,file_hash,write_json
from xtbflow.v1.score_training import ScoreRows,query_from_training_batch,score_states
from xtbflow.v1.scores import BarrierHead,SCORE_CONFIG
from v1a_train import lr_at


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=['scores'])
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--kind',choices=['E','X'],required=True)
    ap.add_argument('--member',type=int,choices=[101,102,103],required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--stop-at',type=int)
    ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();cfg=json.loads(a.config.read_text(encoding='utf-8'))
    split=json.loads(Path(cfg['data']['split_manifest']).read_text())
    if digest({k:v for k,v in split.items() if k!='split_hash'})!=split['split_hash']:
        raise ValueError('split hash mismatch')
    tc=dict(cfg['training'],steps=cfg['score']['training_steps_per_member'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda' and not a.stop_at:
        raise RuntimeError('full training requires allocated GPU')
    torch.manual_seed(a.member);rng=np.random.default_rng(a.member)
    if device.type=='cuda':torch.cuda.manual_seed_all(a.member)
    noise=torch.Generator(device=device).manual_seed(a.member+1)
    cache=T1xCache(Path(cfg['data']['cache']))
    rows=ScoreRows(cache,split,Path(cfg['data']['catalogue_dir'])/'reference_catalog.jsonl')
    model=BarrierHead(a.kind).to(device);ema=copy.deepcopy(model).eval().requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=tc['lr'],weight_decay=tc['weight_decay'])
    labels=[dict(cache_index=i,catalog_barrier_kcal=b) for i,b in sorted(rows.barriers.items())]
    meta=dict(kind=a.kind,member=a.member,score_config=SCORE_CONFIG,median=rows.median,scale=rows.scale,
              training_barrier_quantiles=rows.quantiles,training_parent_ids=rows.parent_ids,
              training_formula_groups=split['train']['formula_groups'],labels_sha256=digest(labels),
              split_hash=split['split_hash'],source_cache_sha256=file_hash(cfg['data']['cache']),
              config_sha256=file_hash(a.config),steps=tc['steps'],ranking_loss_weight=0.,
              state_mixture=[.5,.25,.25],initialization_seed=a.member,
              source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    a.out.mkdir(parents=True,exist_ok=True)
    resume=a.out/'resume.pt';step=0;elapsed=0.
    if a.resume:
        saved=torch.load(resume,map_location=device,weights_only=False)
        for k in ['labels_sha256','split_hash','config_sha256','source_commit','kind','member']:
            if saved['meta'][k]!=meta[k]:raise ValueError('resume provenance mismatch: '+k)
        model.load_state_dict(saved['model']);ema.load_state_dict(saved['ema'])
        opt.load_state_dict(saved['optimizer']);noise.set_state(saved['noise_state'].cpu())
        rng.bit_generator.state=saved['rng_state'];step=saved['step'];elapsed=saved['elapsed_s']
    elif (a.out/'run_meta.json').exists():
        raise FileExistsError('run exists; use --resume')
    write_json(a.out/'run_meta.json',meta);write_json(a.out/'training_labels.json',labels)
    stop=min(a.stop_at or tc['steps'],tc['steps']);start=time.monotonic()
    log=(a.out/'train_log.jsonl').open('a',encoding='utf-8')
    while step<stop:
        batch,y=rows.sample(rng,tc['batch_size'],device)
        q=query_from_training_batch(batch)
        s,source=score_states(a.kind,batch,cfg['flow']['sigma_b'],cfg['flow']['sigma_x'],noise)
        for group in opt.param_groups:group['lr']=lr_at(step,tc)
        pred=model(q,s.b,s.x,s.t_b,s.t_x)
        loss=F.huber_loss(pred,y,delta=1.)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite score loss')
        opt.zero_grad(set_to_none=True);loss.backward()
        gn=torch.nn.utils.clip_grad_norm_(model.parameters(),tc['grad_clip'],error_if_nonfinite=True)
        opt.step()
        with torch.no_grad():
            for pe,p in zip(ema.parameters(),model.parameters()):pe.lerp_(p,1-tc['ema_decay'])
        step+=1
        if step%100==0 or step==stop:
            row=dict(step=step,loss=float(loss.detach()),grad_norm=float(gn),
                     batch_mae_kcal=float((pred-y).detach().abs().mean())*rows.scale,
                     elapsed_s=elapsed+time.monotonic()-start)
            log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();print(json.dumps(row),flush=True)
        if step%1000==0 or step==stop:
            payload=dict(meta=meta,model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),
                         step=step,noise_state=noise.get_state(),rng_state=rng.bit_generator.state,
                         elapsed_s=elapsed+time.monotonic()-start)
            temporary=a.out/'resume.tmp.pt';torch.save(payload,temporary);os.replace(temporary,resume)
        if step==stop:
            torch.save(dict(meta=meta,ema=ema.state_dict(),step=step),a.out/f'ckpt_{step}.pt')
    log.close()
    write_json(a.out/'status.json',dict(status='complete' if step==tc['steps'] else 'bounded_probe',
                                       steps=step,elapsed_s=elapsed+time.monotonic()-start))


if __name__=='__main__':main()
