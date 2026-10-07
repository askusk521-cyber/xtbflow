"""Development five-arm streams using query-only input and logical network cost."""
import argparse
import json
from pathlib import Path
import time

import torch
from rdkit import RDLogger

from xtbflow.v1.assets import load_generator,load_scores,sha256
from xtbflow.v1.costs import weights_for_atoms
from xtbflow.v1.interfaces import assert_query_fields,query_from_parents
from xtbflow.v1.sampler import subset_query
from xtbflow.v1.streams import logical_stream,run_programs


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--queries',type=Path,required=True)
    ap.add_argument('--run-root',type=Path,required=True)
    ap.add_argument('--costs',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--path',choices=['sync','event_lead2'],default='event_lead2')
    ap.add_argument('--alpha-x',type=float,choices=[0.,.15,.3],default=.15)
    ap.add_argument('--alpha-b',type=float,choices=[.1,.25],default=.1)
    ap.add_argument('--start',type=float,default=.5)
    ap.add_argument('--stop',type=float,choices=[.95,.85,.75],default=.95)
    ap.add_argument('--a2-times',type=float,nargs=2,default=[.4,.7])
    ap.add_argument('--seed',type=int,choices=[0,1,2],default=0)
    ap.add_argument('--cap',type=float,default=3200.)
    ap.add_argument('--batch-size',type=int,default=64)
    ap.add_argument('--arms',nargs='+',choices=['A0','A1','A2','B0','B1'],default=['A0','A1','A2','B0','B1'])
    a=ap.parse_args()
    if a.queries.name!='development_queries.jsonl':raise ValueError('development queries required; formal freeze not implemented')
    if not torch.cuda.is_available():raise RuntimeError('allocated GPU required')
    if tuple(a.a2_times) not in ((.4,.7),(.6,.85)):raise ValueError('unplanned A2 checkpoints')
    if not 0<=a.start<a.stop or not 0<a.cap<=3200:raise ValueError('invalid range')
    RDLogger.DisableLog('rdApp.error')
    rows=[json.loads(line) for line in a.queries.read_text().splitlines()]
    for row in rows:assert_query_fields(row)
    q=query_from_parents(rows,'cuda');weights=json.loads(a.costs.read_text())
    networks={};model_meta={};scores={};score_meta={}
    for role in ('joint','event','geometry'):
        networks[role],model_meta[role]=load_generator(a.run_root,role,a.seed)
    for kind in ('E','X'):
        scores[kind],score_meta[kind]=load_scores(a.run_root,kind)
        lower,upper=score_meta[kind][0]['training_barrier_quantiles']
        scores[kind].support=dict(lower_kcal=lower-10,upper_kcal=upper+10,max_sd_kcal=2*scores[kind].scale)
    split=model_meta['joint']['split_hash']
    if any(m['split_hash']!=split for m in list(model_meta.values())+sum(score_meta.values(),[])):
        raise ValueError('model provenance mismatch')
    cfg=dict(path=a.path,alpha_x=a.alpha_x,alpha_b=a.alpha_b,guidance_start=a.start,guidance_stop=a.stop,
             event_start=.5,event_stop=.95,a2_times=a.a2_times,cap_units=a.cap,
             sigma_b=model_meta['joint']['sigma_b'],sigma_x=model_meta['joint']['sigma_x'],
             namespace='development_efficiency')
    a.out.mkdir(parents=True,exist_ok=False);times={};total=0
    with (a.out/'streams.jsonl').open('w') as f:
        for arm in a.arms:
            started=time.monotonic();outputs=[{} for _ in rows]
            programs=[logical_stream(subset_query(q,[i]),arm,a.seed,weights_for_atoms(weights,len(row['atomic_numbers'])),
                                     cfg,outputs[i]) for i,row in enumerate(rows)]
            run_programs(programs,networks,scores,a.batch_size,diagnostics=True,drift=True)
            for out in outputs:f.write(json.dumps(out,allow_nan=False)+'\n')
            f.flush();times[arm]=time.monotonic()-started;total+=len(outputs)
            print(json.dumps(dict(arm=arm,seconds=times[arm],n_queries=len(outputs),
                                 completed=sum(len(o['candidates']) for o in outputs))),flush=True)
    manifest=dict(complete=True,config=cfg,models=model_meta,scores=score_meta,training_seed=a.seed,
                  support={k:v.support for k,v in scores.items()},query_source_sha256=sha256(a.queries),
                  streams_sha256=sha256(a.out/'streams.jsonl'),cost_source_sha256=sha256(a.costs),
                  costs=weights,n_queries=len(rows),arms=a.arms,gpu_wall_s=times,records=total,
                  diagnostics_and_drift_shadow_in_gpu_wall_but_not_online_budget=True)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
