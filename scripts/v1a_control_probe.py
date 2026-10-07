"""Query-only continuous B1/B2 and actual GPU alpha-zero/batching checks."""
import argparse
import json
from pathlib import Path
import time

import torch

from xtbflow.m0.sampler import decode_be
from xtbflow.v1.assets import load_generator,load_scores,sha256
from xtbflow.v1.formal import resolve_queries
from xtbflow.v1.interfaces import assert_query_fields,query_from_parents
from xtbflow.v1.sampler import initial_state,rollout,subset_query


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-root',type=Path,required=True)
    ap.add_argument('--queries',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--frozen',type=Path)
    a=ap.parse_args()
    rows,split_name,freeze=resolve_queries(a.queries,a.frozen)
    c=dict(path='sync',alpha_x=.3,guidance_start=.5,guidance_stop=.95) if freeze is None else freeze['config']
    if c['path']!='sync':raise ValueError('continuous controls implemented for the sync path only')
    alpha,g0,g1=c['alpha_x'],c['guidance_start'],c['guidance_stop']
    for r in rows:assert_query_fields(r)
    expanded=[(r,j) for r in rows for j in range(8)]
    score,score_meta=load_scores(a.run_root,'X');lower,upper=score_meta[0]['training_barrier_quantiles']
    score.support=dict(lower_kcal=lower-10,upper_kcal=upper+10,max_sd_kcal=2*score.scale)
    a.out.mkdir(parents=True,exist_ok=False);models={};counts={};max_batch_error=0.;start=time.monotonic()
    with (a.out/'candidates.jsonl').open('w') as f:
        for seed in (0,1,2):
            net,meta=load_generator(a.run_root,'joint',seed);models[str(seed)]=meta;counts[str(seed)]=0
            if meta['split_hash']!=score_meta[0]['split_hash']:raise ValueError('score/generator split mismatch')
            for lo in range(0,len(expanded),64):
                batch=expanded[lo:lo+64];q=query_from_parents([r for r,j in batch],'cuda');proposals=[j for r,j in batch]
                state=initial_state(q,proposals,seed,namespace='mechanism',sigma_b=meta['sigma_b'],sigma_x=meta['sigma_x'])
                f0=rollout(net,q,state)
                b1=rollout(net,q,state,score=score,alpha=alpha,guidance_start=g0,guidance_stop=g1)
                b2=rollout(net,q,state,score=score,alpha=alpha,guidance_start=g0,guidance_stop=g1,replay=f0.b_trace)
                c0=rollout(net,q,state,score=score,alpha=0,replay=f0.b_trace)
                a0=rollout(net,q,state,score=score,alpha=0)
                if any(r.failed.any() for r in (f0,b1,b2,c0,a0)):raise FloatingPointError('control numerical failure')
                if not torch.equal(b2.b_trace,f0.b_trace):raise ValueError('B2 full event replay mismatch')
                for r in (c0,a0):
                    if not torch.equal(r.b_trace,f0.b_trace) or not torch.equal(r.x_trace,f0.x_trace):
                        raise ValueError('alpha-zero control mismatch')
                # Actual GPU batch-size changes on both unguided and guided trajectories.
                if lo==0:
                    for i in range(3):
                        single=subset_query(q,[i]);s=initial_state(single,[proposals[i]],seed,namespace='mechanism',
                            sigma_b=meta['sigma_b'],sigma_x=meta['sigma_x'])
                        for strength,reference in ((0.,f0),(alpha,b1)):
                            one=rollout(net,single,s,score=score,alpha=strength,guidance_start=g0,guidance_stop=g1)
                            for attr in ('b','x'):
                                v=getattr(one.state,attr);w=getattr(reference.state,attr)[i:i+1]
                                error=float((v-w).abs().max());max_batch_error=max(max_batch_error,error)
                                if not torch.allclose(v,w,atol=2e-4,rtol=2e-4):
                                    raise ValueError('GPU batching exceeds declared float32 tolerance')
                for name,r in [('F0',f0),('B1_cont',b1),('B2_cont',b2)]:
                    for i,(query,j) in enumerate(batch):
                        n=len(query['atomic_numbers']);bp,_=decode_be(r.state.b[i],q.element_index[i],n)
                        f.write(json.dumps(dict(query_id=query['query_id'],proposal=j,training_seed=seed,
                            sampling_seed=0,branch=name,b_dec=None if bp is None else bp.tolist(),
                            x=r.state.x[i,:n].cpu().tolist(),is_fallback=bp is None,
                            decode_status='VALID' if bp is not None else 'DECODE_FAILED',
                            guidance={k:float(v[i]) for k,v in r.diagnostics.items()}),allow_nan=False)+'\n')
                        counts[str(seed)]+=1
                f.flush()
            print(json.dumps(dict(seed=seed,records=counts[str(seed)],batch_error=max_batch_error)),flush=True)
    manifest=dict(complete=True,split_name=split_name,
                  freeze_sha256=None if freeze is None else freeze['freeze_sha256'],
                  models=models,scores=score_meta,support=score.support,path='sync',alpha=alpha,
                  guidance_start=g0,guidance_stop=g1,n_queries=len(rows),proposals=8,counts=counts,
                  query_source_sha256=sha256(a.queries),candidates_sha256=sha256(a.out/'candidates.jsonl'),
                  alpha_zero_exact=True,all_step_B2_replay_exact=True,gpu_float32_batch_atol=2e-4,
                  gpu_float32_batch_rtol=2e-4,max_observed_batch_absolute_error=max_batch_error,
                  gpu_wall_s=time.monotonic()-start)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
