"""Generate all five development windows, then seal outputs for offline audit."""
import argparse
import json
from pathlib import Path
import time

import torch

from xtbflow.m0.sampler import decode_be
from xtbflow.v1.assets import load_generator,load_scores,sha256
from xtbflow.v1.interfaces import assert_query_fields,query_from_parents
from xtbflow.v1.pulses import pulse_branches
from xtbflow.v1.sampler import initial_state,rollout


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--queries',type=Path,required=True)
    ap.add_argument('--run-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--path',choices=['sync','event_lead2'],required=True)
    ap.add_argument('--proposals',type=int,default=8)
    ap.add_argument('--batch-size',type=int,default=64)
    ap.add_argument('--seed',type=int,choices=[0,1,2],default=0)
    ap.add_argument('--times',type=float,nargs='+',default=[.20,.35,.50,.65,.80])
    a=ap.parse_args()
    if a.queries.name!='development_queries.jsonl':raise ValueError('development input required')
    if not torch.cuda.is_available():raise RuntimeError('allocated GPU required')
    rows=[json.loads(line) for line in a.queries.read_text().splitlines()]
    for q in rows:assert_query_fields(q)
    if any(t not in (.20,.35,.50,.65,.80) for t in a.times) or len(a.times)!=len(set(a.times)):
        raise ValueError('times must be distinct prespecified development windows')
    net,model_meta=load_generator(a.run_root,'joint',a.seed)
    score,score_meta=load_scores(a.run_root,'X')
    if any(m['split_hash']!=model_meta['split_hash'] for m in score_meta):
        raise ValueError('generator-score split mismatch')
    lower,upper=score_meta[0]['training_barrier_quantiles']
    # Initial support is predeclared from training labels, not formal outcomes.
    # Full development distribution is reported before freezing the final support.
    score.support=dict(lower_kcal=lower-10,upper_kcal=upper+10,max_sd_kcal=2*score.scale)
    a.out.mkdir(parents=True,exist_ok=False)
    expanded=[(q,j) for q in rows for j in range(a.proposals)]
    started=time.monotonic();count=0
    with (a.out/'candidates.jsonl').open('w') as f:
        for lo in range(0,len(expanded),a.batch_size):
            batch=expanded[lo:lo+a.batch_size];q=query_from_parents([r for r,j in batch],'cuda')
            proposals=[j for r,j in batch]
            initial=initial_state(q,proposals,a.seed,namespace='mechanism',
                                  sigma_b=model_meta['sigma_b'],sigma_x=model_meta['sigma_x'])
            shadow=rollout(net,q,initial,path=a.path)
            for requested in a.times:
                branches,meta=pulse_branches(net,score,q,initial,proposals,path=a.path,
                                             requested_t=requested,shadow=shadow,training_seed=a.seed)
                for name,branch in branches.items():
                    for i,(query,proposal) in enumerate(batch):
                        n=len(query['atomic_numbers'])
                        bp,fb=decode_be(branch.state.b[i],q.element_index[i],n)
                        amp=None if name=='F0' else name.split('_')[-1]
                        pulse=meta['pulses'].get(amp)
                        record=dict(query_id=query['query_id'],proposal=proposal,training_seed=a.seed,
                            sampling_seed=0,path=a.path,branch=name,requested_t=requested,
                            actual_t_b=meta['actual_t_b'],actual_t_x=meta['actual_t_x'],
                            step_index=meta['step_index'],requested_rms=0 if pulse is None else pulse['requested_rms'],
                            actual_rms=0 if pulse is None else float(pulse['actual_rms'][i]),
                            applicable=bool(meta['score']['applicable'][i]),
                            supported=bool(meta['score']['supported'][i]),
                            score_at_pulse=float(meta['score']['score'][i]),
                            mean_kcal_at_pulse=float(meta['score']['mean_kcal'][i]),
                            sd_kcal_at_pulse=float(meta['score']['sd_kcal'][i]),
                            gradient_norm=float(meta['score']['gradient_norm'][i]),
                            b_dec=None if bp is None else bp.tolist(),is_fallback=bp is None,
                            decode_status='VALID' if bp is not None else 'DECODE_FAILED',
                            x=branch.state.x[i,:n].cpu().tolist(),
                            generator_calls=float(branch.diagnostics['generator_forward'][i]))
                        f.write(json.dumps(record,allow_nan=False)+'\n');count+=1
                f.flush()
            print(json.dumps({'batch_end':lo+len(batch),'records':count,'elapsed_s':time.monotonic()-started}),flush=True)
    manifest=dict(complete=True,models=model_meta,scores=score_meta,support=score.support,
                  query_source_sha256=sha256(a.queries),candidates_sha256=sha256(a.out/'candidates.jsonl'),
                  n_queries=len(rows),proposals=a.proposals,path=a.path,training_seed=a.seed,
                  requested_times=a.times,amplitudes=[.05,.10,.20],
                  gpu_wall_s=time.monotonic()-started,records=count,
                  shadow_computed_once_per_start=True,shared_random_direction_across_amplitudes=True)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
