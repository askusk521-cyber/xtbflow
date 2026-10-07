"""Generate development baselines in a process with only reactant-query input.

Offline matching is a separate command/process, after this source is sealed.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from xtbflow.m0.sampler import decode_be
from xtbflow.v1.assets import load_generator,sha256
from xtbflow.v1.interfaces import assert_query_fields,query_from_parents
from xtbflow.v1.sampler import initial_state,rollout


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=['baseline'])
    ap.add_argument('--queries',type=Path,required=True)
    ap.add_argument('--run-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--batch-size',type=int,default=64)
    ap.add_argument('--proposals',type=int,default=8)
    a=ap.parse_args()
    if a.queries.name!='development_queries.jsonl':
        raise ValueError('this command is restricted to the development query view')
    rows=[json.loads(line) for line in a.queries.read_text().splitlines()]
    for q in rows:assert_query_fields(q)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device!='cuda':raise RuntimeError('development generation requires allocated GPU')
    a.out.mkdir(parents=True,exist_ok=False)
    expanded=[(q,j) for q in rows for j in range(a.proposals)]
    metadata={};times={}
    with (a.out/'candidates.jsonl').open('w') as f:
        for name,role,path in [('B_M0re','baseline','sync'),('B0_sync','joint','sync'),
                               ('B0_event_lead2','joint','event_lead2')]:
            model,meta=load_generator(a.run_root,role,0,device)
            metadata[name]=meta;started=time.monotonic()
            for lo in range(0,len(expanded),a.batch_size):
                batch=expanded[lo:lo+a.batch_size]
                q=query_from_parents([r for r,j in batch],device)
                s=initial_state(q,[j for r,j in batch],0,namespace='development_baseline',
                                sigma_b=meta['sigma_b'],sigma_x=meta['sigma_x'])
                result=rollout(model,q,s,path=path,dual_time=role!='baseline')
                for k,(row,proposal) in enumerate(batch):
                    n=len(row['atomic_numbers'])
                    bp,fallback=decode_be(result.state.b[k],q.element_index[k],n)
                    record=dict(query_id=row['query_id'],proposal=proposal,arm=name,
                                training_seed=0,sampling_seed=0,path=path,
                                b_dec=None if bp is None else bp.tolist(),fallback_b=fallback.tolist(),
                                x=result.state.x[k,:n].cpu().tolist(),is_fallback=bp is None,
                                decode_status='NUMERICAL_FAILURE' if result.failed[k] else
                                              ('VALID' if bp is not None else 'DECODE_FAILED'),
                                generator_calls=float(result.diagnostics['generator_forward'][k]))
                    f.write(json.dumps(record,allow_nan=False)+'\n')
                f.flush()
            times[name]=time.monotonic()-started
            del model
            torch.cuda.empty_cache()
    manifest=dict(complete=True,query_source_sha256=sha256(a.queries),
                  candidates_sha256=sha256(a.out/'candidates.jsonl'),models=metadata,
                  gpu_wall_s=times,n_queries=len(rows),proposals=a.proposals,
                  namespace='development_baseline',batch_size=a.batch_size)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(complete=True,n_queries=len(rows),gpu_wall_s=times)))


if __name__=='__main__':main()
