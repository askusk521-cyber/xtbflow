"""Frozen posthoc endpoint-ranker ablation on read-only smc-1 populations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from xtbflow.v1.smc import population_metrics
from xtbflow.v1.metrics import cluster_summary


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],text=True).strip()
def load(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=Path('configs/explore/smc_raw_ranking.json'));parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU-only command')
    if git('status','--porcelain') or not git('branch','-r','--contains','HEAD'):raise RuntimeError('clean pushed source required')
    cfg=load(a.config);root=Path(cfg['input_root']);published=load(cfg['input_results'])
    input_file=root/'endpoint_rows.json'
    if sha(input_file)!=published['raw_output_sha256']['endpoint_rows.json']:raise RuntimeError('input hash mismatch')
    parent_file=Path('/home/lhshen/xtbflow-runs/v1a-20261007/data/parent_catalog.json')
    if sha(parent_file)!=published['reproduction']['c1']['inputs'][str(parent_file)]:raise RuntimeError('parent hash mismatch')
    rows=load(input_file);parents={p['parent_id']:p for p in load(parent_file)};pop={}
    for r in rows:pop.setdefault((r['arm'],r['parent_id'],r['seed']),[]).append(r)
    assert len(rows)==77376 and len(pop)==2418
    arms=sorted({k[0] for k in pop});pids=sorted({k[1] for k in pop})
    assert len(arms)==13 and len(pids)==62
    for v in pop.values():assert len(v)==32 and {r['proposal'] for r in v}==set(range(32))
    values={ranker:{} for ranker in cfg['rankers']}
    for key,rr in pop.items():
        arm,pid,seed=key
        original=population_metrics(rr,parents[pid]['best_channel_ids'],False)
        replaced=[]
        for r in rr:
            raw=r['score']['raw_barrier']
            success=raw is not None and np.isfinite(raw)
            replaced.append(dict(r,score=dict(status='ok' if success else 'failed',barrier=raw if success else None,curvature=None)))
        alternative=population_metrics(replaced,parents[pid]['best_channel_ids'],False)
        for m in ('hit_infinity','event_utility','distinct_legal_events','legal_rate','best_reference_rate'):assert original[m]==alternative[m]
        values['original_relaxed'][key]=original;values['raw'][key]=alternative
    pm={ranker:{arm:{m:{p:float(np.mean([table[arm,p,s][m] for s in (0,1,2)])) for p in pids} for m in next(iter(table.values()))} for arm in arms} for ranker,table in values.items()}
    maxerr=max(abs(pm['original_relaxed'][arm][m][p]-published['parent_metrics'][arm][m][p]) for arm in arms for m in pm['original_relaxed'][arm] for p in pids)
    assert maxerr<=1e-12
    def summary(v):return cluster_summary([v[p] for p in pids],[parents[p]['split_group'] for p in pids])
    absolute={ranker:{arm:{m:summary(v) for m,v in metrics.items()} for arm,metrics in table.items()} for ranker,table in pm.items()}
    pairs=[]
    for s in (.6,.7,.8,.9):
        r,w,z=f'relaxed_{s:.1f}',f'raw_{s:.1f}',f'random_{s:.1f}'
        pairs.extend([(r,'none'),(w,'none'),(r,z),(w,z),(r,w),(z,'none')])
    comparisons={ranker:{f'{x}-{y}':{m:summary({p:table[x][m][p]-table[y][m][p] for p in pids}) for m in table[x]} for x,y in pairs} for ranker,table in pm.items()}
    shifts={arm:{p:pm['raw'][arm]['H'][p]-pm['original_relaxed'][arm]['H'][p] for p in pids} for arm in arms}
    result=dict(version=cfg['version'],exploratory=True,primary_arm=cfg['primary_arm'],arms=absolute,comparisons=comparisons,
                within_arm_ranker_shift={arm:summary(v) for arm,v in shifts.items()},
                difference_in_differences={f'{x}-{y}':summary({p:shifts[x][p]-shifts[y][p] for p in pids}) for x,y in pairs},
                original_reproduction_max_error=maxerr,parent_metrics=pm,
                provenance=dict(source_commit=git('rev-parse','HEAD'),dirty=False,slurm_job_id=None,workers=1,
                                config_sha256=sha(a.config),inputs={str(p):sha(p) for p in (input_file,parent_file,Path(cfg['input_results']))}),
                caveat=cfg['interpretation'])
    a.out.mkdir(parents=True,exist_ok=False);dump(a.out/'results.json',result)
    print('complete: all populations and original metrics verified')


if __name__=='__main__':main()
