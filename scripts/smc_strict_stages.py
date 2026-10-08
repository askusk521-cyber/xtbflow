"""Persisted-state continuation and size-independent analysis plumbing."""
import json
import os
import time
import numpy as np
import torch
import explore_smc as original
from xtbflow.v1.smc import population_metrics
from xtbflow.v1.metrics import cluster_summary


def g2(a):
    from xtbflow.v1.assets import load_generator
    from xtbflow.v1.interfaces import query_from_parents,State
    from xtbflow.v1.sampler import subset_query,rollout
    from xtbflow.v1.clocks import clock_grid
    started=time.monotonic();ctx=original.context(a)
    cfg,base,root,cat,split,parents,refs,queries=ctx
    if not a.smoke and not os.getenv('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    load=lambda n:json.loads((a.out/n).read_text())
    assert load('c1_manifest.json')['gate_passed']
    initial=load('initial_states_manifest.json');assert initial['persisted_before_g2']
    assert initial['plans_sha256']==original.file_hash(a.out/'plans.json')
    device='cpu' if a.smoke else 'cuda';qa=query_from_parents(queries,device)
    tb,tx=clock_grid('geometry_lead3',50);models={};files=[]
    for path,h in sorted(initial['files'].items()):
        assert original.file_hash(a.out/path)==h
        saved=torch.load(a.out/path,map_location='cpu',weights_only=False)
        data=torch.load(a.out/saved['batch'],map_location='cpu',weights_only=False);seed=data['seed'];k=saved['step']
        if seed not in models:models[seed]=load_generator(root,'joint',seed,device)
        net,meta=models[seed];q=subset_query(qa,data['query_indices'])
        b=saved['b'].to(device);x=saved['x'].to(device)
        st=State(b,x,torch.full((len(b),),float(tb[k]),device=device),torch.full((len(b),),float(tx[k]),device=device))
        r=rollout(net,q,st,role='joint',path='geometry_lead3',n_steps=50,start=k,dual_time=meta['dual_time'])
        if r.failed.any():raise RuntimeError('nonfinite continuation')
        name=f"g2-{saved['arm']}-{saved['batch'][3:]}"
        torch.save(dict(b=r.state.b.cpu(),x=r.state.x.cpu(),batch=saved['batch'],arm=saved['arm']),a.out/name);files.append(name)
    original.manifest(a,ctx,'g2',started,dict(files=files,consumed_initial_manifest_sha256=original.file_hash(a.out/'initial_states_manifest.json')))


def analyze_plumbing(a):
    """Same population/paired aggregation at smoke and full size; no smoke metrics saved."""
    ctx=original.context(a);parents=ctx[5]
    load=lambda n:json.loads((a.out/n).read_text())
    for stage in ('s0','c1','c2'):assert load(stage+'_manifest.json')['gate_passed']
    rr=load('endpoint_rows.json');pop={}
    for r in rr:pop.setdefault((r['arm'],r['parent_id'],r['seed']),[]).append(r)
    arms=sorted({k[0] for k in pop});pids=sorted(parents);seeds=a.seeds
    assert len(arms)==13 and set(pop)=={(arm,p,s) for arm in arms for p in pids for s in seeds}
    assert all(len(v)==32 and {r['proposal'] for r in v}==set(range(32)) for v in pop.values())
    metrics={k:population_metrics(v,parents[k[1]]['best_channel_ids'],load('s0_manifest.json')['curvature_layers']) for k,v in pop.items()}
    pm={arm:{m:{p:float(np.mean([metrics[arm,p,s][m] for s in seeds])) for p in pids} for m in next(iter(metrics.values()))} for arm in arms}
    def summary(v):
        groups=[parents[p]['split_group'] for p in pids]
        if len(set(groups))<2:return dict(estimate=float(np.mean(list(v.values()))),ci_two95=None,status='INSUFFICIENT_GROUPS')
        return cluster_summary([v[p] for p in pids],groups)
    pairs=[]
    for t in (.6,.7,.8,.9):
        r,w,z=f'relaxed_{t:.1f}',f'raw_{t:.1f}',f'random_{t:.1f}'
        pairs.extend([(r,'none'),(w,'none'),(r,z),(w,z),(r,w),(z,'none')])
    result=dict(parent_metrics=pm,arms={arm:{m:summary(v) for m,v in mm.items()} for arm,mm in pm.items()},comparisons={x+'-'+y:{m:summary({p:pm[x][m][p]-pm[y][m][p] for p in pids}) for m in pm[x]} for x,y in pairs})
    assert len(result['comparisons'])==24
    if a.smoke:return # Deliberately discard, never print or persist smoke metrics.
    original.cmd_analyze(a)
    published=load('results.json')
    assert result['parent_metrics']==published['parent_metrics']
    assert result['comparisons']==published['comparisons']
    original.dump(a.out/'analysis_plumbing_check.json',dict(passed=True,parent_metrics_equal=True,comparisons_equal=True))
