"""Strict-order repetition of smc-1; original implementation and outputs unchanged."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import time

import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import explore_smc as original
from smc_strict_stages import g2,analyze_plumbing
from xtbflow.v1.smc import restart_state,population_metrics
from xtbflow.v1.interfaces import query_from_parents
from xtbflow.v1.sampler import subset_query
from xtbflow.v1.clocks import clock_grid


def materialize(a):
    ctx=original.context(a);cfg,base,root,cat,split,parents,refs,queries=ctx
    g1=json.loads((a.out/'g1_manifest.json').read_text());plans=json.loads((a.out/'plans.json').read_text())
    qa=query_from_parents(queries,'cpu');tb,tx=clock_grid('geometry_lead3',50);files={};count=0
    dest=a.out/'continuation_initial';dest.mkdir(exist_ok=False)
    for name in g1['batches']:
        data=torch.load(a.out/name,map_location='cpu',weights_only=False);q=subset_query(qa,data['query_indices']);meta=data['meta']
        for arm in sorted({p['arm'] for p in plans}):
            pp=[p for p in plans if p['batch']==name and p['arm']==arm];k=pp[0]['step'];t=[30,35,40,45].index(k)
            b=data['b_trace'][t].clone();x=data['x_trace'][t].clone()
            for p in pp:
                start=p['start'];dd=[start+d for d,s in p['clones']];ss=[start+s for d,s in p['clones']]
                state=restart_state(subset_query(q,ss),data['b_hat'][t,ss],data['x_hat'][t,ss],[s for d,s in p['clones']],data['seed'],k,float(tb[k]),float(tx[k]),meta['sigma_b'],meta['sigma_x'])
                b[dd]=state.b;x[dd]=state.x;count+=len(dd)
            path=dest/f'{arm}-{name}';torch.save(dict(b=b,x=x,step=k,arm=arm,batch=name),path);files[str(path.relative_to(a.out))]=original.file_hash(path)
    original.manifest(a,ctx,'initial_states',time.monotonic(),dict(files=files,clone_count=count,persisted_before_g2=True,plans_sha256=original.file_hash(a.out/'plans.json')))


def smoke(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU smoke requires hidden CUDA')
    if original.git('status','--porcelain'):raise RuntimeError('clean source required')
    a.smoke=True;a.limit_parents=2;a.proposals=32;a.seeds=[0];a.workers=1
    # Suppress scientific summaries. Intermediate candidate scores are plumbing
    # artifacts, not inspected or used to choose parameters.
    old_dump=original.dump
    def smoke_dump(path,obj):
        if Path(path).name=='s0_manifest.json':
            obj={k:v for k,v in obj.items() if k not in ('raw_rho','relaxed_rho','negative_curvature_fraction','statistics','raw_max_error')}
            obj['curvature_layers']=False # fixed smoke-only setting, no S0 tuning
        old_dump(path,obj)
    original.dump=smoke_dump
    completed=[]
    with contextlib.redirect_stdout(io.StringIO()):
        original.cmd_s0(a);completed.append('s0')
        original.cmd_g1(a);completed.append('g1')
        from smc_cpu_baseline import baseline
        cpu_rows=baseline(a);completed.append('cpu_same_platform_repeat')
        old_rows=original.rows
        pg1=original.context(a)[0]['pg1']
        def smoke_rows(path):
            return cpu_rows if str(path)==str(pg1) else old_rows(path)
        # Authorized smoke-only reference substitution. context still verifies
        # the immutable pg1 hash; formal C1 never enters this scoped adapter.
        original.rows=smoke_rows
        try:original.cmd_c1(a)
        finally:original.rows=old_rows
        completed.append('c1_cpu_reference')
        materialize(a);completed.append('initial_states')
        g2(a);completed.append('g2')
        original.cmd_c2(a);completed.append('c2')
        rr=json.loads((a.out/'endpoint_rows.json').read_text());ctx=original.context(a);parents=ctx[5]
        groups={}
        for r in rr:groups.setdefault((r['arm'],r['parent_id'],r['seed']),[]).append(r)
        assert len(groups)==26 and all(len(v)==32 for v in groups.values())
        for (_,pid,_),values in groups.items():
            metrics=population_metrics(values,parents[pid]['best_channel_ids'],False)
            assert set(metrics)=={'hit@1','hit@2','hit@4','H','hit_infinity','event_utility','distinct_legal_events','legal_rate','best_reference_rate'}
        analyze_plumbing(a);completed.append('analysis_aggregation_and_comparisons')
    original.dump=old_dump
    original.dump(a.out/'smoke_complete.json',dict(passed=True,stages=completed,parents=2,proposals=32,seeds=[0],scientific_metrics_emitted=False,c1_reference='independent repeated CPU rollout (authorized smoke-only amendment)',formal_c1_reference='original pg1, unchanged 1e-6 kcal/mol tolerance',source_commit=original.git('rev-parse','HEAD'),dirty=False,slurm_job_id=None))
    print('Full CPU smoke plumbing passed; no scientific metrics emitted.')


def main():
    p=argparse.ArgumentParser();p.add_argument('cmd',choices=['smoke','s0','g1','c1','g2','c2','analyze']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke-proof',type=Path);p.add_argument('--workers',type=int,default=48)
    a=p.parse_args();a.config=Path('configs/explore/smc_selection.json');a.smoke=False;a.limit_parents=None;a.proposals=32;a.seeds=[0,1,2]
    if not 1<=a.workers<=48:raise ValueError('workers')
    torch.set_num_threads(1)
    if a.cmd=='smoke':return smoke(a)
    if a.smoke_proof is None:raise RuntimeError('pre-formal smoke proof required')
    proof=json.loads(a.smoke_proof.read_text())
    if not proof['passed'] or proof['source_commit']!=original.git('rev-parse','HEAD'):raise RuntimeError('smoke proof source mismatch')
    if a.cmd=='g2':
        m=json.loads((a.out/'initial_states_manifest.json').read_text())
        assert m['persisted_before_g2'] and not m['dirty']
        for f,h in m['files'].items():assert original.file_hash(a.out/f)==h
    if a.cmd=='g2':g2(a)
    elif a.cmd=='analyze':analyze_plumbing(a)
    else:getattr(original,'cmd_'+a.cmd)(a)
    if a.cmd=='c1':materialize(a)


if __name__=='__main__':main()
