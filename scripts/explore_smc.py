"""Frozen exploratory SMC stages. Formal stages require a clean pushed source clone."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_geometry_information as gi
from xtbflow.v1.data import file_hash
from xtbflow.v1.smc import selection_plan, restart_state, population_metrics
from xtbflow.v1.xtb_score import score_candidate
from xtbflow.v1.geometry_information import xtb_energy_kcal, event_table, ranking_metrics, within_parent_rho
from xtbflow.v1.metrics import cluster_summary


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + '\n')


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


def context(a):
    cfg = json.loads(a.config.read_text())
    base = json.loads(Path(cfg['input_config']).read_text())
    root, cat, split, parents, refs = gi.load_inputs(base, a.limit_parents)
    queries = [r for r in rows(cat / 'development_queries.jsonl') if r['query_id'] in {p['query_id'] for p in parents.values()}]
    if file_hash(cfg['pg1']) != cfg['pg1_sha256']:
        raise RuntimeError('STOP: pg1 input hash mismatch')
    if not a.smoke:
        if git('status', '--porcelain'):
            raise RuntimeError('formal stages require fully clean clone')
        sha = git('rev-parse', 'HEAD')
        if not git('branch', '-r', '--contains', sha):
            raise RuntimeError('source commit is not present in a fetched remote branch')
        if a.limit_parents or a.proposals != 32 or a.seeds != [0, 1, 2]:
            raise ValueError('formal stages require complete frozen population')
    a.out.mkdir(parents=True, exist_ok=True)
    return cfg, base, root, cat, split, parents, refs, queries


def manifest(a, ctx, stage, started, extra):
    cfg, base, root, cat, split, parents, refs, queries = ctx
    inputs = [a.config, Path(cfg['input_config']), Path(cfg['pg1']), cat / 'split_manifest.json',
              cat / 'parent_catalog.json', cat / 'reference_catalog.jsonl', cat / 'development_queries.jsonl',
              Path(cfg['check_a'])]
    dump(a.out / f'{stage}_manifest.json', dict(stage=stage, source_commit=git('rev-parse', 'HEAD'),
         dirty=bool(git('status', '--porcelain')), config_sha256=file_hash(a.config),
         inputs={str(p): file_hash(p) for p in inputs}, slurm_job_id=os.getenv('SLURM_JOB_ID'),
         smoke=a.smoke, elapsed_s=time.monotonic()-started, workers=a.workers, **extra))


def score_job(item):
    z, x, bh, br, anchor, cfg, relaxed = item
    return score_candidate(z, x, bh, br, anchor, cfg, relaxed=relaxed, gradient_job=gi._force_job)


def anchors(parents, cfg, pool):
    keys = sorted(parents)
    values = pool.map(gi._xtb_job, [(parents[p]['atomic_numbers'], parents[p]['x_r'], cfg) for p in keys])
    if any(s != 'ok' for _, s in values):
        raise RuntimeError('STOP: anchor energy failure')
    return {p: e for p, (e, _) in zip(keys, values)}


def score_summary(scores):
    def dist(values):
        return dict(n=len(values), quantiles=np.quantile(values, [0, .25, .5, .75, 1]).tolist()) if values else dict(n=0, quantiles=[])
    return dict(n=len(scores), failures=sum(s['status'] != 'ok' for s in scores),
                negative_curvature=sum(s['status']=='ok' and s['curvature'] is not None and s['curvature']<0 for s in scores),
                calls=sum(s['calls'] for s in scores), elapsed_s=sum(s['elapsed_s'] for s in scores),
                energy_drop=dist([s['energy_drop'] for s in scores if s['energy_drop'] is not None]))


def cmd_s0(a):
    started=time.monotonic(); ctx=context(a)
    cfg,base,root,cat,split,parents,refs,queries=ctx
    reference={r['reference_id']:r for r in refs}
    original=[r for r in rows(cfg['check_a']) if r['parent_id'] in parents]
    with mp.get_context('spawn').Pool(a.workers) as pool:
        ea=anchors(parents,base['xtb'],pool)
        jobs=[]
        for r in original:
            ref=reference[r['reference_id']]; p=parents[r['parent_id']]
            jobs.append((p['atomic_numbers'],ref['x_ts'],ref['b_p'],p['b_r'],ea[r['parent_id']],base['xtb'],True))
        scores=pool.map(score_job,jobs,chunksize=1)
    out=[]
    for r,s in zip(original,scores):
        out.append(dict(reference_id=r['reference_id'],parent_id=r['parent_id'],formula=r['formula'],
                        channel_id=r['channel_id'],catalog_barrier_kcal=r['catalog_barrier_kcal'],
                        expected_raw=r['xtb_ts'],score=s))
    dump(a.out/'s0_rows.json',out)
    error=max(abs(r['score']['raw_barrier']-r['expected_raw']) if r['score']['raw_barrier'] is not None else float('inf') for r in out)
    def rho(field):
        vals={}
        for p in sorted(parents):
            pr=[dict(r, prediction=r['score'][field]) for r in out if r['parent_id']==p]
            if any(r['prediction'] is None for r in pr): continue
            _, target, pred=event_table(pr,'prediction','min')
            m=ranking_metrics(target,pred,pair_gap=2.)
            if m is not None and m['rho'] is not None: vals[p]=m['rho']
        return cluster_summary(list(vals.values()),[parents[p]['split_group'] for p in vals]) if len({parents[p]['split_group'] for p in vals})>=2 else None
    raw=rho('raw_barrier'); relaxed=rho('barrier')
    fraction=sum(s['status']=='ok' and s['curvature']<0 for s in scores)/len(scores)
    gate=error<=1e-6 and (a.smoke or abs(raw['estimate']-cfg['s0_rho'])<1e-9)
    manifest(a,ctx,'s0',started,dict(n=len(out),raw_max_error=error if np.isfinite(error) else None,
             raw_rho=raw,relaxed_rho=relaxed,negative_curvature_fraction=fraction,
             curvature_layers=fraction>=.5,gate_passed=gate,statistics=score_summary(scores)))
    if not gate: raise RuntimeError('STOP: S0 reproduction gate failed')
    print('s0 plumbing completed' if a.smoke else 'S0 gate passed',flush=True)


def cmd_g1(a):
    from xtbflow.v1.assets import load_generator
    from xtbflow.v1.interfaces import query_from_parents
    from xtbflow.v1.sampler import initial_state, subset_query
    from xtbflow.v1.explore_rollout import recorded_rollout
    started=time.monotonic();ctx=context(a)
    cfg,base,root,cat,split,parents,refs,queries=ctx
    if not a.smoke and not os.getenv('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    device='cpu' if a.smoke else 'cuda'
    q_all=query_from_parents(queries,device); batches=[];models={}
    for seed in a.seeds:
        net,meta=load_generator(root,'joint',seed,device);models[str(seed)]=dict(path=meta['path'],sha256=meta['sha256'])
        pairs=[(i,j) for i in range(len(queries)) for j in range(a.proposals)]
        for start in range(0,len(pairs),256):
            chunk=pairs[start:start+256];q=subset_query(q_all,[i for i,j in chunk])
            st=initial_state(q,[j for i,j in chunk],seed,namespace='explore_time_window',sigma_b=meta['sigma_b'],sigma_x=meta['sigma_x'])
            r=recorded_rollout(net,q,st,path='geometry_lead3',n_steps=50,dual_time=meta['dual_time'])
            saved={k:r[k][[30,35,40,45,50]].cpu() for k in ('b_trace','x_trace','b_hat','x_hat')}
            saved.update(seed=seed,pairs=chunk,query_indices=[i for i,j in chunk],meta=meta)
            name=f'g1-{seed}-{start:04d}.pt';torch.save(saved,a.out/name);batches.append(name)
            print(f'g1 batch {seed}/{start} saved',flush=True)
    manifest(a,ctx,'g1',started,dict(batches=batches,generators=models,queries=queries,steps=[30,35,40,45,50]))


def decode_endpoint(parent, refs, b, x):
    from xtbflow.m0.sampler import decode_be
    from xtbflow.m0.batching import ELEMENT_INDEX
    from xtbflow.v1.proxy import CatalogueMatcher,valid_endpoint
    from xtbflow.v1.data import canonical_event
    n=len(parent['atomic_numbers']); m=CatalogueMatcher(parent,refs)
    elements=torch.tensor([ELEMENT_INDEX[z] for z in parent['atomic_numbers']])
    bp,_=decode_be(b[:n,:n],elements,n)
    ok=bp is not None and valid_endpoint(m.z,m.br,bp)
    bp=bp if ok else None
    channel=canonical_event(m.br,bp,m.perms) if ok else None
    match=m.match(bp,x[:n].numpy(),is_fallback=not ok,decode_status='VALID' if ok else 'DECODE_FAILED')
    known=[r['catalog_barrier_kcal'] for r in refs if r['channel_id']==channel]
    return dict(channel_id=channel,proxy_status=match['proxy_status'],hits_best_event=match['hits_best_event'],
                hits_best_reference=match['hits_best_reference'],event_utility=match['event_utility'],
                event_barrier_kcal=min(known) if known else None)


def cmd_c1(a):
    started=time.monotonic();ctx=context(a)
    cfg,base,root,cat,split,parents,refs,queries=ctx
    s0=json.loads((a.out/'s0_manifest.json').read_text())
    if not s0['gate_passed']:raise RuntimeError('STOP: S0 not passed')
    g1=json.loads((a.out/'g1_manifest.json').read_text());byq={p['query_id']:p for p in parents.values()}
    pg={(r['parent_id'],r['seed'],r['proposal']):r for r in rows(cfg['pg1'])}
    decoded=[];checkpoint=[];plans=[];mismatches=0;maxerr=0.
    with mp.get_context('spawn').Pool(a.workers) as pool:
        ea=anchors(parents,base['xtb'],pool)
        for name in g1['batches']:
            b=torch.load(a.out/name,weights_only=False); seed=b['seed']; jobs=[];keys=[]
            for c,(i,j) in enumerate(b['pairs']):
                p=byq[queries[i]['query_id']];pid=p['parent_id'];n=len(p['atomic_numbers'])
                rr=decode_endpoint(p,[r for r in refs if r['parent_id']==pid],b['b_trace'][-1,c],b['x_trace'][-1,c])
                rr.update(parent_id=pid,seed=seed,proposal=j);decoded.append(rr)
                old=pg[pid,seed,j];mismatches+=int(rr['channel_id']!=old['events'][-1] or rr['proxy_status']!=old['proxy_status'])
                for t,k in enumerate((30,35,40,45)):
                    keys.append((pid,seed,j,k))
                    jobs.append((p['atomic_numbers'],b['x_hat'][t,c,:n].numpy(),b['b_hat'][t,c,:n,:n].numpy(),p['b_r'],ea[pid],base['xtb'],True))
            scored=pool.map(score_job,jobs,chunksize=1);lookup={}
            for key,s in zip(keys,scored):
                pid,seed,j,k=key;target=pg[pid,seed,j]['xtb_delta_kcal'][k//5]
                raw=s['raw_barrier']
                if raw is None or not np.isfinite(target):
                    error=0. if raw is None and not np.isfinite(target) else float('inf')
                else:error=abs(raw-target)
                maxerr=max(maxerr,error)
                record=dict(parent_id=pid,seed=seed,proposal=j,step=k,score=s)
                checkpoint.append(record);lookup[key]=s
            for start in range(0,len(b['pairs']),32):
                i,_=b['pairs'][start];pid=byq[queries[i]['query_id']]['parent_id']
                for k in (30,35,40,45):
                    for scorer in ('raw','relaxed','random'):
                        scores=[lookup[pid,seed,j,k] for j in range(32)]
                        if scorer=='raw':scores=[dict(s,status='ok' if s['raw_barrier'] is not None else 'failed',barrier=s['raw_barrier']) for s in scores]
                        survivors,clones=selection_plan(pid,seed,scores,k,scorer,curvature_layers=s0['curvature_layers'])
                        plans.append(dict(batch=name,start=start,parent_id=pid,seed=seed,step=k,arm=f'{scorer}_{k/50:.1f}',survivors=survivors,clones=clones))
    dump(a.out/'none_decoded.json',decoded);dump(a.out/'checkpoint_scores.json',checkpoint);dump(a.out/'plans.json',plans)
    gate=mismatches==0 and maxerr<=1e-6
    manifest(a,ctx,'c1',started,dict(gate_passed=gate,event_status_mismatches=mismatches,raw_max_error=maxerr if np.isfinite(maxerr) else None,n=len(decoded),statistics=score_summary([r['score'] for r in checkpoint])))
    if not gate:raise RuntimeError('STOP: C1 reproduction gate failed')
    print('C1 gate passed',flush=True)


def cmd_g2(a):
    from xtbflow.v1.assets import load_generator
    from xtbflow.v1.interfaces import query_from_parents, State
    from xtbflow.v1.sampler import subset_query,rollout
    from xtbflow.v1.clocks import clock_grid
    started=time.monotonic();ctx=context(a)
    cfg,base,root,cat,split,parents,refs,queries=ctx
    if not json.loads((a.out/'c1_manifest.json').read_text())['gate_passed']:raise RuntimeError('C1 gate')
    if not a.smoke and not os.getenv('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    device='cpu' if a.smoke else 'cuda';qa=query_from_parents(queries,device)
    plans=json.loads((a.out/'plans.json').read_text());tb,tx=clock_grid('geometry_lead3',50);files=[]
    models={}
    for name in json.loads((a.out/'g1_manifest.json').read_text())['batches']:
        data=torch.load(a.out/name,weights_only=False);seed=data['seed']
        if seed not in models:models[seed]=load_generator(root,'joint',seed,device)
        net,meta=models[seed];q=subset_query(qa,data['query_indices'])
        for arm in sorted({r['arm'] for r in plans}):
            pp=[r for r in plans if r['batch']==name and r['arm']==arm];k=pp[0]['step'];t=[30,35,40,45].index(k)
            b=data['b_trace'][t].to(device).clone();x=data['x_trace'][t].to(device).clone()
            for plan in pp:
                start=plan['start'];dest=[start+d for d,s in plan['clones']];anc=[start+s for d,s in plan['clones']]
                cq=subset_query(q,anc)
                cs=restart_state(cq,data['b_hat'][t,anc].to(device),data['x_hat'][t,anc].to(device),
                    [s for d,s in plan['clones']],seed,k,float(tb[k]),float(tx[k]),meta['sigma_b'],meta['sigma_x'])
                b[dest]=cs.b;x[dest]=cs.x
            state=State(b,x,torch.full((len(b),),float(tb[k]),device=device),torch.full((len(b),),float(tx[k]),device=device))
            out=rollout(net,q,state,role='joint',path='geometry_lead3',n_steps=50,start=k,dual_time=meta['dual_time'])
            if out.failed.any():raise RuntimeError('nonfinite continuation')
            target=f'g2-{arm}-{name[3:]}';torch.save(dict(b=out.state.b.cpu(),x=out.state.x.cpu(),batch=name,arm=arm),a.out/target);files.append(target)
            print(f'g2 {target} saved',flush=True)
    manifest(a,ctx,'g2',started,dict(files=files))


def cmd_c2(a):
    started=time.monotonic();ctx=context(a)
    cfg,base,root,cat,split,parents,refs,queries=ctx
    byq={p['query_id']:p for p in parents.values()};plans=json.loads((a.out/'plans.json').read_text())
    none={(r['parent_id'],r['seed'],r['proposal']):r for r in json.loads((a.out/'none_decoded.json').read_text())}
    tasks=[('none',name) for name in json.loads((a.out/'g1_manifest.json').read_text())['batches']]
    tasks += [('continued',name) for name in json.loads((a.out/'g2_manifest.json').read_text())['files']]
    allrows=[];survivors={}
    with mp.get_context('spawn').Pool(a.workers) as pool:
        ea=anchors(parents,base['xtb'],pool)
        for kind,name in tasks:
            d=torch.load(a.out/name,weights_only=False)
            if kind=='none':source=d;arm='none';bb=d['b_trace'][-1];xx=d['x_trace'][-1];batch=name
            else:source=torch.load(a.out/d['batch'],weights_only=False);arm=d['arm'];bb=d['b'];xx=d['x'];batch=d['batch']
            jobs=[];local=[]
            for c,(i,j) in enumerate(source['pairs']):
                p=byq[queries[i]['query_id']];pid=p['parent_id'];n=len(p['atomic_numbers']);seed=source['seed']
                r=decode_endpoint(p,[r for r in refs if r['parent_id']==pid],bb[c],xx[c]);r.update(parent_id=pid,seed=seed,proposal=j,arm=arm,ancestor=j,is_clone=False)
                if arm!='none':
                    plan=next(v for v in plans if v['batch']==batch and v['arm']==arm and v['parent_id']==pid)
                    clones=dict(plan['clones'])
                    if j in clones:r.update(ancestor=clones[j],is_clone=True)
                    else:
                        st=survivors.setdefault(arm,dict(n=0,mismatches=0,max_abs_dx=0.))
                        st['n']+=1;st['mismatches']+=int(r['channel_id']!=none[pid,seed,j]['channel_id'])
                        st['max_abs_dx']=max(st['max_abs_dx'],float((xx[c]-source['x_trace'][-1,c]).abs().max()))
                local.append(r);jobs.append((p['atomic_numbers'],xx[c,:n].numpy(),bb[c,:n,:n].numpy(),p['b_r'],ea[pid],base['xtb'],True))
            scores=pool.map(score_job,jobs,chunksize=1)
            for r,s in zip(local,scores):r['score']=s
            allrows.extend(local)
            print(f'c2 {name} scored',flush=True)
    for st in survivors.values():st['agreement']=1-st['mismatches']/st['n']
    gate=all(st['agreement']>=.999 for st in survivors.values()) and len(survivors)==12
    dump(a.out/'endpoint_rows.json',allrows)
    manifest(a,ctx,'c2',started,dict(gate_passed=gate,survivor_checks=survivors,n=len(allrows),statistics=score_summary([r['score'] for r in allrows])))
    if not gate:raise RuntimeError('STOP: C2 survivor gate failed')


def cmd_analyze(a):
    ctx=context(a);cfg,base,root,cat,split,parents,refs,queries=ctx
    if a.smoke:raise ValueError('smoke must not output scientific metrics')
    manifests={s:json.loads((a.out/f'{s}_manifest.json').read_text()) for s in ('s0','g1','c1','g2','c2')}
    if not all(manifests[s]['gate_passed'] for s in ('s0','c1','c2')):raise RuntimeError('reproduction gate failed')
    endpoint=json.loads((a.out/'endpoint_rows.json').read_text());plans=json.loads((a.out/'plans.json').read_text())
    checkpoint=json.loads((a.out/'checkpoint_scores.json').read_text())
    layers=manifests['s0']['curvature_layers'];pop={};arms=sorted({r['arm'] for r in endpoint})
    if len(arms)!=13 or len(endpoint)!=13*62*3*32:raise ValueError('incomplete endpoint coverage')
    for r in endpoint:pop.setdefault((r['arm'],r['parent_id'],r['seed']),[]).append(r)
    expected={(arm,p,seed) for arm in arms for p in parents for seed in (0,1,2)}
    if set(pop)!=expected or any(len(v)!=32 or {r['proposal'] for r in v}!=set(range(32)) for v in pop.values()):raise ValueError('population coverage')
    pm={k:population_metrics(v,parents[k[1]]['best_channel_ids'],layers) for k,v in pop.items()}
    def summary(values):
        keys=sorted(values)
        if len({parents[p]['split_group'] for p in keys})<2:return dict(estimate=float(np.mean(list(values.values()))) if keys else None,n_parents=len(keys),ci_two95=None,status='INSUFFICIENT_GROUPS')
        return cluster_summary([values[p] for p in keys],[parents[p]['split_group'] for p in keys])
    parent_metrics={arm:{metric:{p:float(np.mean([pm[arm,p,s][metric] for s in (0,1,2)])) for p in sorted(parents)} for metric in next(iter(pm.values()))} for arm in arms}
    armstats={arm:{m:summary(v) for m,v in mm.items()} for arm,mm in parent_metrics.items()}
    comparisons={}
    for k in (.6,.7,.8,.9):
        r,w,z=f'relaxed_{k:.1f}',f'raw_{k:.1f}',f'random_{k:.1f}'
        for x,y in ((r,'none'),(w,'none'),(r,z),(w,z),(r,w),(z,'none')):
            comparisons[f'{x}-{y}']={m:summary({p:parent_metrics[x][m][p]-parent_metrics[y][m][p] for p in parents}) for m in parent_metrics[x]}
    readings={}
    for arm in arms:
        if arm=='none' or arm.startswith('random'):continue
        low,high=comparisons[arm+'-none']['H']['ci_two95'];random='random_'+arm.split('_')[1]
        rl,rh=comparisons[arm+'-'+random]['H']['ci_two95']
        if high<0:label='SELECTION_HURTS'
        elif low<=0<=high:label='NO_GAIN_OVER_POSTHOC'
        elif rl>0:label='IN_GENERATION_SELECTION_HELPS'
        elif rl<=0<=rh:label='NOT_PHYSICS_SPECIFIC'
        else:label='UNCLASSIFIED_BY_FROZEN_TABLE'
        readings[arm]=label
    diagnostics={}
    for arm in arms:
        if arm=='none':continue
        per={};false_prune={}
        for p in parents:
            vals=[];prune=[]
            for seed in (0,1,2):
                rr=pop[arm,p,seed];nr=pop['none',p,seed];byj={r['proposal']:r for r in nr}
                plan=next(r for r in plans if r['arm']==arm and r['parent_id']==p and r['seed']==seed)
                clones=[r for r in rr if r['is_clone']];surv=[r for r in rr if not r['is_clone']]
                changed=sum(r['channel_id']!=byj[r['ancestor']]['channel_id'] for r in clones)/len(clones)
                only=bool(any(r['hits_best_event'] for r in clones) and not any(r['hits_best_event'] for r in surv))
                vals.append((changed,float(only)))
                if any(r['hits_best_event'] for r in nr):prune.append(float(not any(byj[j]['hits_best_event'] for j in plan['survivors'])))
            per[p]=np.mean(vals,axis=0).tolist()
            if prune:false_prune[p]=float(np.mean(prune))
        diagnostics[arm]=dict(clone_event_change=summary({p:v[0] for p,v in per.items()}),clone_only_best=summary({p:v[1] for p,v in per.items()}),false_prune=summary(false_prune))
    none={(r['parent_id'],r['seed'],r['proposal']):r for r in endpoint if r['arm']=='none'}
    correlations={}
    for step in (30,35,40,45):
        for scorer,field in (('raw','raw_barrier'),('relaxed','barrier')):
            seedvals={}
            for seed in (0,1,2):
                rr=[r for r in checkpoint if r['step']==step and r['seed']==seed and r['score'][field] is not None and none[r['parent_id'],seed,r['proposal']]['event_barrier_kcal'] is not None]
                seedvals[seed]=within_parent_rho([r['score'][field] for r in rr],[none[r['parent_id'],seed,r['proposal']]['event_barrier_kcal'] for r in rr],[r['parent_id'] for r in rr],min_n=4)
            common=set.intersection(*(set(v) for v in seedvals.values()))
            correlations[f'{scorer}_{step/50:.1f}']=summary({p:float(np.mean([seedvals[s][p] for s in (0,1,2)])) for p in sorted(common)})
    def layers_count(rr):
        from xtbflow.v1.smc import score_key
        count=[0,0,0]
        for r in rr:count[score_key(r['score'],curvature_layers=layers)[0]]+=1
        return count
    result=dict(version=cfg['version'],exploratory=True,primary_arm='relaxed_0.7',arms=armstats,comparisons=comparisons,readings=readings,
                diagnostics=diagnostics,checkpoint_correlations=correlations,parent_metrics=parent_metrics,
                reproduction={s:manifests[s] for s in ('s0','c1','c2')},
                scorer_statistics={arm:dict(score_summary([r['score'] for r in endpoint if r['arm']==arm]),layers=layers_count([r for r in endpoint if r['arm']==arm])) for arm in arms},
                resource_manifests=manifests,raw_output_sha256={name:file_hash(a.out/name) for name in ('s0_rows.json','none_decoded.json','checkpoint_scores.json','plans.json','endpoint_rows.json')},
                limitations=cfg['limitations'])
    dump(a.out/'results.json',result)
    print('deterministic analysis written',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('cmd',choices=['s0','g1','c1','g2','c2','analyze'])
    p.add_argument('--config',type=Path,default=Path('configs/explore/smc_selection.json'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=48)
    p.add_argument('--limit-parents',type=int);p.add_argument('--proposals',type=int,default=32)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if not 1<=a.workers<=48:raise ValueError('workers must be 1..48')
    torch.set_num_threads(1)
    globals()['cmd_'+a.cmd](a)


if __name__=='__main__':main()
