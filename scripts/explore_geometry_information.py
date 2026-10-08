"""Exploratory geometry-information checks on the V1a development split.

heads:        train the matched clean-input heads, then score every development
              reference record with frozen/matched heads and GFN2-xTB (check A).
trajectories: unguided joint rollouts with per-step decoded events, xTB on the
              predicted endpoint and h_X on the current state (check B).
analyze:      the pre-specified summaries in configs/explore/geometry_information.json.

Results are exploratory. Inputs are read-only; outputs go to a new directory.
"""
import argparse
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parent))
from xtbflow.m0.batching import collate
from xtbflow.m0.t1x_data import T1xCache
from xtbflow.v1.data import canonical_event,digest,file_hash,write_json
from xtbflow.v1.geometry_information import (commit_index,event_table,first_informative,
                                             predicted_endpoints,ranking_metrics,
                                             within_parent_rho,xtb_energy_kcal)
from xtbflow.v1.metrics import cluster_summary

GRID_STEPS=50
PREDICTORS=('oracle','E_frozen','X_frozen','N_clean','G_clean','xtb_ts')


def load_inputs(cfg,limit=None):
    root=Path(cfg['data_scope']['run_root']);cat=root/'data'
    split=json.loads((cat/'split_manifest.json').read_text())
    if digest({k:v for k,v in split.items() if k!='split_hash'})!=split['split_hash']:
        raise ValueError('split hash mismatch')
    dev=sorted(split['development']['parent_ids'])
    if limit:dev=dev[:limit]
    dev=set(dev)
    parents={p['parent_id']:p for p in json.loads((cat/'parent_catalog.json').read_text()) if p['parent_id'] in dev}
    refs=[json.loads(line) for line in (cat/'reference_catalog.jsonl').read_text().splitlines()]
    refs=[r for r in refs if r['parent_id'] in dev]
    if set(parents)!=dev or {r['parent_id'] for r in refs}!=dev:raise ValueError('incomplete development catalogue')
    return root,cat,split,parents,refs


def provenance(cfg_path):
    return dict(config_sha256=file_hash(cfg_path),
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                dirty=bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip()))


# ---------------------------------------------------------------- check A

def train_clean(variant,member,rows,tc,device):
    """Matched heads: same seed, init and batches; only the geometry input differs."""
    from v1a_train import lr_at
    from xtbflow.v1.score_training import query_from_training_batch
    from xtbflow.v1.scores import BarrierHead
    torch.manual_seed(member);rng=np.random.default_rng(member)
    if device.type=='cuda':torch.cuda.manual_seed_all(member)
    model=BarrierHead('X').to(device);ema=copy.deepcopy(model).eval().requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=tc['lr'],weight_decay=tc['weight_decay'])
    log=[];started=time.monotonic()
    for step in range(tc['steps']):
        batch,y=rows.sample(rng,tc['batch_size'],device)
        q=query_from_training_batch(batch);one=torch.ones(len(y),device=device)
        x=batch['x_ts'] if variant=='ts' else batch['x_r']
        for group in opt.param_groups:group['lr']=lr_at(step,tc)
        loss=F.huber_loss(model(q,batch['b_p'],x,one,one),y,delta=1.)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite clean-head loss')
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),tc['grad_clip'],error_if_nonfinite=True)
        opt.step()
        with torch.no_grad():
            for pe,p in zip(ema.parameters(),model.parameters()):pe.lerp_(p,1-tc['ema_decay'])
        if (step+1)%1000==0 or step+1==tc['steps']:
            log.append(dict(step=step+1,loss=float(loss.detach()),elapsed_s=time.monotonic()-started))
    return ema,log


def _xtb_job(args):
    z,x,cfg=args
    return xtb_energy_kcal(z,x,cfg)


def cmd_heads(a):
    from xtbflow.v1.assets import load_scores
    from xtbflow.v1.score_training import ScoreRows,query_from_training_batch
    from xtbflow.v1.scores import BarrierEnsemble,BarrierHead,SCORE_CONFIG
    cfg=json.loads(a.config.read_text());v1a=json.loads(Path(cfg['data_scope']['v1a_config']).read_text())
    root,cat,split,parents,refs=load_inputs(cfg,a.limit_parents)
    cache=T1xCache(Path(v1a['data']['cache']))
    # GFN2-xTB on each reference TS and each parent's anchor reactant. The CPU pool
    # forks before CUDA is touched, so workers never inherit a GPU context.
    xcfg=cfg['xtb'];anchors=sorted(parents)
    jobs=[(cache.reaction(r['cache_index'])['z'],cache.reaction(r['cache_index'])['x_ts'],xcfg) for r in refs]
    jobs+=[(parents[p]['atomic_numbers'],parents[p]['x_r'],xcfg) for p in anchors]
    with mp.get_context('fork').Pool(a.workers) as pool:energies=pool.map(_xtb_job,jobs,chunksize=4)
    anchor={p:energies[len(refs)+k] for k,p in enumerate(anchors)}
    # Smoke runs are CPU-only so they can check plumbing on the login node.
    device=torch.device('cpu' if a.smoke else 'cuda')
    if not a.smoke and not torch.cuda.is_available():raise RuntimeError('allocated GPU required')
    a.out.mkdir(parents=True,exist_ok=True);heads=a.out/'clean_heads';heads.mkdir(exist_ok=True)
    rows=ScoreRows(cache,split,cat/'reference_catalog.jsonl')
    spec=cfg['check_a_information_bound']['matched_training']
    tc=dict(v1a['training'],steps=a.train_steps or spec['steps'])
    clean={};logs={}
    for variant in ('reactant','ts'):
        members=[]
        for member in spec['members']:
            path=heads/f'{variant}_{member}.pt'
            model=BarrierHead('X').to(device)
            if path.exists():
                saved=torch.load(path,map_location=device,weights_only=False)
                if saved['meta']['steps']!=tc['steps']:raise ValueError('clean head step count differs')
                model.load_state_dict(saved['ema'])
            else:
                ema,log=train_clean(variant,member,rows,tc,device);model.load_state_dict(ema.state_dict())
                torch.save(dict(meta=dict(variant=variant,member=member,steps=tc['steps'],median=rows.median,
                                          scale=rows.scale,score_config=SCORE_CONFIG,log=log),
                                ema=ema.state_dict()),path)
                logs[f'{variant}_{member}']=log
            members.append(model)
        clean[variant]=BarrierEnsemble(members,rows.median,rows.scale)
    frozen={k:load_scores(root,k,device)[0] for k in ('E','X')}
    out=[]
    with torch.no_grad():
        for start in range(0,len(refs),64):
            chunk=refs[start:start+64]
            batch={k:v.to(device) for k,v in collate([cache.reaction(r['cache_index']) for r in chunk]).items()}
            if [int(i) for i in batch['index'].tolist()]!=[r['cache_index'] for r in chunk]:
                raise ValueError('cache index mismatch')
            q=query_from_training_batch(batch);one=torch.ones(len(chunk),device=device);zero=torch.zeros_like(one)
            states=dict(E_frozen=(frozen['E'],batch['x_r'],one,zero),X_frozen=(frozen['X'],batch['x_ts'],one,one),
                        N_clean=(clean['reactant'],batch['x_r'],one,one),G_clean=(clean['ts'],batch['x_ts'],one,one))
            values={}
            for name,(ens,x,tb,tx) in states.items():
                c=ens.components(q,batch['b_p'],x,tb,tx)
                values[name]=c['mean_kcal'].cpu().numpy();values[name+'_sd']=c['sd_kcal'].cpu().numpy()
            for i,r in enumerate(chunk):
                p=parents[r['parent_id']]
                out.append(dict(reference_id=r['reference_id'],parent_id=r['parent_id'],formula=p['split_group'],
                                channel_id=r['channel_id'],cache_index=r['cache_index'],
                                catalog_barrier_kcal=r['catalog_barrier_kcal'],oracle=r['catalog_barrier_kcal'],
                                **{k:float(v[i]) for k,v in values.items()}))
    for row,(e,status) in zip(out,energies[:len(refs)]):
        ea,sa=anchor[row['parent_id']]
        row['xtb_ts']=e-ea if status=='ok' and sa=='ok' else float('nan')
        row['xtb_status']=status if sa=='ok' else 'anchor_'+sa
    with (a.out/'check_a_rows.jsonl').open('w') as f:
        for row in out:f.write(json.dumps(row,allow_nan=True)+'\n')
    write_json(a.out/'check_a_manifest.json',dict(
        stage='check_a',n_parents=len(parents),n_references=len(refs),train_steps=tc['steps'],
        smoke=bool(a.smoke),clean_training_logs=logs,rows_sha256=file_hash(a.out/'check_a_rows.jsonl'),
        cache_sha256=file_hash(v1a['data']['cache']),split_hash=split['split_hash'],**provenance(a.config)))
    print(json.dumps(dict(stage='check_a',references=len(out),
                          xtb_failures=sum(r['xtb_status']!='ok' for r in out)),indent=1))


# ---------------------------------------------------------------- check B

MATCHERS={};XCFG={};ANCHOR={}


def _trajectory_job(item):
    from xtbflow.m0.sampler import decode_be
    from xtbflow.v1.proxy import valid_endpoint
    m=MATCHERS[item['query_id']];n=len(m.z);events=[];changes=[]
    upper=np.triu_indices(n,1)
    for k in range(len(item['b_hat'])):
        bp,fallback=decode_be(torch.from_numpy(item['b_hat'][k]),item['element_index'],n)
        ok=bp is not None and valid_endpoint(m.z,m.br,bp)
        events.append(canonical_event(m.br,bp,m.perms) if ok else None)
        # Atom pairs whose rounded bond order differs from the reactant, valid or not.
        diff=fallback[upper]!=np.asarray(m.br)[upper]
        changes.append(frozenset(zip(upper[0][diff].tolist(),upper[1][diff].tolist())))
        if k==len(item['b_hat'])-1:final_bp=bp if ok else None
    if final_bp is not None:
        final=np.asarray(final_bp)[upper]!=np.asarray(m.br)[upper]
        changes[-1]=frozenset(zip(upper[0][final].tolist(),upper[1][final].tolist()))
    else:changes[-1]=None
    match=m.match(final_bp,item['x_final'],is_fallback=final_bp is None,
                  decode_status='VALID' if final_bp is not None else 'DECODE_FAILED')
    if match['predicted_channel_id']!=events[-1]:raise RuntimeError('final event disagrees with matcher')
    known=[r['catalog_barrier_kcal'] for r in m.refs if r['channel_id']==events[-1]]
    xtb=[];status=[]
    for x in item['x_grid']:
        e,s=xtb_energy_kcal(m.z,x,XCFG);ea,sa=ANCHOR[item['query_id']]
        xtb.append(e-ea if s=='ok' and sa=='ok' else float('nan'));status.append(s)
    return dict(parent_id=m.parent['parent_id'],formula=m.parent['split_group'],query_id=item['query_id'],
                seed=item['seed'],proposal=item['proposal'],events=events,commit_step=commit_index(events),
                bond_change_commit_step=commit_index(changes),
                proxy_status=match['proxy_status'],hits_best_event=match['hits_best_event'],
                hits_best_reference=match['hits_best_reference'],
                event_barrier_kcal=min(known) if known else None,xtb_delta_kcal=xtb,xtb_status=status,
                hx_mean_kcal=item['hx_mean'],hx_sd_kcal=item['hx_sd'])


def cmd_trajectories(a):
    from xtbflow.v1.assets import load_generator,load_scores
    from xtbflow.v1.clocks import clock_grid,observation_index
    from xtbflow.v1.interfaces import assert_query_fields,query_from_parents
    from xtbflow.v1.proxy import CatalogueMatcher
    from xtbflow.v1.sampler import initial_state,rollout,subset_query
    cfg=json.loads(a.config.read_text());spec=cfg['check_b_time_window']
    root,cat,split,parents,refs=load_inputs(cfg,a.limit_parents)
    qrows=[json.loads(line) for line in (cat/'development_queries.jsonl').read_text().splitlines()]
    by_query={p['query_id']:p for p in parents.values()}
    qrows=[r for r in qrows if r['query_id'] in by_query]
    if len(qrows)!=len(parents):raise ValueError('one development query per parent required')
    for r in qrows:assert_query_fields(r)
    by_parent={}
    for r in refs:by_parent.setdefault(r['parent_id'],[]).append(r)
    MATCHERS.update({q:CatalogueMatcher(p,by_parent[p['parent_id']]) for q,p in by_query.items()})
    XCFG.update(cfg['xtb'])
    ctx=mp.get_context('fork')
    with ctx.Pool(a.workers) as pool:
        energies=pool.map(_xtb_job,[(r['atomic_numbers'],r['x_r'],XCFG) for r in qrows])
    ANCHOR.update({r['query_id']:e for r,e in zip(qrows,energies)})
    a.out.mkdir(parents=True,exist_ok=True);path=a.out/'check_b_trajectories.jsonl'
    if path.exists():raise FileExistsError(path)
    models={};started=time.monotonic();count=0
    # Workers fork with the matchers and anchor energies, before CUDA is initialized.
    with path.open('w') as f,ctx.Pool(a.workers) as pool:
        device=torch.device('cpu' if a.smoke else 'cuda')
        if not a.smoke and not torch.cuda.is_available():raise RuntimeError('allocated GPU required')
        q_all=query_from_parents(qrows,device);score=load_scores(root,'X',device)[0]
        samp=spec['sampling'];k_props=a.proposals or samp['proposals_per_parent_per_seed']
        # The grid is read on t_x, which equals the integration parameter s on every path.
        clock_path=a.path or samp['path']
        tb,tx=clock_grid(clock_path,samp['n_steps']);grid=[observation_index(tx,t) for t in spec['grid_t']]
        for seed in a.seeds:
            net,meta=load_generator(root,'joint',seed,device=device);models[seed]=dict(path=meta['path'],sha256=meta['sha256'])
            pairs=[(i,j) for i in range(len(qrows)) for j in range(k_props)]
            for s in range(0,len(pairs),a.batch_size):
                chunk=pairs[s:s+a.batch_size]
                q=subset_query(q_all,[i for i,_ in chunk])
                st=initial_state(q,[j for _,j in chunk],seed,sampling_seed=samp['sampling_seed'],
                                 namespace=samp['namespace'],sigma_b=meta['sigma_b'],sigma_x=meta['sigma_x'],group='joint')
                r=rollout(net,q,st,role='joint',path=clock_path,n_steps=samp['n_steps'],dual_time=meta['dual_time'])
                if r.failed.any():raise FloatingPointError('numeric rollout failure')
                b_hat=predicted_endpoints(r.b_trace,tb);x_hat=predicted_endpoints(r.x_trace,tx)
                hx=[];hs=[]
                with torch.no_grad():
                    for k in grid:
                        c=score.components(q,r.b_trace[k],r.x_trace[k],torch.full((len(chunk),),float(tb[k]),device=device),
                                           torch.full((len(chunk),),float(tx[k]),device=device))
                        hx.append(c['mean_kcal'].cpu().numpy());hs.append(c['sd_kcal'].cpu().numpy())
                b_hat=b_hat.float().cpu().numpy();x_grid=x_hat[grid].float().cpu().numpy()
                x_final=r.x_trace[-1].float().cpu().numpy()
                items=[]
                for c,(i,j) in enumerate(chunk):
                    n=int(q.atom_mask[c].sum())
                    items.append(dict(query_id=qrows[i]['query_id'],seed=seed,proposal=j,
                                      element_index=q.element_index[c].cpu(),b_hat=b_hat[:,c,:n,:n],
                                      x_grid=x_grid[:,c,:n],x_final=x_final[c,:n],
                                      hx_mean=[float(v[c]) for v in hx],hx_sd=[float(v[c]) for v in hs]))
                for row in pool.imap(_trajectory_job,items,chunksize=2):
                    f.write(json.dumps(row,allow_nan=True)+'\n');count+=1
                f.flush()
                print(json.dumps(dict(seed=seed,done=count,elapsed_s=round(time.monotonic()-started))),flush=True)
    write_json(a.out/'check_b_manifest.json',dict(
        stage='check_b',clock_path=clock_path,n_parents=len(qrows),proposals_per_parent_per_seed=k_props,seeds=list(a.seeds),
        trajectories=count,generators=models,grid_steps=grid,rows_sha256=file_hash(path),
        smoke=bool(a.smoke or a.limit_parents or a.proposals),split_hash=split['split_hash'],**provenance(a.config)))


# ---------------------------------------------------------------- analysis

def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _summary(values,formula):
    keys=sorted(values)
    if len(keys)<2 or len({formula[k] for k in keys})<2:return None
    return cluster_summary([values[k] for k in keys],[formula[k] for k in keys])


def _reading(s,labels):
    if s is None:return 'UNAVAILABLE'
    lo,hi=s['ci_two95']
    return labels[0] if lo>0 else labels[1] if hi<0 else 'INCONCLUSIVE'


def analyze_a(rows,spec):
    formula={r['parent_id']:r['formula'] for r in rows}
    gap=2.;out={}
    for pool in ('min','mean'):
        per={k:{} for k in PREDICTORS}
        for p in sorted(formula):
            pr=[r for r in rows if r['parent_id']==p]
            for k in PREDICTORS:
                if not all(np.isfinite(r[k]) for r in pr):continue
                _,t,v=event_table(pr,k,pool)
                m=ranking_metrics(t,v,pair_gap=gap)
                if m is not None:per[k][p]=m
        block=dict(predictors={},comparisons={})
        for k in PREDICTORS:
            block['predictors'][k]={metric:_summary({p:m[metric] for p,m in per[k].items() if m[metric] is not None},formula)
                                    for metric in ('rho','top1','concordance')}
            block['predictors'][k]['n_parents']=len(per[k])
        for x,y in spec['comparisons']:
            both=set(per[x])&set(per[y]);name=f'{x}-{y}';block['comparisons'][name]={}
            for metric in ('rho','top1','concordance'):
                d={p:per[x][p][metric]-per[y][p][metric] for p in both
                   if per[x][p][metric] is not None and per[y][p][metric] is not None}
                block['comparisons'][name][metric]=_summary(d,formula)
        out[pool]=block
    # Record-level sensitivity: every reference record ranked within its parent.
    rec={}
    for k in PREDICTORS:
        vals=np.array([r[k] for r in rows],dtype=float)
        rec[k]=within_parent_rho(vals,[r['catalog_barrier_kcal'] for r in rows],[r['parent_id'] for r in rows],min_n=2)
    out['record_level']={k:_summary(v,formula) for k,v in rec.items()}
    out['record_level_comparisons']={f'{x}-{y}':_summary({p:rec[x][p]-rec[y][p] for p in set(rec[x])&set(rec[y])},formula)
                                     for x,y in spec['comparisons']}
    primary=out['min']['comparisons']
    out['reading']={name:_reading(primary[name]['rho'],('GEOMETRY_ADDS_RANKING_INFORMATION','EVENT_RANKS_BETTER'))
                    for name in ('X_frozen-E_frozen','G_clean-N_clean')}
    out['reading'].update({name:_reading(primary[name]['rho'],('PHYSICS_RANKS_BETTER','LEARNED_RANKS_BETTER'))
                           for name in ('xtb_ts-E_frozen','xtb_ts-X_frozen')})
    absolute={}
    for k in ('E_frozen','X_frozen','N_clean','G_clean','xtb_ts'):
        err=np.array([r[k]-r['catalog_barrier_kcal'] for r in rows if np.isfinite(r[k])])
        absolute[k]=dict(mae_kcal=float(np.abs(err).mean()),median_signed_kcal=float(np.median(err)),n=len(err))
    out['absolute_error_records']=absolute
    out['n_parents']=len(formula);out['n_records']=len(rows)
    out['xtb_status']={s:sum(r['xtb_status']==s for r in rows) for s in sorted({r['xtb_status'] for r in rows})}
    return out


def analyze_b(rows,spec):
    grid=np.asarray(spec['grid_t'],dtype=float);steps=GRID_STEPS
    formula={r['parent_id']:r['formula'] for r in rows}
    valid=[r for r in rows if r['commit_step'] is not None]
    commit=np.array([r['commit_step']/steps for r in valid])
    out=dict(n_trajectories=len(rows),n_valid_final=len(valid),
             final_status={s:sum(r['proxy_status']==s for r in rows) for s in sorted({r['proxy_status'] for r in rows})})
    if not len(commit):raise ValueError('no trajectory ends in a valid event')
    out['commit']=dict(cdf={f'{t:.1f}':float((commit<=t+1e-9).mean()) for t in grid},
                       median_t=float(np.median(commit)),quartiles_t=np.quantile(commit,[.25,.75]).tolist(),
                       committed_by=dict((f'{t}',float((commit<=t+1e-9).mean())) for t in
                                         spec['guidance_reference_times'].values()))
    # Per-parent commit share by t=0.5, with formula-cluster interval.
    by_parent={}
    for r in valid:by_parent.setdefault(r['parent_id'],[]).append(r['commit_step']/steps<=.5+1e-9)
    out['commit']['parent_share_by_0.5']=_summary({p:float(np.mean(v)) for p,v in by_parent.items()},formula)
    target=np.array([r['event_barrier_kcal'] if r['event_barrier_kcal'] is not None else np.nan for r in rows],dtype=float)
    parents=[r['parent_id'] for r in rows]
    xtb=np.array([r['xtb_delta_kcal'] for r in rows],dtype=float)
    hx=np.array([r['hx_mean_kcal'] for r in rows],dtype=float)
    curves={}
    for name,signal,goal in (('xtb_vs_final_event_barrier',xtb,target),('hX_vs_final_event_barrier',hx,target),
                             ('xtb_vs_final_xtb',xtb,xtb[:,-1])):
        curve=[]
        for g in range(len(grid)):
            curve.append(_summary(within_parent_rho(signal[:,g],goal,parents),formula))
        curves[name]=dict(mean=[c['estimate'] if c else None for c in curve],
                          ci_two95=[c['ci_two95'] if c else None for c in curve],
                          n_parents=[c['n_parents'] if c else 0 for c in curve])
        final=curve[-1]
        t_geo=first_informative(grid,curves[name]['mean'],final['ci_two95'][0] if final else None)
        curves[name]['t_geo']=t_geo
        if t_geo is None:
            curves[name]['window_fraction']=None;curves[name]['reading']='SIGNAL_NEVER_INFORMATIVE'
        else:
            w=float((commit>t_geo+1e-9).mean());curves[name]['window_fraction']=w
            curves[name]['reading']='WINDOW_EXISTS' if w>=.2 else 'NO_USABLE_WINDOW'
    soft=np.array([r['bond_change_commit_step']/steps for r in valid])
    out['commit_bond_change_sensitivity']=dict(
        cdf={f'{t:.1f}':float((soft<=t+1e-9).mean()) for t in grid},median_t=float(np.median(soft)),
        window_fraction={k:(float((soft>v['t_geo']+1e-9).mean()) if v['t_geo'] is not None else None)
                         for k,v in curves.items()})
    out['signals']=curves
    out['xtb_status_by_t']={f'{t:.1f}':{s:sum(r['xtb_status'][g]==s for r in rows)
                                         for s in sorted({r['xtb_status'][g] for r in rows})}
                            for g,t in enumerate(grid)}
    half=steps//2;decodable=[r for r in valid if r['events'][half] is not None]
    changed=[r for r in decodable if r['events'][half]!=r['events'][-1]]
    kept=[r for r in decodable if r['events'][half]==r['events'][-1]]
    out['self_correction']=dict(undecodable_at_half=1-len(decodable)/max(len(valid),1),
                                changed_after_half=len(changed)/max(len(decodable),1),
                                best_event_rate_changed=float(np.mean([r['hits_best_event'] for r in changed])) if changed else None,
                                best_event_rate_unchanged=float(np.mean([r['hits_best_event'] for r in kept])) if kept else None,
                                n_changed=len(changed),n_unchanged=len(kept))
    return out


def cmd_analyze(a):
    cfg=json.loads(a.config.read_text())
    result=dict(version=cfg['version'],exploratory=True,**provenance(a.config))
    if a.check_a:
        result['check_a']=analyze_a(_jsonl(a.check_a),cfg['check_a_information_bound'])
        result['check_a_rows_sha256']=file_hash(a.check_a)
    if a.check_b:
        result['check_b']=analyze_b(_jsonl(a.check_b),cfg['check_b_time_window'])
        result['check_b_rows_sha256']=file_hash(a.check_b)
    write_json(a.out,result)
    print(json.dumps({k:result[k].get('reading') if isinstance(result.get(k),dict) else None
                      for k in ('check_a',)},indent=1))


QUALITY=dict(valid=lambda r:r['proxy_status']!='INVALID_OUTPUT',proxy_match=lambda r:r['proxy_status']=='PROXY_MATCH',
             best_event=lambda r:bool(r['hits_best_event']),best_reference=lambda r:bool(r['hits_best_reference']))


def parent_rates(rows):
    by={}
    for r in rows:by.setdefault(r['parent_id'],[]).append(r)
    return {k:{p:float(np.mean([f(r) for r in v])) for p,v in by.items()} for k,f in QUALITY.items()}


def reproduction(rows,reference):
    key=lambda r:(r['parent_id'],r['seed'],r['proposal'])
    ref={key(r):r for r in reference}
    if set(ref)!={key(r) for r in rows}:raise ValueError('sync rerun does not cover the check B trajectories')
    same=[(r['events'][-1]==ref[key(r)]['events'][-1],r['proxy_status']==ref[key(r)]['proxy_status'],
           r['commit_step']==ref[key(r)]['commit_step']) for r in rows]
    return dict(n=len(rows),final_event=float(np.mean([a for a,_,_ in same])),
                proxy_status=float(np.mean([b for _,b,_ in same])),commit_step=float(np.mean([c for _,_,c in same])))


def cmd_analyze_clock(a):
    cfg=json.loads(a.config.read_text());base=json.loads(Path(cfg['base_config']).read_text())
    spec=base['check_b_time_window'];runs={p:_jsonl(f) for p,f in a.run}
    if 'sync' not in runs or set(runs)-set(cfg['paths']):raise ValueError('runs must include sync and only configured paths')
    formula={r['parent_id']:r['formula'] for r in runs['sync']}
    result=dict(version=cfg['version'],exploratory=True,**provenance(a.config),
                base_config_sha256=file_hash(cfg['base_config']),
                rows_sha256={p:file_hash(f) for p,f in a.run},
                reproduction_vs_check_b=reproduction(runs['sync'],_jsonl(a.reference_sync)),paths={})
    rates={p:parent_rates(rows) for p,rows in runs.items()}
    for p,rows in runs.items():
        b=analyze_b(rows,spec);x=b['signals']['xtb_vs_final_event_barrier']
        entry=dict(time_window=b,rates={k:float(np.mean(list(v.values()))) for k,v in rates[p].items()})
        if p!='sync':
            diff={k:_summary({q:rates[p][k][q]-rates['sync'][k][q] for q in rates['sync'][k]},formula)
                  for k in QUALITY}
            entry['quality_vs_sync']=diff
            within=(diff['best_reference']['estimate']>=-.05 and diff['valid']['estimate']>=-.10)
            w=x['window_fraction']
            entry['reading']=('NO_USABLE_WINDOW' if w is None or w<.2 else
                              'GEOMETRY_LEAD_OPENS_WINDOW' if within else 'WINDOW_WITH_QUALITY_LOSS')
        result['paths'][p]=entry
    write_json(a.out,result)
    print(json.dumps({p:dict(reading=e.get('reading'),window=e['time_window']['signals']['xtb_vs_final_event_barrier']['window_fraction'],
                             t_geo=e['time_window']['signals']['xtb_vs_final_event_barrier']['t_geo'],
                             commit_median=e['time_window']['commit']['median_t'],rates=e['rates'])
                      for p,e in result['paths'].items()},indent=1))
    print(json.dumps(result['reproduction_vs_check_b']))


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='cmd',required=True)
    h=sub.add_parser('heads');t=sub.add_parser('trajectories');z=sub.add_parser('analyze')
    c=sub.add_parser('analyze-clock');c.add_argument('--config',type=Path,required=True)
    c.add_argument('--run',nargs=2,action='append',metavar=('PATH','ROWS'),required=True)
    c.add_argument('--reference-sync',type=Path,required=True);c.add_argument('--out',type=Path,required=True)
    for p in (h,t,z):p.add_argument('--config',type=Path,required=True)
    for p in (h,t):
        p.add_argument('--out',type=Path,required=True);p.add_argument('--limit-parents',type=int)
        p.add_argument('--workers',type=int,default=max(1,(os.cpu_count() or 2)-1))
    h.add_argument('--train-steps',type=int);h.add_argument('--smoke',action='store_true')
    t.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);t.add_argument('--proposals',type=int)
    t.add_argument('--batch-size',type=int,default=256);t.add_argument('--smoke',action='store_true')
    t.add_argument('--path',choices=['sync','geometry_lead2','geometry_lead3'],help='clock override (geolead-1)')
    z.add_argument('--check-a',type=Path);z.add_argument('--check-b',type=Path);z.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    {'heads':cmd_heads,'trajectories':cmd_trajectories,'analyze':cmd_analyze,
     'analyze-clock':cmd_analyze_clock}[a.cmd](a)


if __name__=='__main__':main()
