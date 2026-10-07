"""Offline score calibration, cost prefixes and paired geometry-drift diagnostics."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import time

import numpy as np
from rdkit import RDLogger

from xtbflow.v1.costs import BUDGETS,weights_for_atoms
from xtbflow.v1.data import file_hash,write_json
from xtbflow.v1.formal import evaluation_parents
from xtbflow.v1.development import score_calibration
from xtbflow.v1.metrics import cluster_summary,prefix_auc
from xtbflow.v1.proxy import CatalogueMatcher


def audit_ledger(row,weights):
    spent=0.;positions={0.};seen=set()
    for op in row['operations']:
        cost=weights[op['operation']]*op['count'];spent+=cost;positions.add(round(spent,7))
        if abs(cost-op['cost_units'])>1e-7 or abs(spent-op['cumulative_units'])>1e-6:
            raise ValueError('ledger cost differs from common-batch cost table')
    if spent>row['cap_units']+1e-6 or abs(spent-row['spent_units'])>1e-6:
        raise ValueError('ledger overspend or missing operation')
    for attempt in row['attempts']:
        if attempt['attempt_id'] in seen:raise ValueError('attempt terminated twice')
        seen.add(attempt['attempt_id'])
        if round(attempt['completion_units'],7) not in positions:
            raise ValueError('attempt completion not at a logical operation boundary')
    if len(seen)!=row['n_proposals']:raise ValueError('missing attempts')
    complete={r['candidate_id']:r for r in row['attempts'] if r['candidate_id'] is not None}
    if set(complete)!={r['candidate_id'] for r in row['candidates']}:
        raise ValueError('candidate/attempt inventory mismatch')
    for r in row['candidates']:
        if r['completion_units']!=complete[r['candidate_id']]['completion_units']:
            raise ValueError('candidate completion time changed')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--catalogue',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--frozen',type=Path)
    a=ap.parse_args();started=time.monotonic();RDLogger.DisableLog('rdApp.error')
    manifest=json.loads((a.run/'manifest.json').read_text());source=a.run/'streams.jsonl'
    if not manifest['complete'] or file_hash(source)!=manifest['streams_sha256']:
        raise ValueError('unsealed stream')
    split=json.loads((a.catalogue/'split_manifest.json').read_text())
    dev,query_name=evaluation_parents(split,manifest,a.frozen)
    if any(m['split_hash']!=split['split_hash'] for m in manifest['models'].values()):
        raise ValueError('generator split mismatch')
    if manifest['query_source_sha256']!=file_hash(a.catalogue/query_name):
        raise ValueError('query source changed')
    parents={p['query_id']:p for p in json.loads((a.catalogue/'parent_catalog.json').read_text()) if p['parent_id'] in dev}
    refs=defaultdict(list)
    for line in (a.catalogue/'reference_catalog.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['parent_id'] in dev:refs[r['parent_id']].append(r)
    matchers={q:CatalogueMatcher(p,refs[p['parent_id']]) for q,p in parents.items()}
    parent_rows=defaultdict(list);calibration=defaultdict(list);drift=defaultdict(list)
    failures=defaultdict(Counter);evaluated=[];seen=set()
    for line in source.read_text().splitlines():
        row=json.loads(line);qid=row['query_id'];arm=row['arm'];key=(qid,arm)
        if key in seen or qid not in parents or arm not in manifest['arms']:
            raise ValueError('duplicate or unknown arm/query')
        seen.add(key);p=parents[qid];matcher=matchers[qid]
        if row['training_seed']!=manifest['training_seed'] or row['sampling_seed']!=0:
            raise ValueError('seed mismatch')
        audit_ledger(row,weights_for_atoms(manifest['costs'],len(p['atomic_numbers'])))
        failures[arm].update(r['status'] for r in row['attempts'])
        for r in row['candidates']:
            r.update(matcher.match(r['b_dec'],r['x'],r['is_fallback'],r['decode_status']))
            r.update(parent_id=p['parent_id'],split_group=p['split_group'],arm=arm,
                     training_seed=row['training_seed'],sampling_seed=0)
            for snap in r['snapshots']:
                calibration[arm,snap.get('kind','X'),snap['requested_t']].append(dict(r,**snap))
            shadow=r.pop('shadow',None)
            if shadow is not None:
                shadow.update(matcher.match(shadow['b_dec'],shadow['x'],shadow['is_fallback'],shadow['decode_status']))
                # Same decoded event; each endpoint is matched to references using one
                # common event/geometry mapping as enforced by CatalogueMatcher.
                if (r['predicted_channel_id'] is not None and
                    r['predicted_channel_id']==shadow['predicted_channel_id'] and
                    r['best_event_rmsd_angstrom'] is not None and shadow['best_event_rmsd_angstrom'] is not None):
                    drift[arm].append(dict(parent_id=p['parent_id'],split_group=p['split_group'],
                        candidate_id=r['candidate_id'],rmsd_difference=r['best_event_rmsd_angstrom']-shadow['best_event_rmsd_angstrom'],
                        joint_match_difference=int(r['proxy_status']=='PROXY_MATCH')-int(shadow['proxy_status']=='PROXY_MATCH')))
                r['shadow_match']={k:v for k,v in shadow.items() if k not in ('x','b_dec','snapshots','guidance','shadow')}
            evaluated.append(r)
        hits=[];event_hits=[];counts=[];valid=[];unique=[];any_ref=[]
        for budget in BUDGETS:
            rr=[r for r in row['candidates'] if r['completion_units']<=50*budget+1e-8]
            hits.append(int(any(r['hits_best_reference'] for r in rr)))
            event_hits.append(int(any(r['hits_best_event'] for r in rr)))
            any_ref.append(int(any(r['proxy_status']=='PROXY_MATCH' for r in rr)))
            counts.append(len(rr));valid.append(sum(r['proxy_status']!='INVALID_OUTPUT' for r in rr))
            unique.append(len({r['predicted_channel_id'] for r in rr if r['predicted_channel_id'] is not None}))
        budget16=[r for r in row['candidates'] if r['completion_units']<=800+1e-8]
        parent_rows[arm].append(dict(parent_id=p['parent_id'],split_group=p['split_group'],query_id=qid,
                                    n_proposals=row['n_proposals'],
                                    best_event_count=sum(bool(r['hits_best_event']) for r in row['candidates']),
                                    best_reference_count=sum(bool(r['hits_best_reference']) for r in row['candidates']),
                                    v1b_budget16=dict(candidate_ids=[r['candidate_id'] for r in budget16],
                                        attempt_ids=[t['attempt_id'] for t in row['attempts'] if t['completion_units']<=800+1e-8],
                                        best_event_candidate_ids=[r['candidate_id'] for r in budget16 if r['hits_best_event']],
                                        proxy_best_reference_hit=bool(hits[2])),
                                    hits=hits,event_hits=event_hits,
                                    any_reference_hits=any_ref,auc=float(prefix_auc(hits)),event_auc=float(prefix_auc(event_hits)),
                                    completed_candidates=counts,valid_candidates=valid,unique_events=unique,
                                    spent_units=row['spent_units']))
    if seen!={(q,arm) for q in parents for arm in manifest['arms']}:raise ValueError('incomplete streams')
    summaries={}
    for arm,rr in parent_rows.items():
        summaries[arm]=dict(parent_rows=rr,**{key:np.mean([r[key] for r in rr],axis=0).tolist()
            for key in ('hits','event_hits','any_reference_hits','auc','event_auc','completed_candidates','valid_candidates','unique_events')})
    cal={};event_cal={}
    for (arm,kind,t),rr in calibration.items():
        item=score_calibration(rr,'score','mean_kcal')
        matched=[r for r in rr if r['proxy_status']=='PROXY_MATCH']
        item.update(support_rate=float(np.mean([r['supported'] for r in rr])),
                    gradient_norm_quantiles=np.quantile([r['gradient_norm'] for r in rr],[0,.5,.95,1]).tolist(),
                    score_quantiles=np.quantile([r['score'] for r in rr],[.01,.5,.99]).tolist(),
                    member_mae_kcal=None if not matched else [float(np.mean([
                        abs(r['member_kcal'][i]-r['catalog_barrier_kcal']) for r in matched])) for i in range(3)])
        for key in ('barrier_gradient_norm','uncertainty_gradient_norm'):
            if all(key in r for r in rr):item[key+'_median']=float(np.median([r[key] for r in rr]))
        (cal if kind=='X' else event_cal).setdefault(arm,{})[str(t)]=item
    drift_report={}
    for arm in ('A2','B1'):
        if arm not in manifest['arms']:continue
        rr=drift[arm];median=None if not rr else float(np.median([r['rmsd_difference'] for r in rr]))
        delta=None if not rr else float(np.mean([r['joint_match_difference'] for r in rr]))
        drift_report[arm]=dict(n_pairs=len(rr),median_rmsd_increase=median,joint_match_delta=delta,
                              diagnostic_pass=None if not rr else median<=.05 and delta>=-.05,pairs=rr)
    primary=None
    if 'A2' in parent_rows and 'B1' in parent_rows:
        a2={r['parent_id']:r for r in parent_rows['A2']};b1={r['parent_id']:r for r in parent_rows['B1']}
        primary=cluster_summary([b1[p]['auc']-a2[p]['auc'] for p in sorted(a2)],
                                 [a2[p]['split_group'] for p in sorted(a2)])
    screen=manifest.get('split_name','development')=='screen'
    report=dict(status='SCREEN_STREAM_AUDIT_PASS' if screen else 'DEVELOPMENT_STREAM_AUDIT_PASS',
                config=manifest['config'],summaries=summaries,purpose=manifest.get('purpose','development'),
                split_name=manifest.get('split_name','development'),freeze_sha256=manifest.get('freeze_sha256'),
                training_seed=manifest['training_seed'],
                calibration=cal,event_calibration=event_cal,drift=drift_report,
                attempt_status_counts={k:dict(v) for k,v in failures.items()},
                development_primary=primary,source_sha256=manifest['streams_sha256'],split_hash=split['split_hash'],
                matcher_cpu_wall_s=time.monotonic()-started,
                interpretation=('Sealed screen stream; inference only in the single frozen gate analysis.' if screen else
                                'Development calibration and selection only. No formal method comparison or GO decision.'))
    write_json(a.out,report)
    with a.out.with_suffix('.candidates.jsonl').open('w') as f:
        for r in evaluated:
            r.pop('x');f.write(json.dumps(r,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],auc={k:v['auc'] for k,v in summaries.items()},
        drift={k:{kk:vv for kk,vv in v.items() if kk!='pairs'} for k,v in drift_report.items()},
        cpu_wall_s=report['matcher_cpu_wall_s']),indent=2))


if __name__=='__main__':main()
