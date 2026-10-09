"""Deterministic T1b analysis; requires complete screen acquisition, no QC calls."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import gzip
import json
from pathlib import Path
import sys

import numpy as np
from xtbflow.v1.review_enum_metrics import GRID, random_recall, generator_curve, ranked_curve, paired_auc
sys.path.insert(0,str(Path(__file__).resolve().parent))
from followup_enum_gate import b2f2_mask


def score_orders(channels, sn, delta, tau):
    """Finite rank scores preserve exactly the frozen lexicographic orders."""
    n=len(channels)
    tie=lambda i:sha256(channels[i].encode()).hexdigest()
    success=lambda i:delta[i] is not None and np.isfinite(delta[i])
    keys={
        'S_N2':lambda i:(float(sn[i]),tie(i)),
        'S_thermo':lambda i:(0,float(delta[i]),tie(i)) if success(i) else (1,float(sn[i]),tie(i)),
        'S_N_thermo':lambda i:(0 if success(i) and delta[i]<=tau else 1,float(sn[i]),tie(i))}
    if not np.isfinite(sn).all():raise ValueError('Archived S_N is nonfinite')
    out={}
    for name,key in keys.items():
        scores=np.empty(n,dtype=float)
        for rank,i in enumerate(sorted(range(n),key=key)):scores[i]=rank
        out[name]=scores
    return out


def reading(comparison):
    lo,hi=comparison['summary']['ci_two95']
    return 'ENUMERATION_BEATS_GENERATION' if lo>0 else 'GENERATION_BEATS_ENUMERATION' if hi<0 else 'INCONCLUSIVE'


def analyze(root, enumdir, physical, tau):
    manifest=json.loads((physical/'manifest.json').read_text())
    if not manifest.get('complete'):raise RuntimeError('Screen acquisition incomplete')
    parents={p['parent_id']:p for p in json.loads((root/'data/parent_catalog.json').read_text())}
    ids=sorted(r['parent_id'] for r in manifest['parent_results'])
    if len(ids)!=342 or len(set(ids))!=342:raise ValueError('Full screen coverage required')
    groups={p:parents[p]['split_group'] for p in ids}
    gen=defaultdict(lambda:defaultdict(list));raw=defaultdict(lambda:defaultdict(list))
    stream=Path('/home/lhshen/xtbflow-runs/v1a-r-20261008/extract/streams.jsonl.gz')
    with gzip.open(stream,'rt') as handle:
        for line in handle:
            r=json.loads(line)
            if r['parent_id'] in groups:
                gen[r['arm']][r['parent_id']].append(generator_curve(r['channel'],parents[r['parent_id']]['best_channel_ids']))
                raw[r['arm']][r['parent_id']].append(r)
    curves={s:{} for s in ('S_rand2','S_N2','S_thermo','S_N_thermo')}
    rows=[];failures=Counter();total=success=filtered=0;event_times=[]
    for pid in ids:
        path=physical/(pid+'.jsonl')
        expected=next(r['output_sha256'] for r in manifest['parent_results'] if r['parent_id']==pid)
        if sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Physical output hash mismatch')
        acquired=[json.loads(s) for s in path.read_text().splitlines()]
        by={r['channel_id']:r for r in acquired}
        if len(by)!=len(acquired):raise ValueError('Duplicate physical channel')
        with np.load(enumdir/(pid+'.npz')) as data,np.load(enumdir/(pid+'.scores.npz')) as scores:
            mask=b2f2_mask(data['b'],parents[pid]['b_r'])
            channels=data['channels'][mask].tolist();sn=scores['S_N'][mask]
        if set(channels)!=set(by):raise ValueError('b2f2 physical coverage mismatch')
        delta=[by[c]['delta_e_kcal'] for c in channels]
        best=set(parents[pid]['best_channel_ids']);n=len(channels);m=len(set(channels)&best)
        curves['S_rand2'][pid]={k:random_recall(n,m,k) for k in GRID}
        for name,rank in score_orders(channels,sn,delta,tau).items():curves[name][pid]=ranked_curve(channels,rank,best)
        meta=json.loads((enumdir/(pid+'.json')).read_text())
        rows.append(dict(parent_id=pid,n_b2f2=n,best_covered=bool(m),
                         catalogue_coverage=len(set(channels)&set(meta['catalogue_channels']))/len(meta['catalogue_channels'])))
        for c in channels:
            r=by[c];total+=1;event_times.append(r['product']['wall_s'])
            if r['delta_e_kcal'] is None:failures['product:'+r['product']['status']+';reactant:'+r['reactant']['status']]+=1
            else:success+=1;filtered+=r['delta_e_kcal']>tau
    comparisons={}
    pairs=[('S_N_thermo',a) for a in ('B0','A0','A1','A2','B1')]+[('S_thermo','B0'),('S_N2','B0')]
    for scorer,arm in pairs:
        result=paired_auc(curves[scorer],gen[arm],groups);result['reading']=reading(result)
        comparisons[scorer+'-'+arm]=result
    thermo_help=paired_auc(curves['S_N_thermo'],{p:[c] for p,c in curves['S_N2'].items()},groups)
    thermo_help['reading']='THERMO_FILTER_HELPS' if thermo_help['summary']['ci_two95'][0]>0 else 'NO_CONFIRMED_THERMO_GAIN'
    comparisons['S_N_thermo-S_N2']=thermo_help
    covered={r['parent_id'] for r in rows if r['best_covered']}
    conditional=paired_auc({p:curves['S_N_thermo'][p] for p in covered},gen['B0'],groups)
    conditional['reading']=reading(conditional)
    rates={s:{str(k):float(np.mean([curve[k] for curve in by.values()])) for k in GRID} for s,by in curves.items()}
    for arm,by in gen.items():
        rates[arm]={str(k):float(np.mean([np.mean([r[k] for r in by[p] if r[k] is not None]) for p in ids if any(r[k] is not None for r in by[p])])) for k in GRID}
    outside=set(ids)-covered
    hits={arm:sorted(p for p in outside if any(any(r['event']) for r in raw[arm][p])) for arm in sorted(raw)}
    counts=[r['n_b2f2'] for r in rows]
    return dict(exploratory=True,primary='S_N_thermo-B0',n_parents=len(ids),tau_kcal=tau,
                comparisons=comparisons,conditional_best_in_b2f2=conditional,rates=rates,parents=rows,
                best_coverage=len(covered)/len(ids),catalogue_coverage_parent_mean=float(np.mean([r['catalogue_coverage'] for r in rows])),
                n_b2f2=dict(total=sum(counts),median=float(np.median(counts)),p95=float(np.percentile(counts,95))),
                outside_best_parents=sorted(outside),generator_hit_outside_best=hits,
                generator_hit_outside_best_count={a:len(v) for a,v in hits.items()},
                pipeline=dict(total=total,successful=success,success_rate=success/total,failure_reasons=dict(failures),
                              filtered_successes=filtered,fraction_filtered_of_all=filtered/total,
                              fraction_filtered_of_success=filtered/success if success else None,
                              mean_product_wall_s=float(np.mean(event_times))),
                source_acquisition=manifest['source_commit'],limitations=['Exploratory proxy event recall, not TS certification.',
                    'Frozen full screen; no official M0 val/test.', 'Failures ranked behind successful thermochemistry as frozen.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--enumdir',type=Path,required=True)
    p.add_argument('--physical',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    tau=json.loads(Path('configs/explore/enum_thermo_tau.json').read_text())['tau_kcal']
    result=analyze(a.root,a.enumdir,a.physical,tau)
    if a.out.exists():raise FileExistsError(a.out)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
