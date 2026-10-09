"""Deterministic frozen enum-versus-generation analysis from archived outputs."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import math
import numpy as np
from xtbflow.v1.data import write_json
from xtbflow.v1.review_enum_metrics import GRID,random_recall,generator_curve,ranked_curve,paired_auc


def analyze(root,enumdir):
    parents={r['parent_id']:r for r in json.loads((root/'data/parent_catalog.json').read_text())}
    split=json.loads((root/'data/split_manifest.json').read_text())
    ids=sorted(json.loads((root/'formal/freeze.json').read_text())['parent_ids']);groups={p:parents[p]['split_group'] for p in ids}
    assert len(ids)==342
    gen=defaultdict(lambda:defaultdict(list));rawgen=defaultdict(lambda:defaultdict(list))
    with gzip.open('/home/lhshen/xtbflow-runs/v1a-r-20261008/extract/streams.jsonl.gz','rt') as handle:
        for line in handle:
            r=json.loads(line);pid=r['parent_id']
            if pid not in groups:continue
            gen[r['arm']][pid].append(generator_curve(r['channel'],parents[pid]['best_channel_ids']))
            rawgen[r['arm']][pid].append(r)
    curves={s:{} for s in ('S_rand','S_E','S_N')};metadata=[]
    for pid in ids:
        meta=json.loads((enumdir/(pid+'.json')).read_text());data=np.load(enumdir/(pid+'.npz'));scores=np.load(enumdir/(pid+'.scores.npz'))
        channels=data['channels'].tolist();best=parents[pid]['best_channel_ids'];n=len(channels);m=len(set(channels)&set(best))
        curves['S_rand'][pid]={k:random_recall(n,m,k) for k in GRID}
        for k in GRID:
            sample=min(k,n)
            independent=0. if n==0 else 1-(math.comb(n-m,sample) if sample<=n-m else 0)/math.comb(n,sample)
            assert abs(curves['S_rand'][pid][k]-independent)<1e-12
        for scorer in ('S_E','S_N'):curves[scorer][pid]=ranked_curve(channels,scores[scorer],best)
        metadata.append(meta)
    comparisons={}
    for scorer,arm in [('S_N','B0'),('S_N','A1'),('S_N','B1'),('S_N','A2'),('S_N','A0'),('S_E','B0'),('S_rand','B0')]:
        result=paired_auc(curves[scorer],gen[arm],groups)
        lo,hi=result['summary']['ci_two95']
        result['reading']='ENUMERATION_BEATS_GENERATION' if lo>0 else 'GENERATION_BEATS_ENUMERATION' if hi<0 else 'INCONCLUSIVE'
        comparisons[scorer+'-'+arm]=result
    rates={}
    for scorer,by in curves.items():rates[scorer]={str(k):float(np.mean([by[p][k] for p in ids])) for k in GRID}
    for arm,by in gen.items():
        rates[arm]={}
        for k in GRID:
            means=[np.mean([r[k] for r in by[p] if r[k] is not None]) for p in ids if any(r[k] is not None for r in by[p])]
            rates[arm][str(k)]=float(np.mean(means)) if means else None
    outside=[r['parent_id'] for r in metadata if not r['covered_best']]
    outside_hits={arm:sorted(p for p in outside if any(any(r['event']) for r in rawgen[arm][p])) for arm in sorted(rawgen)}
    n=np.array([r['n_enum'] for r in metadata])
    return dict(exploratory=True,random_analytic_check_passed=True,n_parents=len(ids),comparisons=comparisons,rates=rates,
        n_enum=dict(median=float(np.median(n)),p95=float(np.quantile(n,.95)),total=int(n.sum())),
        best_coverage=float(np.mean([bool(r['covered_best']) for r in metadata])),
        channel_coverage_parent_mean=float(np.mean([len(r['covered_channels'])/len(r['catalogue_channels']) for r in metadata])),
        outside_best_parents=outside,generator_hit_outside_best=outside_hits,parents=metadata)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--devdir',type=Path);p.add_argument('--root',type=Path,default=Path('/home/lhshen/xtbflow-runs/v1a-20261007'))
    p.add_argument('--enumdir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=analyze(a.root,a.enumdir)
    if a.devdir:
        ids=sorted(json.loads((a.root/'data/split_manifest.json').read_text())['development']['parent_ids'])
        rows=[json.loads((a.devdir/(pid+'.json')).read_text()) for pid in ids]
        counts=np.array([r['n_enum'] for r in rows])
        dev_curves={s:[] for s in ('S_rand','S_E','S_N')}
        for pid,meta in zip(ids,rows):
            data=np.load(a.devdir/(pid+'.npz'));scores=np.load(a.devdir/(pid+'.scores.npz'))
            channels=data['channels'].tolist();best=meta['best_channels'];n=len(channels);m=len(set(channels)&set(best))
            dev_curves['S_rand'].append({k:random_recall(n,m,k) for k in GRID})
            for scorer in ('S_E','S_N'):dev_curves[scorer].append(ranked_curve(channels,scores[scorer],best))
        result['development']=dict(rates={s:{str(k):float(np.mean([row[k] for row in curves])) for k in GRID} for s,curves in dev_curves.items()},n_parents=len(rows),best_coverage=float(np.mean([bool(r['covered_best']) for r in rows])),
            channel_coverage_parent_mean=float(np.mean([len(r['covered_channels'])/len(r['catalogue_channels']) for r in rows])),
            n_enum=dict(median=float(np.median(counts)),p95=float(np.quantile(counts,.95)),total=int(counts.sum())),parents=rows)
    write_json(a.out,result)
