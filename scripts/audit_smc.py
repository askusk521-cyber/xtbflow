"""Independent coverage audit for SMC artifacts (no recomputation of chemistry)."""
import argparse
import json
from pathlib import Path
import sys
import hashlib

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_smc import rows
from xtbflow.v1.smc import selection_plan


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out
    load=lambda name:json.loads((out/name).read_text())
    cfg=load('s0_manifest.json');g1=load('g1_manifest.json');c1=load('c1_manifest.json')
    plans=load('plans.json');cp=load('checkpoint_scores.json');none=load('none_decoded.json')
    parents=sorted({r['parent_id'] for r in none});seeds=(0,1,2);steps=(30,35,40,45)
    assert len(parents)==62 and len(none)==5952 and len(cp)==23808 and len(plans)==2232
    assert all(not m['dirty'] and not m['smoke'] for m in (cfg,g1,c1))
    assert cfg['n']==558 and cfg['gate_passed'] and cfg['raw_max_error']<=1e-6
    assert abs(cfg['raw_rho']['estimate']-.7491015704725383)<1e-9
    assert c1['n']==5952 and c1['event_status_mismatches']==0 and c1['raw_max_error']<=1e-6
    pgpath=next(k for k in c1['inputs'] if k.endswith('check_b_trajectories.jsonl'))
    pg={(r['parent_id'],r['seed'],r['proposal']):r for r in rows(pgpath)}
    assert len(pg)==5952
    for r in none:
        old=pg[r['parent_id'],r['seed'],r['proposal']]
        assert r['channel_id']==old['events'][-1] and r['proxy_status']==old['proxy_status']
        assert r['event_utility']==old['event_utility'] and r['hits_best_event']==old['hits_best_event']
    lookup={(r['parent_id'],r['seed'],r['proposal'],r['step']):r['score'] for r in cp}
    assert len(lookup)==23808
    paired_raw_failures=0
    for (pid,seed,j,k),score in lookup.items():
        target=pg[pid,seed,j]['xtb_delta_kcal'][k//5]
        if score['raw_barrier'] is None:
            assert not np.isfinite(target);paired_raw_failures+=1
        else:assert abs(score['raw_barrier']-target)<=1e-6
    for plan in plans:
        pid,seed,k=plan['parent_id'],plan['seed'],plan['step'];scorer=plan['arm'].split('_')[0]
        scores=[lookup[pid,seed,j,k] for j in range(32)]
        if scorer=='raw':scores=[dict(s,status='ok' if s['raw_barrier'] is not None else 'failed',barrier=s['raw_barrier']) for s in scores]
        s,c=selection_plan(pid,seed,scores,k,scorer,curvature_layers=cfg['curvature_layers'])
        assert s==plan['survivors'] and [list(v) for v in c]==plan['clones']
    shapes=[]
    for name in g1['batches']:
        d=torch.load(out/name,map_location='cpu',weights_only=False)
        assert d['b_trace'].shape[0]==d['x_trace'].shape[0]==5
        assert torch.equal(d['b_trace'][-1],d['b_hat'][-1])
        assert torch.equal(d['x_trace'][-1],d['x_hat'][-1])
        assert len(d['pairs']) in (256,192)
        for start in range(0,len(d['pairs']),32):assert [j for i,j in d['pairs'][start:start+32]]==list(range(32))
        shapes.append(list(d['b_trace'].shape))
    result=dict(s0_passed=True,c1_independently_reproduced=True,paired_raw_failures=paired_raw_failures,
                population_count=186,checkpoint_candidates=23808,plans_verified=len(plans),g1_batches=len(shapes))
    if (out/'endpoint_rows.json').exists():
        c2=load('c2_manifest.json');g2=load('g2_manifest.json');ep=load('endpoint_rows.json')
        assert c2['gate_passed'] and not c2['dirty'] and not g2['dirty']
        assert len(ep)==77376 and len(g2['files'])==288
        assert len(c2['survivor_checks'])==12
        assert all(s['n']==2976 and s['agreement']>=.999 for s in c2['survivor_checks'].values())
        groups={}
        for r in ep:groups.setdefault((r['arm'],r['parent_id'],r['seed']),[]).append(r)
        assert len(groups)==2418 and all(len(v)==32 for v in groups.values())
        for (arm,pid,seed),rr in groups.items():
            assert {r['proposal'] for r in rr}==set(range(32))
            if arm!='none':
                assert sum(r['is_clone'] for r in rr)==16
                ss=[r for r in rr if not r['is_clone']];cc=[r for r in rr if r['is_clone']]
                assert {r['proposal'] for r in ss}=={r['ancestor'] for r in cc}
        result.update(endpoint_candidates=len(ep),arms=13,complete_populations=len(groups),c2_passed=True)
    text=json.dumps(result,sort_keys=True,indent=2)+'\n';(out/'coverage_audit.json').write_text(text);print(text)


if __name__=='__main__':main()
