"""V1b report: Stage-A stratified HT and the single Stage-B decision (guide 9, 10, 15.3)."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

from xtbflow.v1.audit_statistics import (bundle_interval,delta_cert_decision,paired_finite_population_ci,
                                         stratified_total)
from xtbflow.v1.data import write_json

KCAL=627.509474


def jsonl(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]


def verdict(run,cid):
    f=run/'work'/'candidates'/sha256(cid.encode()).hexdigest()[:20]/'verdict.json'
    return json.loads(f.read_text()) if f.exists() else None


def labels(v,T,anchor_e):
    """(ts_optfreq, strict_joint, strict_low, best_cert) intervals for one verdict."""
    if v is None:return [(0,1)]*4
    opt=(v['ts_optfreq_success'],)*2 if 'ts_optfreq_success' in v else (0,1)
    sj=v.get('strict_joint_graph');joint=(0,1) if sj is None else (sj,sj)
    L=None if anchor_e is None or 'ts' not in v or 'energy_hartree' not in v['ts'] else (v['ts']['energy_hartree']-anchor_e)*KCAL
    def window(w):
        if joint==(0,0):return (0,0)
        if v.get('strict_status')=='ENERGY_EXCLUDED_FOR_BEST':return (0,0) if w==1. else (0,1)
        if T is None or L is None or joint!=(1,1):return (0,1)
        return (1,1) if L<=T+w+1e-6 else (0,0)
    return opt,joint,window(2.),window(1.)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--v1a-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    import sys;sys.path.insert(0,str(Path(__file__).parent))
    from v1b_run import Source,threshold
    src=Source(a.run,a.v1a_root);manifest=src.manifest
    def T_of(pid):
        T,status=threshold(a.run/'work',pid,src);f=a.run/'work'/'references'/pid/'anchor'/'verdict.json'
        return T,status,(json.loads(f.read_text())['energy_hartree'] if f.exists() and T is not None else None)
    # Stage A
    sample=jsonl(a.run/'sample_A.jsonl');by=defaultdict(list)
    for r in sample:by[r['stratum']].append(r)
    stage_a={}
    for h,rs in sorted(by.items()):
        N,n=rs[0]['N_h'],rs[0]['n_h'];vals=defaultdict(lambda:([],[]))
        for r in rs:
            if r['proxy_class']=='INVALID_OUTPUT':
                ls=[(0,0)]*4
            else:
                T,_,ae=T_of(r['parent_id']);ls=labels(verdict(a.run,r['unit_id']),T,ae)
            for name,(lo,hi) in zip(('ts_optfreq_success','strict_joint_graph','strict_low_graph','best_cert_hit'),ls):
                vals[name][0].append(lo);vals[name][1].append(hi)
        stage_a[h]={name:dict(lower=stratified_total(N,n,lo),upper=stratified_total(N,n,hi))
                    for name,(lo,hi) in vals.items()}
    # Stage B
    rows=[];detail=[]
    for r in jsonl(a.run/'sample_B_parents.jsonl'):
        pid=r['unit_id'];T,status,ae=T_of(pid);arms={}
        for arm in ('A2','B1'):
            cands=src.bundles[pid,arm]['best_event_candidate_ids']
            iv=[labels(verdict(a.run,c),T,ae)[3] for c in cands]
            arms[arm]=bundle_interval(iv,len(cands))
        rows.append(dict(stratum=r['stratum'],N_h=r['N_h'],n_h=r['n_h'],pi=r['pi'],A2=arms['A2'],B1=arms['B1']))
        detail.append(dict(parent_id=pid,stratum=r['stratum'],threshold_status=status,T_H_kcal=T,
                           proxy_A2=r['proxy_A2'],proxy_B1=r['proxy_B1'],cert_A2=arms['A2'],cert_B1=arms['B1']))
    M=manifest['summary']['M'];complete=all(x['A2'][0]==x['A2'][1] and x['B1'][0]==x['B1'][1] for x in rows)
    decision=delta_cert_decision(rows,M,manifest['delta_proxy'],complete)
    try:sensitivity=paired_finite_population_ci(rows,M,alpha=.10)
    except ValueError as e:sensitivity=dict(error=str(e))
    precision={}
    for arm in ('A2','B1'):
        pos=[d for d in detail if d['proxy_'+arm]==1];neg=[d for d in detail if d['proxy_'+arm]==0]
        precision[arm]=dict(proxy_positive_sampled=len(pos),certified_among_positive=[sum(d['cert_'+arm][0] for d in pos),sum(d['cert_'+arm][1] for d in pos)],
                            proxy_negative_sampled=len(neg),certified_among_negative=[sum(d['cert_'+arm][0] for d in neg),sum(d['cert_'+arm][1] for d in neg)],
                            note='unweighted sampled counts; domain HT ratios use the stratum weights in parent_detail')
    report=dict(schema='xtbflow-v1b-report/1',decision=decision,delta_proxy=manifest['delta_proxy'],M=M,
                hypergeometric_sensitivity=sensitivity,stage_A=stage_a,stage_B_parents=detail,
                arm_precision=precision,protocol=manifest.get('protocol'),
                scope='Fixed model, budget 16, training seed 0; finite-population comparison only.')
    write_json(a.out,report)
    print(json.dumps(dict(decision=decision,sensitivity=sensitivity),indent=1))


if __name__=='__main__':main()
