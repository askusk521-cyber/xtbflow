"""Deterministic frozen xTB-chain summary and Wilson intervals."""
import argparse
from collections import Counter
import json
from pathlib import Path
import math
import numpy as np
from xtbflow.v1.data import write_json


def wilson(success,total):
    if total==0:return None
    z=1.959963984540054;p=success/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return dict(success=success,total=total,rate=p,ci_two95=[center-half,center+half])


def analyze(config,root):
    cfg=json.loads(config.read_text());out=dict(exploratory=True,groups={},complete=True)
    allrows={}
    for group,selected in sorted(cfg['groups'].items()):
        rows=[]
        for sample in selected:
            ident=sample.get('candidate_id',sample.get('reference_id'))
            path=root/group/ident/'verdict.json'
            if not path.exists():out['complete']=False;continue
            r=json.loads(path.read_text());rows.append(dict(r,id=ident,parent_id=sample['parent_id']))
        allrows[group]=rows
        out['groups'][group]=dict(planned=len(selected),completed=len(rows),
            statuses=dict(sorted(Counter(r['status'] for r in rows).items())),
            certification=wilson(sum(r['status']=='STRICT_JOINT_GRAPH_VALID' for r in rows),len(rows)),
            wall_seconds=sum(r['wall_seconds'] for r in rows),gradient_calls=sum(r['gradient_calls'] for r in rows))
    positive=out['groups']['PROXY_MATCH']
    out['positive_control_passed']=positive['completed']==positive['planned'] and positive['certification']['rate']>=.5
    deltas=[];excluded=[]
    baseline={r['parent_id']:r for r in allrows['BASELINE']}
    if out['positive_control_passed']:
        for r in allrows['OUTSIDE_CATALOGUE_EVENT']:
            if r['status']!='STRICT_JOINT_GRAPH_VALID':continue
            b=baseline.get(r['parent_id'])
            if b is None or b['status']!='STRICT_JOINT_GRAPH_VALID':excluded.append(r['parent_id']);continue
            deltas.append(dict(parent_id=r['parent_id'],candidate_id=r['id'],
                delta_ea_kcal=(r['ts']['energy_hartree']-b['ts']['energy_hartree'])*627.509474))
    out['delta_ea']=dict(rows=deltas,excluded_baseline_parents=sorted(excluded),
        below_zero=wilson(sum(r['delta_ea_kcal']<0 for r in deltas),len(deltas)),
        at_most_plus5=wilson(sum(r['delta_ea_kcal']<=5 for r in deltas),len(deltas)))
    out['verdicts']=[dict(group=g,id=r['id'],parent_id=r['parent_id'],status=r['status'],
        wall_seconds=r['wall_seconds'],gradient_calls=r['gradient_calls']) for g in sorted(allrows) for r in allrows[g]]
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/review/open_world_o1.json'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    write_json(a.out,analyze(a.config,a.root))
