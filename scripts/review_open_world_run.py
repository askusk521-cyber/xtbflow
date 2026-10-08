"""Run the frozen positive control first, then other xTB chains if it passes."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from xtbflow.v1.data import write_json
from xtbflow.v1.xtb_chain import certification_chain


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/home/lhshen/xtbflow-runs/v1a-20261007'))
    p.add_argument('--config',type=Path,default=Path('configs/review/open_world_o1.json'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=7000)
    a=p.parse_args()
    cfg=json.loads(a.config.read_text())
    parents={r['parent_id']:r for r in json.loads((a.root/'data/parent_catalog.json').read_text())}
    ids={r['candidate_id'] for rows in cfg['groups'].values() for r in rows if 'candidate_id' in r}
    candidates={}
    with (a.root/'screen/efficiency_s0.candidates.jsonl').open() as handle:
        for line in handle:
            r=json.loads(line)
            if r['candidate_id'] in ids:candidates[r['candidate_id']]=r
    geometry={}
    with (a.root/'screen/efficiency_s0/streams.jsonl').open() as handle:
        for line in handle:
            r=json.loads(line)
            for c in r.get('candidates',[r]):
                if c.get('candidate_id') in ids:geometry[c['candidate_id']]=c['x']
    refs={}
    with (a.root/'data/reference_catalog.jsonl').open() as handle:
        for line in handle:
            r=json.loads(line);refs[r['reference_id']]=r
    start=time.monotonic()
    for group in ('PROXY_MATCH','KNOWN_EVENT_GEOMETRY_MISS','BASELINE','OUTSIDE_CATALOGUE_EVENT'):
        verdicts=[]
        for row in cfg['groups'][group]:
            ident=row.get('candidate_id',row.get('reference_id'))
            work=a.out/group/ident
            if (work/'verdict.json').exists():
                verdicts.append(json.loads((work/'verdict.json').read_text()))
                continue
            if time.monotonic()-start>a.max_seconds:
                print('CHUNK_TIME_LIMIT',flush=True);return
            parent=parents[row['parent_id']]
            if group=='BASELINE':
                r=refs[ident];x=r['x_ts'];bp=r['b_p']
            else:
                x=geometry[ident];bp=candidates[ident]['b_dec']
            result=certification_chain(parent['atomic_numbers'],np.array(x),parent['b_r'],parent['permutations'],bp,work)
            result.update(group=group,id=ident,parent_id=row['parent_id'])
            write_json(work/'verdict.json',result)
            verdicts.append(result)
            print(json.dumps({k:result[k] for k in ('id','group','status','wall_seconds','gradient_calls')}),flush=True)
        if group=='PROXY_MATCH':
            passed=sum(v['status']=='STRICT_JOINT_GRAPH_VALID' for v in verdicts)
            write_json(a.out/'positive_control_gate.json',dict(certified=passed,total=len(verdicts),passed=passed/len(verdicts)>=0.5))
            if passed/len(verdicts)<0.5:
                print('POSITIVE_CONTROL_GATE_FAILED; OUTSIDE NOT RUN OR INTERPRETED',flush=True);return


if __name__=='__main__':main()
