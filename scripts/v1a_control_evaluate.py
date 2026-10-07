"""Offline matched continuous controls, with all planned parent/seed pairs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
from rdkit import RDLogger

from xtbflow.v1.data import file_hash,write_json
from xtbflow.v1.formal import evaluation_parents
from xtbflow.v1.metrics import cluster_summary
from xtbflow.v1.proxy import CatalogueMatcher


def main():
    ap=argparse.ArgumentParser()
    for key in ('run','catalogue','out'):ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--frozen',type=Path)
    a=ap.parse_args();started=time.monotonic();RDLogger.DisableLog('rdApp.error')
    m=json.loads((a.run/'manifest.json').read_text())
    if not m['complete'] or file_hash(a.run/'candidates.jsonl')!=m['candidates_sha256']:
        raise ValueError('unsealed control source')
    split=json.loads((a.catalogue/'split_manifest.json').read_text())
    dev,query_name=evaluation_parents(split,m,a.frozen)
    if not m['alpha_zero_exact'] or not m['all_step_B2_replay_exact']:raise ValueError('failed controls')
    if m['query_source_sha256']!=file_hash(a.catalogue/query_name):
        raise ValueError('query source mismatch')
    if any(v['split_hash']!=split['split_hash'] for v in m['models'].values()):raise ValueError('model split mismatch')
    parents={p['query_id']:p for p in json.loads((a.catalogue/'parent_catalog.json').read_text()) if p['parent_id'] in dev}
    refs=defaultdict(list)
    for line in (a.catalogue/'reference_catalog.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['parent_id'] in dev:refs[r['parent_id']].append(r)
    matchers={q:CatalogueMatcher(p,refs[p['parent_id']]) for q,p in parents.items()}
    index={}
    for line in (a.run/'candidates.jsonl').read_text().splitlines():
        r=json.loads(line);key=(r['query_id'],r['training_seed'],r['proposal'],r['branch'])
        if key in index:raise ValueError('duplicate control')
        r.update(matchers[r['query_id']].match(r['b_dec'],r['x'],r['is_fallback'],r['decode_status']))
        index[key]=r
    expected={(q,s,j,b) for q in parents for s in (0,1,2) for j in range(8) for b in ('F0','B1_cont','B2_cont')}
    if set(index)!=expected:raise ValueError('incomplete continuous controls')
    output=[]
    for q,p in parents.items():
        values=defaultdict(list)
        for seed in (0,1,2):
            for j in range(8):
                f0,b1,b2=(index[q,seed,j,b] for b in ('F0','B1_cont','B2_cont'))
                if f0['b_dec']!=b2['b_dec']:raise ValueError('replay decode differs')
                for label,r in [('B1',b1),('B2',b2)]:
                    values[label+'_utility_minus_F0'].append(r['event_utility']-f0['event_utility'])
                    values[label+'_match_minus_F0'].append(int(r['proxy_status']=='PROXY_MATCH')-int(f0['proxy_status']=='PROXY_MATCH'))
                values['B1_minus_B2_utility'].append(b1['event_utility']-b2['event_utility'])
        output.append(dict(parent_id=p['parent_id'],split_group=p['split_group'],
                           **{k:float(np.mean(v)) for k,v in values.items()}))
    report=dict(status='CONTINUOUS_CONTROLS_PASS',source_manifest=m,source_sha256=m['candidates_sha256'],
                parent_rows=output,summary={k:cluster_summary([r[k] for r in output],[r['split_group'] for r in output]) for k in values},
                split_hash=split['split_hash'],matcher_cpu_wall_s=time.monotonic()-started,
                split_name=m.get('split_name','development'),freeze_sha256=m.get('freeze_sha256'),
                interpretation='Three-seed continuous B1/B2 controls; explanatory, not a separate gate.')
    write_json(a.out,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_manifest','parent_rows')},indent=2))


if __name__=='__main__':main()
