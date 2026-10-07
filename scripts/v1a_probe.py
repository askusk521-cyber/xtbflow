"""Offline development audits read sealed candidates and the reference catalogue."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from xtbflow.v1.data import file_hash,write_json
from xtbflow.v1.proxy import CatalogueMatcher


def baseline(a):
    source=json.loads((a.run/'manifest.json').read_text())
    if not source['complete'] or source['candidates_sha256']!=file_hash(a.run/'candidates.jsonl'):
        raise ValueError('candidate source is not sealed')
    parents=json.loads((a.catalogue/'parent_catalog.json').read_text())
    split=json.loads((a.catalogue/'split_manifest.json').read_text())
    if any(m['split_hash']!=split['split_hash'] for m in source['models'].values()):
        raise ValueError('baseline weights use a different training split')
    if source['query_source_sha256']!=file_hash(a.catalogue/'development_queries.jsonl'):
        raise ValueError('development query view changed')
    dev=set(split['development']['parent_ids'])
    refs=defaultdict(list)
    for line in (a.catalogue/'reference_catalog.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['parent_id'] in dev:refs[r['parent_id']].append(r)
    parents={p['query_id']:p for p in parents if p['parent_id'] in dev}
    matchers={q:CatalogueMatcher(p,refs[p['parent_id']]) for q,p in parents.items()}
    evaluated=[];start=time.monotonic()
    for line in (a.run/'candidates.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['query_id'] not in parents:raise ValueError('non-development query')
        p=parents[r['query_id']]
        r.update(matchers[r['query_id']].match(r['b_dec'],r['x'],r['is_fallback'],r['decode_status']))
        r.update(parent_id=p['parent_id'],split_group=p['split_group'])
        evaluated.append(r)
    grouped=defaultdict(list)
    for r in evaluated:grouped[r['arm'],r['parent_id']].append(r)
    summaries={}
    for arm in ['B_M0re','B0_sync','B0_event_lead2']:
        rows=[]
        for pid in sorted(dev):
            rr=grouped[arm,pid]
            if len(rr)!=source['proposals'] or len({r['proposal'] for r in rr})!=len(rr):
                raise ValueError('missing or duplicate paired proposals')
            rows.append(dict(parent_id=pid,split_group=rr[0]['split_group'],
                             joint_recall=any(r['proxy_status']=='PROXY_MATCH' for r in rr),
                             best_recall=any(r['hits_best_reference'] for r in rr),
                             valid_rate=np.mean([r['proxy_status']!='INVALID_OUTPUT' for r in rr]).item(),
                             unique_events=len({r['predicted_channel_id'] for r in rr if r['predicted_channel_id']})))
        summaries[arm]=dict(parent_rows=rows,
            **{k:float(np.mean([r[k] for r in rows])) for k in
               ['joint_recall','best_recall','valid_rate','unique_events']})
    delta_joint=summaries['B0_sync']['joint_recall']-summaries['B_M0re']['joint_recall']
    delta_valid=summaries['B0_sync']['valid_rate']-summaries['B_M0re']['valid_rate']
    report=dict(status='HOLD_BASELINE' if delta_joint<-.05 or delta_valid<-.10 else 'BASELINE_DEVELOPMENT_PASS',
                joint_recall_drop=delta_joint,valid_rate_drop=delta_valid,
                summaries=summaries,matcher_cpu_wall_s=time.monotonic()-start,
                source_hash=source['candidates_sha256'],split_hash=split['split_hash'],
                interpretation='Development diagnostic thresholds; not an equivalence test or formal outcome.')
    write_json(a.out,report)
    with a.out.with_suffix('.candidates.jsonl').open('w') as f:
        for r in evaluated:f.write(json.dumps(r,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='summaries'},indent=2))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=['baseline'])
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--catalogue',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    return baseline(a)


if __name__=='__main__':main()
