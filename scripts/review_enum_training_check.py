"""Mandatory recovery of every in-domain training catalogue event."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from rdkit import RDLogger
from xtbflow.v1.enumeration import enumerate_events
from xtbflow.v1.data import write_json

spec=importlib.util.spec_from_file_location('audit',Path(__file__).with_name('review_enum_domain_audit.py'))
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--size',type=int,required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    ids=sorted(json.loads((a.data/'split_manifest.json').read_text())['train']['parent_ids'])[a.shard::a.shards]
    parents={r['parent_id']:r for r in json.loads((a.data/'parent_catalog.json').read_text()) if r['parent_id'] in ids}
    expected={pid:set() for pid in ids}
    with (a.data/'reference_catalog.jsonl').open() as handle:
        for line in handle:
            r=json.loads(line);pid=r['parent_id']
            if pid not in parents:continue
            parent=parents[pid];e=audit.event_domain(parent['atomic_numbers'],parent['b_r'],r['b_p'])
            if e['broken']<=a.size and e['formed']<=a.size and all(e[k] for k in ('diagonal_allowed','touched_only','conserved','charge_allowed')):
                expected[pid].add(r['channel_id'])
    RDLogger.DisableLog('rdApp.*');rows=[]
    if a.out.exists():rows=json.loads(a.out.read_text())['parents']
    done={r['parent_id'] for r in rows}
    for pid in ids:
        if pid in done:continue
        parent=parents[pid];start=time.monotonic();found=set()
        # May end once every expected channel has actually been regenerated.
        if expected[pid]:
            for channel,_ in enumerate_events(parent['atomic_numbers'],parent['b_r'],parent['permutations'],a.size):
                if channel in expected[pid]:found.add(channel)
                if found==expected[pid]:break
        missing=sorted(expected[pid]-found)
        rows.append(dict(parent_id=pid,expected=len(expected[pid]),recovered=len(found),missing=missing,seconds=time.monotonic()-start))
        write_json(a.out,dict(complete=len(rows)==len(ids),passed=all(not r['missing'] for r in rows),parents=rows,size=a.size))
        print(pid,len(found),len(missing),time.monotonic()-start,flush=True)
        if missing:raise AssertionError('training catalogue event missing')


if __name__=='__main__':main()
