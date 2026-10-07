"""Evaluate a sealed development pulse stream; generated input never sees labels."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

from xtbflow.v1.data import file_hash,write_json
from xtbflow.v1.development import window_summary
from xtbflow.v1.proxy import CatalogueMatcher


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--catalogue',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();start=time.monotonic()
    manifest=json.loads((a.run/'manifest.json').read_text())
    if not manifest['complete'] or file_hash(a.run/'candidates.jsonl')!=manifest['candidates_sha256']:
        raise ValueError('unsealed source')
    split=json.loads((a.catalogue/'split_manifest.json').read_text())
    if manifest['models']['split_hash']!=split['split_hash']:
        raise ValueError('split mismatch')
    if file_hash(a.catalogue/'development_queries.jsonl')!=manifest['query_source_sha256']:
        raise ValueError('development queries changed')
    dev=set(split['development']['parent_ids'])
    parents={p['query_id']:p for p in json.loads((a.catalogue/'parent_catalog.json').read_text())
             if p['parent_id'] in dev}
    refs=defaultdict(list)
    for line in (a.catalogue/'reference_catalog.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['parent_id'] in dev:refs[r['parent_id']].append(r)
    matchers={q:CatalogueMatcher(p,refs[p['parent_id']]) for q,p in parents.items()}
    evaluated=[];cache={}
    with (a.run/'candidates.jsonl').open() as f:
        for line in f:
            r=json.loads(line);q=r['query_id']
            if q not in parents:raise ValueError('non-development input')
            key=(q,json.dumps([r['b_dec'],r['x'],r['decode_status'],r['is_fallback']]))
            if key not in cache:
                cache[key]=matchers[q].match(r['b_dec'],r['x'],r['is_fallback'],r['decode_status'])
            r.update(cache[key]);r.update(parent_id=parents[q]['parent_id'],split_group=parents[q]['split_group'])
            r.pop('x');evaluated.append(r)
    report=window_summary(evaluated,manifest,parents)
    report.update(source_sha256=manifest['candidates_sha256'],split_hash=split['split_hash'],
                  training_seed=manifest['training_seed'],path=manifest['path'],
                  matcher_cpu_wall_s=time.monotonic()-start,unique_endpoints_matched=len(cache))
    write_json(a.out,report)
    with a.out.with_suffix('.candidates.jsonl').open('w') as f:
        for r in evaluated:f.write(json.dumps(r,allow_nan=False)+'\n')
    print(json.dumps(dict(integrity=report['integrity'],cpu_wall_s=report['matcher_cpu_wall_s'],
        windows=[dict(t=w['requested_t'],M0=w['summary']['M0']['estimate'],MR=w['summary']['MR']['estimate'],
                      changed=w['summary']['valid_event_change']['estimate'],
                      calibration=w['matched_F0_calibration']['within_parent_concordance']) for w in report['windows']]),indent=2))


if __name__=='__main__':main()
