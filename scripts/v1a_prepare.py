"""Build an offline V1a catalogue and reserve formula-isolated train/dev/screen."""
import argparse
import json
from pathlib import Path

from xtbflow.v1.data import build_catalogue, file_hash, inventory, reserve_split, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['inventory', 'build', 'split'])
    ap.add_argument('--cache', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cap', type=int, default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.command in ('inventory', 'build'):
        if not a.cache:
            ap.error('--cache required')
        parents, refs, failures = build_catalogue(a.cache)
        counts = inventory(parents, refs, failures)
        counts['source_cache_sha256'] = file_hash(a.cache)
        write_json(a.out/'cohort_counts.json', counts)
        write_json(a.out/'parent_catalog.json', parents)
        with (a.out/'reference_catalog.jsonl').open('w', encoding='utf-8') as f:
            for row in refs:
                f.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in counts.items() if k!='official_splits'}))
        print(json.dumps({s:{k:v for k,v in row.items() if k not in ('group_sizes','top10_groups')}
                          for s,row in counts['official_splits'].items()}, indent=2))
    if a.command in ('split','build'):
        if not a.cap:
            ap.error('--cap must be chosen after inventory')
        parents=json.loads((a.out/'parent_catalog.json').read_text(encoding='utf-8'))
        split=reserve_split(parents,a.cap)
        write_json(a.out/'split_manifest.json',split)
        print(json.dumps({s:{'parents':len(split[s]['parent_ids']),
                             'formulas':len(split[s]['formula_groups'])}
                          for s in ('train','development','screen_reserve')}))


if __name__=='__main__':
    main()
