"""Audit required enumeration outputs, recovery and source provenance."""
import argparse
import json
from pathlib import Path
import subprocess
from xtbflow.v1.data import file_hash,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=Path('/home/lhshen/xtbflow-runs/v1a-20261007')
    splits=json.loads((root/'data/split_manifest.json').read_text())
    screen=json.loads((root/'formal/freeze.json').read_text())['parent_ids']
    expected={'enum-screen-9403d6c':screen,'enum-dev-7c28436':splits['development']['parent_ids']}
    files=[]
    for directory,ids in expected.items():
        for pid in sorted(ids):
            suffixes=('.json','.npz','.scores.npz') if directory.startswith('enum-screen') else ('.json','.npz')
            for suffix in suffixes:
                path=a.run/directory/(pid+suffix)
                if not path.exists():raise RuntimeError('missing '+str(path))
                files.append(path)
    recovery=[]
    for i in range(4):
        path=a.run/'training-b3-6f8ab54'/f'shard-{i}.json';d=json.loads(path.read_text())
        if not d['complete'] or not d['passed']:raise RuntimeError('training recovery incomplete')
        files.append(path);recovery.extend(d['parents'])
    if {r['parent_id'] for r in recovery}!=set(splits['train']['parent_ids']):raise RuntimeError('training parent set mismatch')
    for name in ('b0-reproduction.json','domain-decision.json','training-domain-1d63a02.json'):
        files.append(a.run/name)
    configs=[Path('configs/review')/name for name in ('enum_baseline.json','enum_domain_audit.json','enum_domain_selected.json')]
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    manifest=dict(source_commit=source,dirty=bool(subprocess.check_output(['git','status','--porcelain']).strip()),
        config_sha256={str(c):file_hash(c) for c in configs},
        training_parents=len(recovery),training_events_recovered=sum(r['recovered'] for r in recovery),
        enumeration_sources=['9403d6c','7c28436','d414174'],training_sources=['6f8ab54','d414174'],
        scorer_sources=['75cd0cc','d414174'],
        note='Resumed only missing parents from pushed clean clones; canonical event domain unchanged. Earlier pilot using 344 split-reserve parents excluded.',
        files=[dict(path=str(f),sha256=file_hash(f)) for f in files])
    a.out.mkdir(parents=True,exist_ok=True);write_json(a.out/'manifest.json',manifest)
    (a.out/'SHA256SUMS.remote').write_text(''.join(file_hash(f)+'  '+str(f)+'\n' for f in files))


if __name__=='__main__':main()
