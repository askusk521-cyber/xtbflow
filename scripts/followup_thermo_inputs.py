"""Prepare training catalogue or two-dev smoke inputs in xtbflow RDKit environment.
No screen thermodynamic acquisition is possible through this entry point.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--mode',choices=['smoke','timing','train'],required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    files=[a.root/'data/parent_catalog.json',a.root/'data/reference_catalog.jsonl',a.root/'data/split_manifest.json']
    parents={r['parent_id']:r for r in json.loads(files[0].read_text())}
    split=json.loads(files[2].read_text())
    ids=sorted(split['development' if a.mode=='smoke' else 'train']['parent_ids'])
    if a.mode=='smoke':ids=ids[:2]
    elif a.mode=='timing':ids=ids[:5]
    else:assert len(ids)==386
    wanted=set(ids)
    representatives={}
    with files[1].open() as handle:
        for line in handle:
            r=json.loads(line)
            if r['parent_id'] not in wanted:continue
            key=(r['parent_id'],r['channel_id'])
            if key not in representatives or r['reference_id']<representatives[key]['reference_id']:
                representatives[key]=r
    with a.out.open('x') as out:
        for key,r in sorted(representatives.items()):
            parent=parents[r['parent_id']]
            row=dict(parent_id=r['parent_id'],channel_id=r['channel_id'],reference_id=r['reference_id'],
                     z=parent['atomic_numbers'],b_r=parent['b_r'],b=r['b_p'])
            out.write(json.dumps(row,sort_keys=True)+'\n')
    manifest=dict(mode=a.mode,parents=ids,rows=len(representatives),
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
                  slurm_job_id=None,input_sha256={str(f):sha256(f.read_bytes()).hexdigest() for f in files},
                  output_sha256=sha256(a.out.read_bytes()).hexdigest())
    if manifest['dirty']:raise RuntimeError('Dirty source')
    a.out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print('Inputs prepared without scientific metrics.')


if __name__=='__main__':main()
