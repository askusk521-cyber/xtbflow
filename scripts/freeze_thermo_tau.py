"""Freeze the training-only 95th percentile, never inspecting screen energies."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--rows',type=Path,required=True)
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    expected=[json.loads(s) for s in a.input.read_text().splitlines()]
    rows=[json.loads(s) for s in a.rows.read_text().splitlines()]
    train=set(json.loads(a.split.read_text())['train']['parent_ids'])
    assert len(train)==386
    assert {r['parent_id'] for r in rows}==train
    key=lambda r:(r['parent_id'],r['channel_id'])
    assert list(map(key,expected))==list(map(key,rows))
    assert len(set(map(key,rows)))==len(rows)
    values=[r['delta_e_kcal'] for r in rows if r['delta_e_kcal'] is not None]
    assert values and np.isfinite(values).all()
    result=dict(schema='xtbflow-enum-thermo-tau/1',exploratory=True,train_parents=386,
                total=len(rows),successful=len(values),success_rate=len(values)/len(rows),
                tau_kcal=float(np.percentile(values,95)),percentile=95,interpolation='linear',
                distribution={str(q):float(np.percentile(values,q)) for q in (0,1,5,25,50,75,95,99,100)},
                failure_reasons=dict(Counter('product:'+r['product']['status']+';reactant:'+r['reactant']['status'] for r in rows if r['delta_e_kcal'] is None)),
                input_sha256={str(f):sha256(f.read_bytes()).hexdigest() for f in (a.input,a.rows,a.split)})
    if a.out.exists():raise FileExistsError(a.out)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ('total','successful','tau_kcal')}))


if __name__=='__main__':main()
