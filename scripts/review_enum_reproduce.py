"""Mandatory completed-candidate B0 replication, independent of enumeration."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import numpy as np
from xtbflow.v1.data import write_json

EXPECTED={1:0.10721247563352823,2:0.19103313840155944,4:0.30994152046783624,8:0.4688109161793372,16:0.6257309941520468}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    by=defaultdict(list)
    with gzip.open('/home/lhshen/xtbflow-runs/v1a-r-20261008/extract/streams.jsonl.gz','rt') as handle:
        for line in handle:
            r=json.loads(line)
            if r['arm']=='B0':by[r['parent_id']].append(r)
    assert len(by)==342 and all(len(rows)==3 for rows in by.values())
    rates={str(k):float(np.mean([np.mean([any(r['event'][:k]) for r in rows]) for pid,rows in sorted(by.items())])) for k in EXPECTED}
    errors={str(k):abs(rates[str(k)]-v) for k,v in EXPECTED.items()}
    passed=max(errors.values())<1e-9
    write_json(a.out,dict(passed=passed,rates=rates,errors=errors,reference='PR107 reanalysis.json analysis_a.completed_candidates.event_best.arm_rates.B0'))
    assert passed


if __name__=='__main__':main()
