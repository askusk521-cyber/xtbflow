"""Fresh 558-TS AIMNet2/g-xTB reproduction gate; CPU-only, fixed methods.
Acquire methods in their approved environments, then analyze in xtbflow.
No new SMC geometry generation is allowed until the gate passes.
"""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from review_oracle_benchmark import energy
from xtbflow.v1.geometry_information import event_table,ranking_metrics
from xtbflow.v1.metrics import cluster_summary

EXPECTED={'AIMNet2':0.9239845589178955,'g-xTB':0.9089169115946805}


def acquire(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Disable CUDA')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('Dirty source')
    resource.setrlimit(resource.RLIMIT_CPU,(a.max_cpu_seconds,a.max_cpu_seconds))
    files=[a.root/'data/parent_catalog.json',a.root/'data/reference_catalog.jsonl',Path('docs/evidence/explore_geoinfo/check_a_rows.jsonl'),Path('configs/review/oracle_benchmark.json')]
    parents={r['parent_id']:r for r in json.loads(files[0].read_text())}
    refs={r['reference_id']:r for r in map(json.loads,files[1].open())}
    selected=list(map(json.loads,files[2].open()));assert len(selected)==558
    cfg=json.loads(files[3].read_text())['xtb'];anchors={};t0=time.perf_counter()
    if a.method=='AIMNet2':
        model=Path(os.environ['REVIEW_AIMNET_MODELS'])/'aimnet2_wb97m_d3_0.pt'
        assert sha256(model.read_bytes()).hexdigest()=='f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28'
        files.append(model)
    else:
        binary=Path(os.environ['REVIEW_GXTB_BINARY'])
        assert sha256(binary.read_bytes()).hexdigest()=='1b4e30b68ed4e88b4075f60d97f4756ee440fe92d3294cc20826d53f8121cd26'
        files.append(binary)
    with a.out.open('x') as output:
        for row in selected:
            r=refs[row['reference_id']];p=parents[r['parent_id']];pid=p['parent_id']
            if pid not in anchors:anchors[pid]=energy(a.method,p['atomic_numbers'],np.asarray(p['x_r']),cfg)
            value,status=energy(a.method,p['atomic_numbers'],np.asarray(r['x_ts']),cfg)
            anchor,anchor_status=anchors[pid]
            result=dict(method=a.method,parent_id=pid,reference_id=r['reference_id'],channel_id=r['channel_id'],
                        split_group=p['split_group'],catalog_barrier_kcal=r['catalog_barrier_kcal'],
                        barrier=value-anchor if status=='ok' and anchor_status=='ok' else None,
                        status=status,anchor_status=anchor_status)
            output.write(json.dumps(result,sort_keys=True)+'\n');output.flush()
    m=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dirty=False,
           config_sha256=sha256(Path('configs/explore/smc_oracle.json').read_bytes()).hexdigest(),slurm_job_id=None,
           input_sha256={str(f):sha256(f.read_bytes()).hexdigest() for f in files},wall_s=time.perf_counter()-t0,
           cpu_self_s=time.process_time(),cpu_children_s=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime)
    a.out.with_suffix('.manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n')


def analyze(a):
    out={}
    for path in a.rows:
        rows=list(map(json.loads,path.open()));assert len(rows)==558
        method=rows[0]['method'];assert method in EXPECTED
        grouped=defaultdict(list)
        for r in rows:
            if r['barrier'] is not None:grouped[r['parent_id']].append(r)
        values={};groups={r['parent_id']:r['split_group'] for r in rows}
        for pid,records in sorted(grouped.items()):
            _,target,pred=event_table(records,'barrier')
            metrics=ranking_metrics(target,pred)
            if metrics is not None:values[pid]=metrics['rho']
        ids=sorted(values)
        summary=cluster_summary([values[p] for p in ids],[groups[p] for p in ids])
        out[method]=dict(summary=summary,expected=EXPECTED[method],passed=abs(summary['estimate']-EXPECTED[method])<1e-9,
                         rows_sha256=sha256(path.read_bytes()).hexdigest(),failures=sum(r['barrier'] is None for r in rows))
    assert set(out)==set(EXPECTED)
    result=dict(passed=all(v['passed'] for v in out.values()),methods=out)
    if a.out.exists():raise FileExistsError(a.out)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    if not result['passed']:raise RuntimeError('STOP: oracle reproduction gate failed')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    ac=sub.add_parser('acquire');ac.add_argument('--root',type=Path,required=True);ac.add_argument('--method',choices=list(EXPECTED),required=True)
    ac.add_argument('--out',type=Path,required=True);ac.add_argument('--max-cpu-seconds',type=int,required=True)
    an=sub.add_parser('analyze');an.add_argument('--rows',type=Path,nargs=2,required=True);an.add_argument('--out',type=Path,required=True)
    a=p.parse_args();acquire(a) if a.cmd=='acquire' else analyze(a)


if __name__=='__main__':main()
