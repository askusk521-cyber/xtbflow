"""Additive smc-2 acquisition; immutable old arms, imported smc-1 continuation."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import explore_smc as old
from review_oracle_benchmark import energy
from xtbflow.v1.geometry_information import min_distance
from xtbflow.v1.smc import selection_plan

SM=Path('/home/lhshen/xtbflow-runs/explore-smc-20261009/smc-2116d43')
R=Path('/home/lhshen/xtbflow-runs/followup-20261009')
V=Path('/home/lhshen/xtbflow-runs/v1a-20261007/data')
METHODS={'aimnet':'AIMNet2','gxtb':'g-xTB','gfn2':'GFN2-xTB'}


def digest(path):
    h=sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def clean():
    if old.git('status','--porcelain'):raise RuntimeError('Dirty source')
    if not old.git('branch','-r','--contains',old.git('rev-parse','HEAD')):raise RuntimeError('Unpushed source')


def guard_cpu():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Disable CUDA')
    if os.getpriority(os.PRIO_PROCESS,0)<10:raise RuntimeError('nice10 required')
    if float(Path('/proc/loadavg').read_text().split()[0])>120:raise RuntimeError('STOP high load')
    torch.set_num_threads(1)


def inputs():
    parents={p['parent_id']:p for p in json.loads((V/'parent_catalog.json').read_text())}
    g1=json.loads((SM/'g1_manifest.json').read_text())
    byq={p['query_id']:p for p in parents.values()}
    return parents,g1,byq


def verify_reuse():
    path=Path('docs/evidence/explore_smc/SHA256SUMS')
    checked={}
    for line in path.read_text().splitlines():
        expected,name=line.split(maxsplit=1);f=Path(name.strip().lstrip('*'))
        if digest(f)!=expected:raise RuntimeError('STOP smc-1 input hash mismatch: '+str(f))
        checked[str(f)]=expected
    if not json.loads((R/'oracle-gate.json').read_text())['passed']:raise RuntimeError('STOP oracle gate')
    gate=json.loads((R/'smc-parent-gate.json').read_text())
    # The archived independent gate has an explicit passed field.
    if not gate.get('passed'):raise RuntimeError('STOP original parent gate')
    return checked


def manifest(out,stage,started,files,extra=None):
    own=resource.getrusage(resource.RUSAGE_SELF);child=resource.getrusage(resource.RUSAGE_CHILDREN)
    m=dict(stage=stage,source_commit=old.git('rev-parse','HEAD'),dirty=False,
           config_sha256=digest('configs/explore/smc_oracle.json'),slurm_job_id=os.getenv('SLURM_JOB_ID'),
           input_sha256={str(f):digest(f) for f in files},wall_s=time.monotonic()-started,
           cpu_self_s=own.ru_utime+own.ru_stime,cpu_children_s=child.ru_utime+child.ru_stime,
           **(extra or {}))
    old.dump(out,m)


def score(z,x,anchor,cfg,method):
    if min_distance(x)<.5:return dict(status='collapsed',barrier=None)
    value,status=energy(method,z,x,cfg)
    ok=status=='ok' and np.isfinite(value) and anchor[1]=='ok' and np.isfinite(anchor[0])
    return dict(status='ok' if ok else (status if status!='ok' else 'anchor:'+anchor[1]),
                barrier=float(value-anchor[0]) if ok else None)


def checkpoint(a):
    clean();guard_cpu();started=time.monotonic();verified=verify_reuse()
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir()
    parents,g1,byq=inputs();cfg=json.loads(Path('configs/review/oracle_benchmark.json').read_text())['xtb']
    method=METHODS[a.method];anchors={};files=[V/'parent_catalog.json',SM/'g1_manifest.json',R/'oracle-gate.json',R/'smc-parent-gate.json']
    asset=Path(os.environ['REVIEW_AIMNET_MODELS'])/'aimnet2_wb97m_d3_0.pt' if a.method=='aimnet' else Path(os.environ['REVIEW_GXTB_BINARY'])
    expected='f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28' if a.method=='aimnet' else '1b4e30b68ed4e88b4075f60d97f4756ee440fe92d3294cc20826d53f8121cd26'
    if digest(asset)!=expected:raise RuntimeError('STOP model/binary hash mismatch')
    files.append(asset)
    with (a.out/'rows.jsonl').open('x') as output:
        for name in g1['batches']:
            path=SM/name;files.append(path);d=torch.load(path,weights_only=False)
            for c,(i,j) in enumerate(d['pairs']):
                p=byq[g1['queries'][i]['query_id']];pid=p['parent_id'];n=len(p['atomic_numbers'])
                if pid not in anchors:anchors[pid]=energy(method,p['atomic_numbers'],np.asarray(p['x_r']),cfg)
                for t,k in enumerate((30,35,40,45)):
                    s=score(p['atomic_numbers'],d['x_hat'][t,c,:n].numpy(),anchors[pid],cfg,method)
                    r=dict(parent_id=pid,seed=d['seed'],proposal=j,step=k,batch=name,start=(c//32)*32,method=a.method,score=s)
                    output.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
                output.flush()
            print('Scored checkpoint batch '+name,flush=True)
    manifest(a.out/'manifest.json','checkpoint',started,files,dict(method=a.method,complete=True,reuse_sha256=verified,rows_sha256=digest(a.out/'rows.jsonl')))


def make_plans(records):
    groups={}
    for r in records:
        key=(r['batch'],r['start'],r['parent_id'],r['seed'],r['step'],r['method'])
        if r['proposal'] in groups.setdefault(key,{}):raise ValueError('duplicate proposal')
        groups[key][r['proposal']]=r['score']
    plans=[]
    for (batch,start,pid,seed,k,method),scores in sorted(groups.items()):
        if set(scores)!=set(range(32)):raise ValueError('incomplete population')
        survivors,clones=selection_plan(pid,seed,[scores[j] for j in range(32)],k,'raw',curvature_layers=False)
        plans.append(dict(batch=batch,start=start,parent_id=pid,seed=seed,step=k,arm=f'{method}_{k/50:.1f}',survivors=survivors,clones=clones))
    return plans


def plan(a):
    clean();guard_cpu();started=time.monotonic();verify_reuse()
    if a.out.exists():raise FileExistsError(a.out)
    records=[];files=[]
    for path in a.scores:
        m=json.loads((path/'manifest.json').read_text());rows=path/'rows.jsonl'
        if not m['complete'] or digest(rows)!=m['rows_sha256']:raise RuntimeError('STOP checkpoint hash')
        records.extend(old.rows(rows));files.extend([rows,path/'manifest.json'])
    plans=make_plans(records)
    if len(plans)!=62*3*8 or {r['arm'] for r in plans}!={f'{method}_{k/50:.1f}' for method in ('aimnet','gxtb') for k in (30,35,40,45)}:raise ValueError('incomplete plan coverage')
    a.out.mkdir();old.dump(a.out/'plans.json',plans)
    _,g1,_=inputs()
    for name in g1['batches']+['g1_manifest.json']:(a.out/name).symlink_to(SM/name)
    old.dump(a.out/'c1_manifest.json',dict(gate_passed=True,explanation='smc-2 checkpoint plan; three independently verified pre-generation gates'))
    manifest(a.out/'plan_manifest.json','plan',started,files,dict(plans_sha256=digest(a.out/'plans.json'),n=len(plans)))


def g2(a):
    clean()
    if not os.getenv('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    started=time.monotonic();verify_reuse()
    m=json.loads((a.out/'plan_manifest.json').read_text())
    if digest(a.out/'plans.json')!=m['plans_sha256']:raise RuntimeError('STOP plan hash')
    if (a.out/'g2_manifest.json').exists() or list(a.out.glob('g2-*.pt')):raise RuntimeError('No external retry')
    # Import the original implementation unchanged: exact restart noise, batch layout and rollout.
    args=SimpleNamespace(config=Path('configs/explore/smc_selection.json'),out=a.out,workers=1,
                         limit_parents=None,proposals=32,seeds=[0,1,2],smoke=False)
    old.cmd_g2(args)
    manifest(a.out/'oracle_g2_manifest.json','g2',started,[a.out/'plans.json',a.out/'plan_manifest.json',a.out/'g2_manifest.json'])


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='stage',required=True)
    s=sub.add_parser('checkpoint');s.add_argument('--method',choices=['aimnet','gxtb'],required=True)
    s.add_argument('--out',type=Path,required=True)
    s=sub.add_parser('plan');s.add_argument('--scores',type=Path,nargs=2,required=True);s.add_argument('--out',type=Path,required=True)
    s=sub.add_parser('g2');s.add_argument('--out',type=Path,required=True)
    a=p.parse_args();globals()[a.stage](a)


if __name__=='__main__':main()
