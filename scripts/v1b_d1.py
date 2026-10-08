"""V1b D1: fixed development chains, measured cost and gpu4pyscf benchmark.

plan: choose 8 development candidates (never formal-screen parents) by a fixed
      hash order inside four proxy classes x small/large molecules, before any H.
run:  execute one planned item (candidate chain, anchor Opt+Freq, native best
      reference chain, or CPU/GPU parity benchmark) into its own directory.
"""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np

CLASSES=('MATCH_BEST','MATCH_NONBEST','OUTSIDE_CATALOGUE_EVENT','KNOWN_EVENT_GEOMETRY_MISS')


def key(s):return sha256(s.encode()).hexdigest()


def plan(a):
    from xtbflow.v1.audit_sampling import proxy_class
    split=json.loads((a.catalogue/'split_manifest.json').read_text())
    dev=set(split['development']['parent_ids'])
    parents={p['query_id']:p for p in json.loads((a.catalogue/'parent_catalog.json').read_text()) if p['parent_id'] in dev}
    raw={}
    for line in a.dev_stream.read_text().splitlines():
        row=json.loads(line)
        for c in row['candidates']:raw[c['candidate_id']]=c
    pools=defaultdict(list)
    for line in a.dev_candidates.read_text().splitlines():
        c=json.loads(line)
        if c['parent_id'] not in dev or c['completion_units']>800+1e-8:continue
        cls=proxy_class(c)
        if cls not in CLASSES:continue
        n=len(parents[c['candidate_id'].split('/')[0]]['atomic_numbers'])
        pools[cls,'small' if n<=11 else 'large'].append(c)
    items=[]
    for cls in CLASSES:
        for size in ('small','large'):
            pool=sorted(pools[cls,size],key=lambda c:key(c['candidate_id']))
            if not pool:raise ValueError(f'no development candidate for {cls}/{size}')
            c=pool[0];q=c['candidate_id'].split('/')[0];p=parents[q];r=raw[c['candidate_id']]
            items.append(dict(kind='candidate',item_id=key(c['candidate_id'])[:12],candidate_id=c['candidate_id'],
                              proxy_class=cls,size=size,query_id=q,parent_id=p['parent_id'],z=p['atomic_numbers'],
                              b_r=p['b_r'],perms=p['permutations'],b_predicted=r['b_dec'],x_start=r['x']))
    queries=sorted({i['query_id'] for i in items})
    for q in queries:
        p=parents[q]
        items.append(dict(kind='anchor',item_id=key('anchor/'+q)[:12],query_id=q,parent_id=p['parent_id'],
                          z=p['atomic_numbers'],x_start=p['x_r']))
    # One native best reference, in query atom order, for the D2 reference path.
    p=parents[sorted(queries,key=key)[0]]
    ref=None
    for line in (a.catalogue/'reference_catalog.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['parent_id']==p['parent_id'] and r['reference_id'] in p['best_reference_ids']:ref=r;break
    perm=np.asarray(p['permutations'][0])
    items.append(dict(kind='reference',item_id=key('ref/'+ref['reference_id'])[:12],query_id=p['query_id'],
                      parent_id=p['parent_id'],reference_id=ref['reference_id'],z=p['atomic_numbers'],b_r=p['b_r'],
                      perms=p['permutations'],b_predicted=np.asarray(ref['b_p'])[np.ix_(perm,perm)].tolist(),
                      x_start=np.asarray(ref['x_ts'])[perm].tolist()))
    small=min((i for i in items if i['kind']=='anchor'),key=lambda i:len(i['z']))
    large=max((i for i in items if i['kind']=='anchor'),key=lambda i:len(i['z']))
    for i in (small,large):
        items.append(dict(kind='benchmark',item_id=key('bench/'+i['query_id'])[:12],query_id=i['query_id'],
                          z=i['z'],x_start=i['x_start']))
    out=dict(schema='xtbflow-v1b-d1-plan/1',frozen_before_any_H=True,items=items,
             selection='first by sha256(candidate_id) per proxy class x {<=11, >11 atoms}; development split only')
    a.out.write_text(json.dumps(out,indent=1))
    print(json.dumps([dict(kind=i['kind'],id=i['item_id'],cls=i.get('proxy_class'),n=len(i['z'])) for i in items],indent=0))


def run(a):
    from xtbflow.v1 import qc_protocol as qc
    items={i['item_id']:i for i in json.loads(a.plan.read_text())['items']}
    i=items[a.item];work=a.work/f"{i['kind']}_{i['item_id']}"
    reuse=None;reuse_irc=None
    if a.resume_endpoints:
        # Endpoint retry rule added after IRC revision 2: keep the completed IRC, redo endpoints.
        old=work/'verdict.json';prev=json.loads(old.read_text());reuse,reuse_irc=prev['ts'],prev['irc']
        if prev.get('irc',{}).get('status')!='IRC_COMPLETE':raise ValueError('resume endpoints only after a complete IRC')
        old.rename(work/'verdict_rev2_endpoints1.json');work=work/'endpoint_revision2';work.mkdir()
    elif a.resume_irc:
        # Re-run IRC/endpoints from the SAME certified TS under the revised IRC settings;
        # the first-pass verdict is kept beside the new one, never overwritten.
        old=work/'verdict.json';first=json.loads(old.read_text());reuse=first['ts']
        if first.get('protocol_version') is not None or reuse.get('status')!='TS_OPTFREQ_PASS':
            raise ValueError('resume only first-pass chains with a passing TS')
        old.rename(work/'verdict_pass1.json');work=work/'irc_revision2';work.mkdir()
    elif (work/'verdict.json').exists():raise FileExistsError('item already ran')
    work.mkdir(parents=True,exist_ok=True)
    z=np.asarray(i['z']);x=np.asarray(i['x_start'],dtype=float)
    method=qc.Method(device='gpu');meter=qc.Meter();t0=time.perf_counter()
    if i['kind'] in ('candidate','reference'):
        out=qc.certification_chain(z,x,i['b_r'],i['perms'],i['b_predicted'],method,work,meter,reuse_ts=reuse,reuse_irc=reuse_irc)
        if reuse is not None:(work.parent/'verdict.json').write_text(json.dumps(out,indent=1,default=float))
    elif i['kind']=='anchor':
        out=qc.minimum_stage(z,x,method,meter,work,'anchor')
        qc.finish(out,meter,t0,work)
    else:
        out={}
        for device in ('gpu','cpu'):
            m=qc.Method(device=device);mt=qc.Meter()
            e,h=qc.hessian(z,x,m,mt,f'bench_{device}')
            _,g,_=qc.single_point(z,x,m,mt,f'bench_{device}_grad')
            out[device]=dict(energy_hartree=e,gradient_norm=float(np.linalg.norm(g)),timings=mt.rows,
                             hessian_sha=sha256(np.round(h,6).tobytes()).hexdigest()[:16])
            np.save(work/f'hessian_{device}.npy',h)
        hg,hc=np.load(work/'hessian_gpu.npy'),np.load(work/'hessian_cpu.npy')
        out['parity']=dict(energy_diff_hartree=out['gpu']['energy_hartree']-out['cpu']['energy_hartree'],
                           hessian_max_abs_diff=float(np.abs(hg-hc).max()))
        qc.finish(out,meter,t0,work)
    import subprocess
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],
                       capture_output=True,text=True).stdout.strip()
    import gpu4pyscf,pyscf,geometric
    env=dict(gpu=gpu,gpu4pyscf=gpu4pyscf.__version__,pyscf=pyscf.__version__,geometric=geometric.__version__)
    (work/'environment.json').write_text(json.dumps(env,indent=1))
    print(json.dumps(dict(item=a.item,kind=i['kind'],status=out.get('strict_status',out.get('status')),
                          wall_s=time.perf_counter()-t0),indent=1))


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('plan')
    for k in ('catalogue','dev-stream','dev-candidates','out'):p.add_argument('--'+k,type=Path,required=True)
    r=sub.add_parser('run')
    r.add_argument('--plan',type=Path,required=True);r.add_argument('--item',required=True)
    r.add_argument('--work',type=Path,required=True)
    r.add_argument('--resume-irc',action='store_true')
    r.add_argument('--resume-endpoints',action='store_true')
    a=ap.parse_args()
    plan(a) if a.cmd=='plan' else run(a)


if __name__=='__main__':main()
