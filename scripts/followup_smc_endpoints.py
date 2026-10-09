"""Decode only new SMC arms, enforce survivor gate, score all immutable endpoints."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import followup_smc_oracle as common
from followup_smc_oracle import old,SM,R,V,METHODS,clean,guard_cpu,inputs,manifest,digest,score,energy


def tasks(new):
    _,g1,_=inputs()
    for name in g1['batches']:
        d=torch.load(SM/name,weights_only=False)
        yield 'none',name,d,d['b_trace'][-1],d['x_trace'][-1],SM/name
    for root in (SM,new):
        g2=json.loads((root/'g2_manifest.json').read_text())
        for name in g2['files']:
            d=torch.load(root/name,weights_only=False)
            source=torch.load(SM/d['batch'],weights_only=False)
            yield d['arm'],d['batch'],source,d['b'],d['x'],root/name


def decode(a):
    clean();guard_cpu();started=time.monotonic();common.verify_reuse()
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir()
    parents,g1,byq=inputs();refs=old.rows(V/'reference_catalog.jsonl');byparent={pid:[] for pid in parents}
    for r in refs:byparent[r['parent_id']].append(r)
    plans=json.loads((a.new/'plans.json').read_text());lookup={(p['arm'],p['parent_id'],p['seed']):p for p in plans}
    none={(r['parent_id'],r['seed'],r['proposal']):r for r in json.loads((SM/'none_decoded.json').read_text())}
    output=[];checks={};files=[a.new/'plans.json',V/'reference_catalog.jsonl',SM/'none_decoded.json']
    g2=json.loads((a.new/'g2_manifest.json').read_text())
    for name in g2['files']:
        path=a.new/name;files.append(path);d=torch.load(path,weights_only=False);source=torch.load(SM/d['batch'],weights_only=False)
        for c,(i,j) in enumerate(source['pairs']):
            p=byq[g1['queries'][i]['query_id']];pid=p['parent_id'];seed=source['seed'];arm=d['arm']
            r=old.decode_endpoint(p,byparent[pid],d['b'][c],d['x'][c]);plan=lookup[arm,pid,seed];clones=dict(plan['clones'])
            r.update(parent_id=pid,seed=seed,proposal=j,arm=arm,ancestor=clones.get(j,j),is_clone=j in clones)
            if j not in clones:
                st=checks.setdefault(arm,dict(n=0,mismatches=0,max_abs_dx=0.))
                st['n']+=1;st['mismatches']+=int(r['channel_id']!=none[pid,seed,j]['channel_id'])
                st['max_abs_dx']=max(st['max_abs_dx'],float((d['x'][c]-source['x_trace'][-1,c]).abs().max()))
            output.append(r)
        print('Decoded '+name,flush=True)
    for st in checks.values():st['agreement']=1-st['mismatches']/st['n']
    passed=len(checks)==8 and all(st['agreement']>=.999 for st in checks.values())
    old.dump(a.out/'new_rows.json',output)
    manifest(a.out/'manifest.json','decode',started,files,dict(gate_passed=passed,survivor_checks=checks,n=len(output)))
    if not passed:raise RuntimeError('STOP survivor gate failed')


def acquire(a):
    clean();guard_cpu();started=time.monotonic();common.verify_reuse()
    if not json.loads((a.decoded/'manifest.json').read_text())['gate_passed']:raise RuntimeError('STOP survivor gate')
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir()
    parents,g1,byq=inputs();cfg=json.loads(Path('configs/review/oracle_benchmark.json').read_text())['xtb'];method=METHODS[a.method]
    anchors={};cache={};files=[a.decoded/'manifest.json',SM/'endpoint_rows.json']
    oldrows=json.loads((SM/'endpoint_rows.json').read_text())
    oldscore={(r['arm'],r['parent_id'],r['seed'],r['proposal']):r['score']['raw_barrier'] for r in oldrows}
    assets={'aimnet':('REVIEW_AIMNET_MODELS','f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28'),
            'gxtb':('REVIEW_GXTB_BINARY','1b4e30b68ed4e88b4075f60d97f4756ee440fe92d3294cc20826d53f8121cd26')}
    if a.method in assets:
        env,expected=assets[a.method];asset=Path(os.environ[env])
        if a.method=='aimnet':asset=asset/'aimnet2_wb97m_d3_0.pt'
        if digest(asset)!=expected:raise RuntimeError('STOP asset hash mismatch')
        files.append(asset)
    count=0
    with (a.out/'rows.jsonl').open('x') as output:
        for arm,batch,source,bb,xx,path in tasks(a.new):
            files.append(path)
            for c,(i,j) in enumerate(source['pairs']):
                p=byq[g1['queries'][i]['query_id']];pid=p['parent_id'];n=len(p['atomic_numbers']);seed=source['seed'];x=xx[c,:n].numpy()
                key=(arm,pid,seed,j)
                if a.method=='gfn2' and key in oldscore:
                    value=oldscore[key];s=dict(status='ok' if value is not None else 'failed',barrier=value)
                else:
                    if pid not in anchors:anchors[pid]=energy(method,p['atomic_numbers'],np.asarray(p['x_r']),cfg)
                    # Exact geometry-byte cache only; no tolerance or chemical equivalence shortcut.
                    cachekey=(pid,x.dtype.str,x.shape,x.tobytes())
                    if cachekey not in cache:cache[cachekey]=score(p['atomic_numbers'],x,anchors[pid],cfg,method)
                    s=cache[cachekey]
                output.write(json.dumps(dict(arm=arm,parent_id=pid,seed=seed,proposal=j,score=s),sort_keys=True,allow_nan=False)+'\n');count+=1
            output.flush();print('Scored '+arm+' '+batch,flush=True)
    if count!=21*62*3*32:raise RuntimeError('Incomplete endpoint coverage')
    manifest(a.out/'manifest.json','endpoint-score',started,files,dict(method=a.method,complete=True,n=count,rows_sha256=digest(a.out/'rows.jsonl'),unique_computed=len(cache)))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='stage',required=True)
    d=sub.add_parser('decode');d.add_argument('--new',type=Path,required=True);d.add_argument('--out',type=Path,required=True)
    s=sub.add_parser('acquire');s.add_argument('--new',type=Path,required=True);s.add_argument('--decoded',type=Path,required=True)
    s.add_argument('--method',choices=list(METHODS),required=True);s.add_argument('--out',type=Path,required=True)
    a=p.parse_args();globals()[a.stage](a)


if __name__=='__main__':main()
