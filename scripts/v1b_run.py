"""V1b formal execution over the frozen samples (guide Sections 5, 9, 10).

queue references : anchor Opt+Freq and native best-reference chains for every
                   parent needed by Stage A (strict_low) or Stage B (best_cert_hit)
queue stage-a    : full strict chain for every sampled Stage-A candidate (invalid: no QC)
queue stage-b    : per sampled parent and arm A2/B1, the frozen C* order with
                   witness stop and the energy short circuit
--check-only     : refuse unless GO, frozen samples and a validated D1 protocol exist

Work is sharded by --worker/--workers; finished items (verdict.json) are skipped,
so a resumed queue never creates new statistical units.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np

KCAL=627.509474
EPS_CERT=1.0
LOW_WINDOW=2.0
TOL=1e-6


def jsonl(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]


def shard(key,worker,workers):return int(sha256(key.encode()).hexdigest(),16)%workers==worker


class Source:
    def __init__(self,run,v1a_root):
        self.manifest=json.loads((run/'manifest.json').read_text())
        if not self.manifest.get('samples_frozen'):raise ValueError('samples not frozen; run v1b_freeze first')
        if not self.manifest.get('d1_protocol_validated'):raise ValueError('D1 protocol not recorded as validated')
        cat=v1a_root/'data'
        self.parents={p['parent_id']:p for p in json.loads((cat/'parent_catalog.json').read_text())}
        self.by_query={p['query_id']:p for p in self.parents.values()}
        self.native=[r for r in jsonl(run/'native_references.jsonl')]
        self.frame={r['unit_id']:r for r in jsonl(run/'source_candidates.jsonl')}
        self.bundles={(b['parent_id'],b['arm']):b for b in jsonl(run/'source_bundles.jsonl')}
        self.raw={};self.labels={}
        for line in (v1a_root/'screen/efficiency_s0/streams.jsonl').read_text().splitlines():
            row=json.loads(line)
            for c in row['candidates']:self.raw[c['candidate_id']]=c
        for line in (v1a_root/'screen/efficiency_s0.candidates.jsonl').read_text().splitlines():
            c=json.loads(line);self.labels[c['candidate_id']]=c

    def candidate_inputs(self,cid):
        c=self.raw[cid];p=self.by_query[cid.split('/')[0]]
        return np.asarray(p['atomic_numbers']),np.asarray(c['x'],dtype=float),p['b_r'],p['permutations'],c['b_dec']


def ref_catalogue(v1a_root,pids):
    out={}
    for line in (v1a_root/'data/reference_catalog.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['parent_id'] in pids:out[r['reference_id']]=r
    return out


def threshold(work,pid,src):
    """T_H (kcal/mol) from a complete anchor + all original best references, else None."""
    from xtbflow.v1.qc_protocol import perceive_be
    a=work/'references'/pid/'anchor'/'verdict.json'
    if not a.exists():return None,'PENDING'
    anchor=json.loads(a.read_text());p=src.parents[pid]
    if anchor['status'] not in ('MINIMUM_PASS','MINIMUM_GRAY_ZONE'):return None,'ANCHOR_UNRESOLVED'
    be=perceive_be(p['atomic_numbers'],np.asarray(anchor['x']));perms=np.asarray(p['permutations'])
    if be is None or not np.all(be[perms[:,:,None],perms[:,None,:]]==np.asarray(p['b_r']),axis=(1,2)).any():
        return None,'ANCHOR_GRAPH_CHANGED'
    values=[]
    for r in [r for r in src.native if r['parent_id']==pid]:
        f=work/'references'/pid/r['reference_id'].replace('/','_')/'verdict.json'
        if not f.exists():return None,'PENDING'
        v=json.loads(f.read_text())
        if v.get('strict_joint_graph')!=1:return None,'REFERENCE_UNRESOLVED'
        values.append((v['ts']['energy_hartree']-anchor['energy_hartree'])*KCAL)
    return min(values),'COMPLETE'


def run_item(work,fn):
    if (work/'verdict.json').exists():return json.loads((work/'verdict.json').read_text())
    work.mkdir(parents=True,exist_ok=True)
    return fn()


def queue_references(a,src,method):
    from xtbflow.v1 import qc_protocol as qc
    pids=sorted({r['parent_id'] for r in jsonl(a.run/'sample_B_parents.jsonl')}|
                {src.frame[r['unit_id']]['parent_id'] for r in jsonl(a.run/'sample_A.jsonl')})
    refs=ref_catalogue(a.v1a_root,set(pids))
    for pid in pids:
        if not shard(pid,a.worker,a.workers):continue
        p=src.parents[pid];z=np.asarray(p['atomic_numbers'])
        wd=a.run/'work'/'references'/pid
        def anchor():
            meter=qc.Meter();t0=time.perf_counter()
            out=qc.minimum_stage(z,np.asarray(p['x_r']),method,meter,wd/'anchor','anchor')
            return qc.finish(out,meter,t0,wd/'anchor')
        run_item(wd/'anchor',anchor)
        perm=np.asarray(p['permutations'][0])
        for rid in p['best_reference_ids']:
            r=refs[rid];d=wd/rid.replace('/','_')
            run_item(d,lambda:qc.certification_chain(z,np.asarray(r['x_ts'])[perm],p['b_r'],p['permutations'],
                                                      np.asarray(r['b_p'])[np.ix_(perm,perm)],method,d))
        print(json.dumps(dict(parent=pid,threshold=threshold(a.run/'work',pid,src))),flush=True)


def queue_stage_a(a,src,method):
    from xtbflow.v1 import qc_protocol as qc
    for r in jsonl(a.run/'sample_A.jsonl'):
        cid=r['unit_id']
        if not shard(cid,a.worker,a.workers) or r['proxy_class']=='INVALID_OUTPUT':continue
        z,x,br,perms,bp=src.candidate_inputs(cid);d=a.run/'work'/'candidates'/sha256(cid.encode()).hexdigest()[:20]
        if bp is None:continue
        v=run_item(d,lambda:qc.certification_chain(z,x,br,perms,bp,method,d))
        print(json.dumps(dict(stage='A',candidate=cid,status=v.get('strict_status'))),flush=True)


def stage_b_order(src,pid,arm):
    b=src.bundles[pid,arm];cands=b['best_event_candidate_ids']
    first=[c for c in cands if src.labels[c]['hits_best_reference']]
    rest=sorted([c for c in cands if c not in first],key=lambda c:src.labels[c]['completion_units'])
    return sorted(first,key=lambda c:src.labels[c]['completion_units'])+rest


def best_cert(v,T):
    """Candidate best_cert_hit interval from its verdict and the parent threshold T (kcal/mol)."""
    if v is None:return (0,1)
    if v.get('strict_status')=='ENERGY_EXCLUDED_FOR_BEST':return (0,0)
    if v.get('strict_joint_graph')==0:return (0,0)
    if v.get('strict_joint_graph') is None:return (0,1)
    if T is None:return (0,1)
    L=v.get('relative_energy_anchor_kcal')
    return (1,1) if L is not None and L<=T+EPS_CERT+TOL else (0,0)


def queue_stage_b(a,src,method):
    from xtbflow.v1 import qc_protocol as qc
    for row in jsonl(a.run/'sample_B_parents.jsonl'):
        pid=row['unit_id']
        if not shard(pid,a.worker,a.workers) or row['stratum']=='neither_empty':continue
        T,status=threshold(a.run/'work',pid,src)
        if T is None:
            print(json.dumps(dict(stage='B',parent=pid,threshold=status)),flush=True);continue
        anchor=json.loads((a.run/'work'/'references'/pid/'anchor'/'verdict.json').read_text())
        exclusion=anchor['energy_hartree']+(T+EPS_CERT+TOL)/KCAL
        for arm in ('A2','B1'):
            for cid in stage_b_order(src,pid,arm):
                z,x,br,perms,bp=src.candidate_inputs(cid)
                d=a.run/'work'/'candidates'/sha256(cid.encode()).hexdigest()[:20]
                v=run_item(d,lambda:qc.certification_chain(z,x,br,perms,bp,method,d,energy_exclusion_hartree=exclusion))
                if 'ts' in v and 'energy_hartree' in v['ts']:
                    v['relative_energy_anchor_kcal']=(v['ts']['energy_hartree']-anchor['energy_hartree'])*KCAL
                if best_cert(v,T)==(1,1):break
            print(json.dumps(dict(stage='B',parent=pid,arm=arm)),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--v1a-root',type=Path,required=True)
    ap.add_argument('--queue',choices=['references','stage-a','stage-b'])
    ap.add_argument('--worker',type=int,default=0);ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--check-only',action='store_true')
    a=ap.parse_args()
    src=Source(a.run,a.v1a_root)
    if src.manifest.get('delta_proxy') is None or not src.manifest.get('decision_rule_frozen'):
        raise ValueError('Delta^proxy and the V1b decision rule must be frozen before H work')
    if a.check_only:
        print(json.dumps(dict(ok=True,stage=src.manifest['stage']),indent=1));return
    from xtbflow.v1.qc_protocol import Method
    method=Method(device='gpu')
    dict(references=queue_references,**{'stage-a':queue_stage_a,'stage-b':queue_stage_b})[a.queue](a,src,method)


if __name__=='__main__':main()
