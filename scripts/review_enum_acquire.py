"""Exact enumeration acquisition and frozen GPU scoring in separate modes."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from rdkit import RDLogger
from xtbflow.v1.enumeration import enumerate_events
from xtbflow.v1.data import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['enumerate','score'])
    p.add_argument('--root',type=Path,default=Path('/home/lhshen/xtbflow-runs/v1a-20261007'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--size',type=int,choices=[2,3],required=True)
    p.add_argument('--split',choices=['train','development','screen_reserve'],required=True)
    p.add_argument('--index',type=int);a=p.parse_args()
    manifest=json.loads((a.root/'data/split_manifest.json').read_text())
    ids=sorted(manifest[a.split]['parent_ids'])
    if a.index is not None:ids=ids[a.index:a.index+1]
    parents={r['parent_id']:r for r in json.loads((a.root/'data/parent_catalog.json').read_text()) if r['parent_id'] in ids}
    refs={pid:[] for pid in ids}
    with (a.root/'data/reference_catalog.jsonl').open() as handle:
        for line in handle:
            r=json.loads(line)
            if r['parent_id'] in refs:refs[r['parent_id']].append(r)
    a.out.mkdir(parents=True,exist_ok=True)
    RDLogger.DisableLog('rdApp.*')
    if a.mode=='score':
        import torch
        from xtbflow.v1.assets import load_scores
        from xtbflow.v1.scores import BarrierHead,BarrierEnsemble
        from xtbflow.v1.interfaces import query_from_parents
        assert torch.cuda.is_available()
        e,_=load_scores(a.root,'E','cuda')
        members=[]
        for seed in (101,102,103):
            obj=torch.load(f'/home/lhshen/xtbflow-runs/explore-geoinfo-20261008/full-68f5733/heads/clean_heads/reactant_{seed}.pt',map_location='cuda',weights_only=False)
            head=BarrierHead('X',**obj['meta']['score_config']).to('cuda');head.load_state_dict(obj['ema']);members.append(head)
        n=BarrierEnsemble(members,obj['meta']['median'],obj['meta']['scale'])
    for pid in ids:
        parent=parents[pid];path=a.out/(pid+'.npz');meta=a.out/(pid+'.json')
        if a.mode=='enumerate':
            if path.exists() and meta.exists():continue
            start=time.monotonic();events=dict(enumerate_events(parent['atomic_numbers'],parent['b_r'],parent['permutations'],a.size))
            channels=sorted(events)
            np.savez_compressed(path,channels=np.array(channels),b=np.array([events[c] for c in channels],dtype=np.int8))
            known={r['channel_id'] for r in refs[pid]};best=set(parent['best_channel_ids'])
            write_json(meta,dict(parent_id=pid,split=a.split,size=a.size,n_enum=len(channels),
                covered_channels=sorted(known&events.keys()),catalogue_channels=sorted(known),
                best_channels=sorted(best),covered_best=sorted(best&events.keys()),seconds=time.monotonic()-start))
            print(pid,len(channels),time.monotonic()-start,flush=True)
        else:
            target=a.out/(pid+'.scores.npz')
            if target.exists():continue
            data=np.load(path);b=data['b'];scores={'S_E':[],'S_N':[]}
            with torch.no_grad():
                for start in range(0,len(b),256):
                    batch=b[start:start+256];q=query_from_parents([parent]*len(batch),'cuda')
                    tensor=torch.tensor(batch,dtype=torch.float32,device='cuda');t=torch.ones(len(batch),device='cuda')
                    for key,ens in [('S_E',e),('S_N',n)]:
                        scores[key].extend(ens.components(q,tensor,q.x_r,t,t)['mean_kcal'].cpu().tolist())
            np.savez_compressed(target,**{k:np.array(v) for k,v in scores.items()})
            print('scored',pid,len(b),flush=True)


if __name__=='__main__':main()
