"""Read-only diagnostic of saved CPU smoke and previous GPU states."""
import json
from pathlib import Path
import hashlib
import torch

R=Path('/home/lhshen/xtbflow-runs/explore-smc-20261009')
old=R/'smc-2116d43';new=R/'strict-smoke-d878762'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    torch.set_num_threads(1)
    om=load(old/'g1_manifest.json');nm=load(new/'g1_manifest.json')
    inputs={};lookup={};detail=[]
    for name in om['batches']:
        d=torch.load(old/name,map_location='cpu',weights_only=False)
        if d['seed']!=0:continue
        inputs[str(old/name)]=sha(old/name)
        for c,(i,j) in enumerate(d['pairs']):
            qid=om['queries'][i]['query_id']
            lookup[qid,j]=(d,c,name)
    for name in nm['batches']:
        d=torch.load(new/name,map_location='cpu',weights_only=False);inputs[str(new/name)]=sha(new/name)
        for c,(i,j) in enumerate(d['pairs']):
            query=nm['queries'][i];qid=query['query_id'];n=len(query['atomic_numbers'])
            o,oc,oname=lookup[qid,j]
            assert query==next(q for q in om['queries'] if q['query_id']==qid)
            assert d['meta']['sha256']==o['meta']['sha256']
            for t,k in enumerate((30,35,40,45,50)):
                diff={}
                for key in ('b_trace','x_trace','b_hat','x_hat'):
                    x=d[key][t,c,:n];y=o[key][t,oc,:n]
                    if key.startswith('b'):x=x[:,:n];y=y[:,:n]
                    diff[key]=float((x-y).abs().max())
                detail.append(dict(query_id=qid,proposal=j,step=k,max_abs_difference=diff))
    summary={str(k):{key:max(r['max_abs_difference'][key] for r in detail if r['step']==k) for key in ('b_trace','x_trace','b_hat','x_hat')} for k in (30,35,40,45,50)}
    result=dict(scope='Read-only saved-state comparison, no new generator or quantum calls, no H or utility inspected',
                trajectories=len(detail)//5,query_payloads_equal=True,model_hashes_equal=True,
                max_abs_differences=summary,inputs=inputs,rows=detail,
                limitation='Initial states were not saved by G1; not compared. Platform and batch size vary together, so this does not isolate their causal contributions. Original gate remains failed.')
    out=R/'strict-saved-state-diagnostic.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','rows')},indent=2))

if __name__=='__main__':main()
