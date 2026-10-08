"""Independent repeated CPU rollout as smoke-only numerical reference."""
import copy
import json
import torch
import explore_smc as original


def baseline(a):
    repeat=copy.copy(a);repeat.out=a.out/'cpu_repeat'
    original.cmd_g1(repeat)
    manifest=json.loads((repeat.out/'g1_manifest.json').read_text())
    first=json.loads((a.out/'g1_manifest.json').read_text())
    ctx=original.context(a);cfg,base,root,cat,split,parents,refs,queries=ctx
    byq={p['query_id']:p for p in parents.values()};result=[];maxdiff=0.
    for name in manifest['batches']:
        d=torch.load(repeat.out/name,weights_only=False,map_location='cpu')
        old=torch.load(a.out/name,weights_only=False,map_location='cpu')
        assert d['pairs']==old['pairs']
        for key in ('b_trace','x_trace','b_hat','x_hat'):
            maxdiff=max(maxdiff,float((d[key]-old[key]).abs().max()))
            if not torch.equal(d[key],old[key]):raise RuntimeError('CPU repeat is not bitwise identical')
        for c,(i,j) in enumerate(d['pairs']):
            p=byq[queries[i]['query_id']];pid=p['parent_id'];n=len(p['atomic_numbers'])
            decoded=original.decode_endpoint(p,[r for r in refs if r['parent_id']==pid],d['b_trace'][-1,c],d['x_trace'][-1,c])
            anchor,status=original.xtb_energy_kcal(p['atomic_numbers'],p['x_r'],base['xtb'])
            if status!='ok':raise RuntimeError('CPU smoke anchor failed')
            energies=[None]*11
            for t,k in enumerate((30,35,40,45)):
                e,s=original.xtb_energy_kcal(p['atomic_numbers'],d['x_hat'][t,c,:n].numpy(),base['xtb'])
                energies[k//5]=e-anchor if s=='ok' else float('nan')
            result.append(dict(parent_id=pid,seed=d['seed'],proposal=j,events=[decoded['channel_id']],proxy_status=decoded['proxy_status'],xtb_delta_kcal=energies))
    original.dump(a.out/'cpu_repeat_check.json',dict(passed=True,max_state_difference=maxdiff,n=len(result),reference='fresh CPU rollout, same device and batch layout; smoke only',source_commit=original.git('rev-parse','HEAD')))
    return result
