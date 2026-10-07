"""Measure frozen per-call costs at one common batch size and atom-bin size."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from xtbflow.m0.model import ReactionFlowNet
from xtbflow.v1.data import write_json
from xtbflow.v1.interfaces import query_from_parents
from xtbflow.v1.scores import BarrierHead


def timing(fn,repeats):
    for _ in range(5):fn()
    torch.cuda.synchronize()
    values=[]
    for _ in range(5):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):fn()
        end.record();torch.cuda.synchronize()
        values.append(start.elapsed_time(end)/repeats/1000)
    return float(np.median(values)),values


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--batch-size',type=int,default=64)
    a=ap.parse_args();cfg=json.loads(a.config.read_text());batch=a.batch_size
    if not torch.cuda.is_available():raise RuntimeError('allocated GPU required')
    device='cuda';torch.manual_seed(101)
    networks={k:ReactionFlowNet(role,**c,dual_time=k=='f').to(device).eval().requires_grad_(False)
              for k,role,c in [('f','joint',cfg['model']),('g','event',cfg['cascade_model']),
                                ('h','geometry',cfg['cascade_model'])]}
    networks.update({k:BarrierHead(k).to(device).eval().requires_grad_(False) for k in ('E','X')})
    bins=[];lower=1
    for n in [8,12,16,20,24]:
        # Common dense shape and batch across every network. Atom-count normalization
        # is an operational benchmark, not a physical molecular-energy calculation.
        q=query_from_parents([dict(query_id=f'benchmark_{i}',atomic_numbers=[6]*n,
                                  charge=0,multiplicity=1,x_r=torch.randn(n,3).tolist(),
                                  b_r=torch.zeros(n,n).tolist()) for i in range(batch)],device)
        b=q.b_r+torch.randn_like(q.b_r)*.1;x=q.x_r+torch.randn_like(q.x_r)*.2
        t=torch.full((batch,),.5,device=device);costs={};raw={}
        for name,net in networks.items():
            if name in ('f','g','h'):
                kwargs=dict(z=q.element_index,atom_mask=q.atom_mask,x_r=q.x_r,b_r=q.b_r,
                            x_cur=q.x_r if name=='g' else x,b_cur=q.b_r if name=='h' else b,t=t)
                if name=='f':kwargs['t_x']=t
                def fn(net=net,kwargs=kwargs):
                    with torch.no_grad():net(**kwargs)
            else:
                def fn(net=net):
                    with torch.no_grad():net(q,b,x,t,t)
            median,values=timing(fn,10);costs[name]=median;raw[name]=values
            if name in ('E','X'):
                # Benchmark forward+input backward; subtract a graph-building forward
                # benchmark (not no_grad forward) to isolate input backward time.
                target=(b if name=='E' else x).detach().clone().requires_grad_(True)
                def graph_forward(net=net,name=name,target=target):
                    return net(q,target if name=='E' else b,x if name=='E' else target,t,t)
                def backward():
                    out=graph_forward();torch.autograd.grad(out.sum(),target)
                fwd,fwd_raw=timing(graph_forward,10);both,both_raw=timing(backward,10)
                if both<=fwd:raise RuntimeError('input backward timing unresolved')
                costs[name]=fwd
                costs[name+'_back']=both-fwd
                raw[name+'_graph_forward']=fwd_raw;raw[name+'_forward_backward']=both_raw
        bins.append(dict(min_atoms=lower,max_atoms=n,batch_size=batch,
                         weights={k:v/costs['f'] for k,v in costs.items()},
                         seconds_per_batch=costs,raw_seconds_per_call=raw))
        lower=n+1
    write_json(a.out,dict(method='same_large_batch_gpu_time',bins=bins,torch=torch.__version__,
                          cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),
                          repeats_per_measurement=10,measurements=5,
                          b0_candidate_units=50,weights_frozen=False,
                          note='Freeze after development; inference batch size never changes these weights.'))
    print(json.dumps({str(r['max_atoms']):r['weights'] for r in bins},indent=2))


if __name__=='__main__':main()
