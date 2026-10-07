"""Independent read-only input/forward diagnostic; no training or artifact mutation."""
import json, random, argparse
from pathlib import Path
import numpy as np
import torch
from xtbflow.m0.t1x_data import T1xCache
from xtbflow.m0.batching import collate
from xtbflow.m0.model import ReactionFlowNet
from xtbflow.m0.flow import training_inputs, flow_loss

ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--out',required=True); args=ap.parse_args()
torch.set_num_threads(2)
c=T1xCache(Path('/home/lhshen/data/t1x/t1x_m0_v1.npz'))
assert c.split[7451]==0
cfg=json.loads(Path(args.config).read_text()); batch=collate([c.reaction(7451)])
rows=[]
for role in ('joint','geometry'):
 for seed in (0,1,2):
  random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
  model=ReactionFlowNet(role,**cfg['model' if role=='joint' else 'cascade_model'])
  gen=torch.Generator().manual_seed(seed+1)
  inputs,targets=training_inputs(role,batch,1.0 if role=='joint' else .5,.5,gen)
  # Sweep shared interpolation t while preserving the generated start state.
  t_old=inputs['t'].clone(); x0=(inputs['x_cur']-t_old[:,None,None]*batch['x_ts'])/(1-t_old[:,None,None])
  b0=(inputs['b_cur']-t_old[:,None,None]*batch['b_p'])/(1-t_old[:,None,None]) if role=='joint' else None
  for t in (0.,.25,.5,.75,.9,.99,1.):
   inputs['t']=torch.tensor([t]); inputs['x_cur']=(1-t)*x0+t*batch['x_ts']
   if role=='joint': inputs['b_cur']=(1-t)*b0+t*batch['b_p']
   traces=[]; handles=[]
   def hook(name):
    def record(module, inp, out):
     vals=list(out.values()) if isinstance(out,dict) else (list(out) if isinstance(out,tuple) else [out])
     traces.append({'module':name,'outputs':[{'finite':bool(torch.isfinite(v).all()),'maxabs':float(v.detach().abs().max())} for v in vals if isinstance(v,torch.Tensor) and v.dtype!=torch.bool]})
    return record
   for name,m in model.named_modules():
    if name: handles.append(m.register_forward_hook(hook(name)))
   with torch.no_grad(): out=model(**inputs); loss=flow_loss(out,targets,batch['atom_mask'])
   for h in handles: h.remove()
   rows.append({'role':role,'seed':seed,'t':t,'loss':{k:float(v) for k,v in loss.items()},'first_nonfinite':next((v for v in traces if any(not x['finite'] for x in v['outputs'])),None),'layers':[v for v in traces if v['module'].startswith('trunk.layers.') and v['module'].count('.')==2]})
Path(args.out).write_text(json.dumps({'sample':7451,'scope':'initial weights, single training sample, forward only; not exact failed trained state','rows':rows},indent=2))
for r in rows:
 print(r['role'],r['seed'],r['t'],r['loss'],'first_nonfinite',r['first_nonfinite'])
