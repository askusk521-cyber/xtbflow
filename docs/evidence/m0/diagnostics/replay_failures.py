"""Bounded diagnostic replay of joint seed 0, never resumes/writes formal runs."""
import copy, json, random, time, hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from xtbflow.m0.batching import T1xDataset,collate
from xtbflow.m0.t1x_data import T1xCache
from xtbflow.m0.model import ReactionFlowNet
from xtbflow.m0.flow import training_inputs,flow_loss
from m0_train import lr_at
import argparse
ap=argparse.ArgumentParser(); ap.add_argument('--source',required=True); ap.add_argument('--out',required=True); ap.add_argument('--run',required=True); a=ap.parse_args()
meta=json.loads((Path('/home/lhshen/xtbflow-runs/m0/full-20261007')/a.run/'run_meta.json').read_text()); role=meta['role']; seed=meta['seed']; bound={'joint_s2':10893,'geometry_s1':11447,'geometry_s2':10095}[a.run]
p=Path(a.out); p.mkdir(exist_ok=False)
cfg=json.loads((Path(a.source)/'configs/m0/base.json').read_text()); tc=meta['train']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
device=torch.device('cuda'); model=ReactionFlowNet(role,**meta['model_cfg']).to(device); ema=copy.deepcopy(model).eval()
for v in ema.parameters(): v.requires_grad_(False)
opt=torch.optim.AdamW(model.parameters(),lr=tc['lr'],weight_decay=tc['weight_decay'])
loader=DataLoader(T1xDataset(T1xCache(Path('/home/lhshen/data/t1x/t1x_m0_v1.npz')),0),batch_size=tc['batch_size'],shuffle=True,drop_last=True,collate_fn=collate,num_workers=tc['num_workers'],generator=torch.Generator().manual_seed(seed),persistent_workers=True)
gen=torch.Generator(device=device).manual_seed(seed+1); step=0; start=time.time()
def stat(x):
 return {'finite':bool(torch.isfinite(x).all()),'maxabs':float(x.detach().abs().max())}
def save(obj):
 (p/'result.json').write_text(json.dumps(obj,indent=2)); print(json.dumps(obj),flush=True)
while step<=bound:
 for batch in loader:
  batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
  for group in opt.param_groups: group['lr']=lr_at(step,tc)
  inputs,targets=training_inputs(role,batch,meta['sigma_b'],meta['sigma_x'],gen)
  capture=7451 in batch['index'].tolist() or step>=bound-1
  traces=[]; handles=[]
  if capture:
   def hook(name):
    def f(m,ins,out):
     vals=list(out.values()) if isinstance(out,dict) else (list(out) if isinstance(out,tuple) else [out])
     traces.append({'module':name,'outputs':[stat(x) for x in vals if isinstance(x,torch.Tensor) and x.is_floating_point()]})
    return f
   for name,m in model.named_modules():
    if name: handles.append(m.register_forward_hook(hook(name)))
  out=model(**inputs); losses=flow_loss(out,targets,batch['atom_mask'])
  for h in handles:h.remove()
  if not torch.isfinite(losses['total']):
   per=((out['x_vel']-targets['x_vel']).square()*batch['atom_mask'][...,None]).sum((1,2))
   bad=torch.nonzero(~torch.isfinite(per)).flatten().tolist()
   report={'phase':'forward_loss','step':step,'elapsed_s':time.time()-start,'indices':batch['index'].tolist(),'bad_rows':bad,'bad_indices':[int(batch['index'][k]) for k in bad],'t_bad':[float(inputs['t'][k]) for k in bad],'inputs':{k:stat(v) for k,v in inputs.items() if v.is_floating_point()},'losses':{k:float(v) for k,v in losses.items()},'traces':traces,'parameter_state':{k:stat(v) for k,v in model.named_parameters()}}
   # Save diagnostic-only pre-failure state and exact inputs for operator localization.
   torch.save({'model':model.state_dict(),'inputs':inputs,'targets':targets,'batch':batch,'step':step},p/'diagnostic_state.pt')
   save(report); raise SystemExit(0)
  opt.zero_grad(set_to_none=True); losses['total'].backward()
  badgrad=[k for k,v in model.named_parameters() if v.grad is not None and not torch.isfinite(v.grad).all()]
  if badgrad:
   save({'phase':'backward','step':step,'badgrad':badgrad,'indices':batch['index'].tolist(),'traces':traces}); raise SystemExit(0)
  gnorm=torch.nn.utils.clip_grad_norm_(model.parameters(),tc['grad_clip']); opt.step()
  badparam=[k for k,v in model.named_parameters() if not torch.isfinite(v).all()]
  if badparam:
   save({'phase':'optimizer','step':step,'badparam':badparam}); raise SystemExit(0)
  with torch.no_grad():
   for pe,v in zip(ema.parameters(),model.parameters()): pe.lerp_(v,1-tc['ema_decay'])
  if capture:
   with (p/'sample_visits.jsonl').open('a') as f:f.write(json.dumps({'step':step,'loss':float(losses['total']),'grad_norm':float(gnorm),'t7451':[float(inputs['t'][k]) for k in range(len(batch['index'])) if int(batch['index'][k])==7451],'traces':traces})+'\n')
  step+=1
  if step%1000==0: print('step',step,'seconds',time.time()-start,flush=True)
  if step>bound:break
save({'phase':'no_failure_within_bound','steps':step,'elapsed_s':time.time()-start})
