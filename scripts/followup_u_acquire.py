"""T2a physical Hessian acquisition or real optimized-water validation.

CPU only. Run serially after other heavy tasks. GFN2 gradients use the existing
_force_job conversion; AIMNet2 forces use eV/Angstrom. No protocol tuning.
Outputs raw Hessian NPZ separately from JSON diagnostics for independent replay.
"""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from followup_u_math import finite_difference_hessian, hessian_metrics, internal_basis
from explore_geometry_information import _force_job
from xtbflow.v1.xtb_score import gradient_to_force
from xtbflow.v1.explore_rollout import event_direction
from review_oracle_benchmark import energy


class Gradient:
    def __init__(self,z,method,cfg):
        self.z=np.asarray(z);self.method=method;self.cfg=cfg;self.calls=0
        self.calculator=None
        if method=='AIMNet2':
            from aimnet.calculators import AIMNet2Calculator
            self.calculator=AIMNet2Calculator(str(Path(os.environ['REVIEW_AIMNET_MODELS'])/'aimnet2_wb97m_d3_0.pt'),device='cpu')

    def __call__(self,x):
        self.calls+=1
        if self.method=='GFN2-xTB':
            g,status=_force_job((self.z,x,self.cfg))
            if status!='ok':raise RuntimeError('gradient:'+status)
            return -gradient_to_force(g)
        result=self.calculator({'coord':torch.tensor(np.asarray(x),dtype=torch.float32),
                                'numbers':torch.tensor(self.z,dtype=torch.int64),
                                'charge':torch.tensor([0.])},forces=True)
        return -result['forces'].detach().cpu().numpy().reshape(-1,3)*23.060547830619


def direction(bp,br,x):
    n=len(x)
    u,valid=event_direction(torch.tensor(np.asarray(bp)[None],dtype=torch.float64),
                            torch.tensor(np.asarray(br)[None],dtype=torch.float64),
                            torch.tensor(np.asarray(x)[None],dtype=torch.float64),
                            torch.ones((1,n),dtype=torch.bool))
    if not valid.item():raise ValueError('Undefined event direction')
    return u.numpy()[0]


def water_test(cfg):
    from scipy.optimize import minimize
    z=np.array([8,1,1]);x=np.array([[0.,0.,0.],[.95,0.,0.],[-.2,.9,0.]])
    gradient=Gradient(z,'GFN2-xTB',cfg)
    def objective(flat):
        coords=flat.reshape(-1,3)
        value,status=energy('GFN2-xTB',z,coords,cfg)
        if status!='ok':raise RuntimeError(status)
        return value,gradient(coords).ravel()
    opt=minimize(objective,x.ravel(),method='BFGS',jac=True,options={'gtol':1e-4,'maxiter':300})
    if not opt.success and np.max(np.abs(opt.jac))>=1e-4:raise RuntimeError('Water optimization did not converge')
    x=opt.x.reshape(-1,3)
    h=finite_difference_hessian(x,gradient)
    q,rank=internal_basis(x)
    metrics=hessian_metrics(z,x,h,q[:,0])
    assert rank==6
    assert len(metrics['frequencies_cm1'])==3
    assert min(metrics['frequencies_cm1'])>0
    assert np.array_equal(h,h.T)
    return dict(passed=True,method='GFN2-xTB',x=x.tolist(),hessian=h.tolist(),gradient_max=float(np.max(np.abs(opt.jac))),
                metrics=metrics,gradient_calls=gradient.calls)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['water','references'])
    p.add_argument('--method',choices=['GFN2-xTB','AIMNet2'],default='GFN2-xTB')
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-cpu-seconds',type=int,required=True)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Disable CUDA explicitly')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('Dirty source')
    torch.set_num_threads(1)
    resource.setrlimit(resource.RLIMIT_CPU,(a.max_cpu_seconds,a.max_cpu_seconds))
    a.out.mkdir(exist_ok=False)
    cfgpath=Path('configs/review/oracle_benchmark.json')
    cfg=json.loads(cfgpath.read_text())['xtb']
    inputs=[cfgpath,Path('configs/explore/u_diagnostic.json')]
    started=time.time();cpu0=time.process_time()
    manifest=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dirty=False,
                  config_sha256=sha256(inputs[1].read_bytes()).hexdigest(),slurm_job_id=None,
                  method=a.method,stage=a.stage,started_unix=started,cpu_cap_s=a.max_cpu_seconds)
    if a.method=='AIMNet2':
        model=Path(os.environ['REVIEW_AIMNET_MODELS'])/'aimnet2_wb97m_d3_0.pt'
        assert sha256(model.read_bytes()).hexdigest()=='f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28'
        inputs.append(model)
    if a.stage=='water':
        result=water_test(cfg)
        (a.out/'water.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    else:
        parentfile=a.root/'data/parent_catalog.json';reffile=a.root/'data/reference_catalog.jsonl'
        checkfile=Path('docs/evidence/explore_geoinfo/check_a_rows.jsonl')
        splitfile=a.root/'data/split_manifest.json';inputs.extend([parentfile,reffile,checkfile,splitfile])
        parents={r['parent_id']:r for r in json.loads(parentfile.read_text())}
        refs={r['reference_id']:r for r in map(json.loads,reffile.open())}
        check=list(map(json.loads,checkfile.open()))
        dev=set(json.loads(splitfile.read_text())['development']['parent_ids'])
        assert len(check)==558 and len({r['reference_id'] for r in check})==558
        with (a.out/'rows.jsonl').open('x') as output:
            for selected in check:
                r=refs[selected['reference_id']];pid=r['parent_id'];assert pid in dev
                parent=parents[pid];x=np.asarray(r['x_ts']);z=np.asarray(parent['atomic_numbers'])
                start=time.perf_counter()
                gradient=Gradient(z,a.method,cfg)
                record=dict(reference_id=r['reference_id'],parent_id=pid,split_group=parent['split_group'])
                try:
                    u=direction(r['b_p'],parent['b_r'],x)
                    h=finite_difference_hessian(x,gradient)
                    metrics=hessian_metrics(z,x,h,u)
                    d=np.asarray(r['b_p'])-np.asarray(parent['b_r']);i,j=np.triu_indices(len(z),1)
                    metrics['event_size']=int(np.abs(d[i,j]).sum())
                    key=sha256(r['reference_id'].encode()).hexdigest()[:16]
                    np.savez(a.out/(key+'.npz'),z=z,x=x,u=u,h=h)
                    record.update(status='ok',metrics=metrics,hessian_file=key+'.npz')
                except (ValueError,RuntimeError) as exc:
                    record.update(status='failed',reason=type(exc).__name__+':'+str(exc))
                record.update(gradient_calls=gradient.calls,wall_s=time.perf_counter()-start)
                output.write(json.dumps(record,sort_keys=True)+'\n');output.flush()
    manifest.update(complete=True,cpu_s=time.process_time()-cpu0,wall_s=time.time()-started,
                    input_sha256={str(path):sha256(path.read_bytes()).hexdigest() for path in inputs})
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print('Acquisition complete; independent analysis still required.')


if __name__=='__main__':main()
