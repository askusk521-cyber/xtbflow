"""Bounded full-screen acquisition after committed training tau.

16 CPU workers maximum, one heavy CPU task. Each worker has a hard cumulative
RLIMIT_CPU cap: combined worker allocation <=48 core-hours, leaving 12 for
training, planning, tests and analysis within the handoff's 60-core-hour limit.
No decoding or catalogue matching is performed in the AIMNet environment.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
import resource
import subprocess
import time

import numpy as np
from followup_enum_gate import b2f2_mask
from followup_thermo import single_energy, file_hash


def initialize(cpu_limit):
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
    import torch
    torch.set_num_threads(1)


def acquire_parent(task):
    parent, enumdir, outdir, cfg = task
    pid=parent['parent_id']
    out=Path(outdir)/(pid+'.jsonl')
    if out.exists():raise FileExistsError(out)
    cpu0=time.process_time();t0=time.perf_counter()
    anchor=single_energy(parent['atomic_numbers'],parent['b_r'],pid,'reactant',cfg)
    failures=0;count=0
    with np.load(Path(enumdir)/(pid+'.npz')) as data, out.open('x') as handle:
        indices=np.flatnonzero(b2f2_mask(data['b'],parent['b_r']))
        for i in indices:
            channel=str(data['channels'][i])
            product=single_energy(parent['atomic_numbers'],data['b'][i],pid,channel,cfg)
            ok=product['energy_kcal'] is not None and anchor['energy_kcal'] is not None
            row=dict(parent_id=pid,channel_id=channel,product=product,reactant=anchor,
                     delta_e_kcal=product['energy_kcal']-anchor['energy_kcal'] if ok else None)
            handle.write(json.dumps(row,sort_keys=True)+'\n')
            count+=1;failures+=not ok
    return dict(parent_id=pid,rows=count,failures=failures,wall_s=time.perf_counter()-t0,
                cpu_s=time.process_time()-cpu0,output_sha256=file_hash(out))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--enumdir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--gate',type=Path,required=True)
    p.add_argument('--budget',type=Path,required=True)
    a=p.parse_args()
    if not 1<=a.workers<=16:raise ValueError('Worker limit 16')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Explicitly disable CUDA')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('Dirty source')
    tau=Path('configs/explore/enum_thermo_tau.json')
    if not tau.exists():raise RuntimeError('Training tau must be frozen before screen')
    gate=json.loads(a.gate.read_text());budget=json.loads(a.budget.read_text())
    if not gate['passed'] or budget['stop']:raise RuntimeError('Gate/budget stop')
    if budget['selection']!='full':raise RuntimeError('This version implements only authorized full-screen plan')
    model=Path(os.environ['REVIEW_AIMNET_MODELS'])/'aimnet2_wb97m_d3_0.pt'
    if file_hash(model)!='f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28':raise RuntimeError('Model hash mismatch')
    parents_path=a.root/'data/parent_catalog.json';freeze=a.root/'formal/freeze.json'
    parents={r['parent_id']:r for r in json.loads(parents_path.read_text())}
    ids=sorted(json.loads(freeze.read_text())['parent_ids'])
    if len(ids)!=342:raise ValueError('Expected full 342-parent screen')
    for pid in ids:
        path=a.enumdir/(pid+'.npz')
        if file_hash(path)!=gate['input_sha256'][str(path)]:raise RuntimeError('Enum input hash mismatch')
    cfgpath=Path('configs/review/oracle_benchmark.json')
    cfg=json.loads(cfgpath.read_text())['xtb']
    a.out.mkdir(exist_ok=False)
    limit=int(48*3600/a.workers)
    manifest=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dirty=False,
                  config_sha256=file_hash('configs/explore/enum_thermo.json'),tau_sha256=file_hash(tau),
                  input_sha256={str(f.resolve()):file_hash(f) for f in (tau,a.gate,a.budget,parents_path,freeze,cfgpath,model)},
                  enum_input_sha256=gate['input_sha256'],slurm_job_id=None,workers=a.workers,
                  worker_cpu_cap_s=limit,aggregate_worker_cpu_cap_s=limit*a.workers,started_unix=time.time())
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize,initargs=(limit,)) as pool:
        futures=[pool.submit(acquire_parent,(parents[pid],str(a.enumdir),str(a.out),cfg)) for pid in ids]
        for future in as_completed(futures):
            row=future.result()
            if row['rows']!=gate['counts'][row['parent_id']]:raise RuntimeError('Incomplete b2f2 rows')
            results.append(row)
            (a.out/'progress.json').write_text(json.dumps(sorted(results,key=lambda r:r['parent_id']),indent=2,sort_keys=True)+'\n')
            print(json.dumps(dict(completed=len(results),parent_id=row['parent_id'],cpu_s=row['cpu_s'])),flush=True)
    manifest.update(complete=True,ended_unix=time.time(),cpu_s=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
                    parent_results=sorted(results,key=lambda r:r['parent_id']))
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
