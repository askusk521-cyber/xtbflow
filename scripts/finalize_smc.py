"""Deterministic supplementary evidence; never changes the frozen smc-1 analysis.

prepare: materialize conditional restart states from the exact G2 formula, CPU only.
results: run frozen analysis in a clean source checkout, then append measured resource
and coverage evidence. Keep the original scientific JSON separately for comparison.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_smc import dump, git
from xtbflow.v1.sampler import subset_query
from xtbflow.v1.interfaces import query_from_parents
from xtbflow.v1.clocks import clock_grid
from xtbflow.v1.smc import restart_state


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU-only command requires CUDA_VISIBLE_DEVICES=')
    if git('status','--porcelain'):raise RuntimeError('clean pushed source required')
    if not git('branch','-r','--contains','HEAD'):raise RuntimeError('pushed source required')
    out=a.out;dest=out/'clone_initial_states';dest.mkdir(exist_ok=True)
    load=lambda name:json.loads((out/name).read_text())
    g1=load('g1_manifest.json');plans=load('plans.json');tb,tx=clock_grid('geometry_lead3',50)
    qa=query_from_parents(g1['queries'],'cpu');count=0;files={};noise_pair_checks=0;seen={}
    for name in g1['batches']:
        data=torch.load(out/name,map_location='cpu',weights_only=False)
        q=subset_query(qa,data['query_indices']);seed=data['seed'];meta=data['meta']
        for arm in sorted({p['arm'] for p in plans}):
            pp=[p for p in plans if p['batch']==name and p['arm']==arm]
            k=pp[0]['step'];t=[30,35,40,45].index(k)
            b=data['b_trace'][t].clone();x=data['x_trace'][t].clone();mapping=[]
            for p in pp:
                start=p['start'];destinations=[start+d for d,s in p['clones']];ancestors=[start+s for d,s in p['clones']]
                cq=subset_query(q,ancestors)
                state=restart_state(cq,data['b_hat'][t,ancestors],data['x_hat'][t,ancestors],
                                    [s for d,s in p['clones']],seed,k,float(tb[k]),float(tx[k]),meta['sigma_b'],meta['sigma_x'])
                b[destinations]=state.b;x[destinations]=state.x
                for i,((d,s),qid) in enumerate(zip(p['clones'],cq.query_id)):
                    key=(qid,seed,k,s)
                    fingerprint=hashlib.sha256(state.b[i].numpy().tobytes()+state.x[i].numpy().tobytes()).hexdigest()
                    if key in seen:
                        assert seen[key]==fingerprint;noise_pair_checks+=1
                    else:seen[key]=fingerprint
                    mapping.append(dict(parent_id=p['parent_id'],destination=start+d,ancestor=start+s))
                    count+=1
                survivor_slots=[start+s for s in p['survivors']]
                assert torch.equal(b[survivor_slots],data['b_trace'][t,survivor_slots])
                assert torch.equal(x[survivor_slots],data['x_trace'][t,survivor_slots])
            path=dest/f'{arm}-{name}'
            torch.save(dict(b=b,x=x,step=k,seed=seed,arm=arm,batch=name,clones=mapping),path)
            files[str(path.relative_to(out))]=sha(path)
    assert count==35712 and len(files)==288
    manifest=dict(stage='supplemental_clone_materialization',source_commit=git('rev-parse','HEAD'),dirty=False,
                  slurm_job_id=None,device='cpu',workers=1,config_sha256=sha('configs/explore/smc_selection.json'),
                  inputs={str(out/'plans.json'):sha(out/'plans.json'),str(out/'g1_manifest.json'):sha(out/'g1_manifest.json'),
                          **{str(out/n):sha(out/n) for n in g1['batches']}},
                  files=files,n_clones=count,n_batches=len(files),paired_noise_checks=noise_pair_checks,
                  timing='posthoc persistence of exact frozen G2 initial-state formula, not a new experiment')
    dump(out/'clone_initial_manifest.json',manifest)
    print('materialized and paired-audited clone initial states',flush=True)


def results(a):
    if git('status','--porcelain'):raise RuntimeError('clean source required')
    out=a.out;run=out.parent
    frozen=run/'source-2116d43'
    subprocess.run([sys.executable,'scripts/explore_smc.py','analyze','--out',str(out),'--workers','1'],cwd=frozen,check=True)
    original=(out/'results.json').read_bytes()
    if (run/'results-first.json').read_bytes()!=original:raise RuntimeError('frozen scientific result drift')
    result=json.loads(original)
    def time_values(path):
        values={}
        for line in path.read_text().splitlines():
            for label,key in [('User time (seconds):','user_seconds'),('System time (seconds):','system_seconds')]:
                if line.strip().startswith(label):values[key]=float(line.split(label)[1].strip())
        values['cpu_core_hours']=(values['user_seconds']+values['system_seconds'])/3600
        return values
    times={stage:time_values(run/f'{stage}-time.txt') for stage in ('c2','analyze')}
    gpu={}
    for job in (2699,2702):
        text=(run/f'slurm-{job}.txt').read_text()
        assert 'JobState=COMPLETED' in text and 'ExitCode=0:0' in text
        runtime=next(w.split('=')[1] for w in text.split() if w.startswith('RunTime='))
        h,m,s=map(int,runtime.split(':'));gpu[str(job)]=h*3600+m*60+s
    bounds={stage:48*result['resource_manifests'][stage]['elapsed_s']/3600 for stage in ('s0','c1')}
    coverage=json.loads((out/'coverage_audit.json').read_text())
    result['supplemental_evidence']=dict(
        source_commit=git('rev-parse','HEAD'),dirty=False,
        scientific_results_sha256=hashlib.sha256(original).hexdigest(),
        coverage_audit=coverage,
        clone_initial_manifest=json.loads((out/'clone_initial_manifest.json').read_text()),
        resource_accounting=dict(gpu_job_wall_seconds=gpu,gpu_total_wall_seconds=sum(gpu.values()),
            measured_cpu=times,early_stage_48_worker_wall_bounds_core_hours=bounds,
            early_bound_plus_measured_c2_core_hours=sum(bounds.values())+times['c2']['cpu_core_hours'],
            caveat='Early CPU values are 48-worker wall bounds, not process measurements; parent overhead, tests, GPU-host CPU and loading are not a measured total.'),
        sidecar_sha256={name:sha(run/name) for name in ('slurm-2699.txt','slurm-2702.txt','c2-time.txt','analyze-time.txt')},
        smoke_scope='Pre-formal CPU smoke covered two-parent G1 only, not the full stage chain; historical deviation retained.')
    dump(out/'results.json',result)
    print('supplemental deterministic results written',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('cmd',choices=['prepare','results']);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();torch.set_num_threads(1);globals()[a.cmd](a)


if __name__=='__main__':main()
