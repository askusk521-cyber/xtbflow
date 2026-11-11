#!/usr/bin/env python3
"""#198: exactly three retained states, at most 3 E/F and 45 trained forwards."""
from __future__ import annotations

import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''

import argparse
from dataclasses import asdict, replace
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import socket
import subprocess
import sys
import types

import torch

from xtbflow.sampling.integrated_guidance import FlowState, prepare_injection
from xtbflow.sampling.physical_diagnostic import (StageMeter, displacement_metrics, evaluate_physical,
                                                measure_direction, physical_fields)
from xtbflow.validation.certification import file_hash, identity, write_json


def load_frozen(args, cfg, root):
    import xtbflow
    frozen=json.loads((root/'configs/research/narrowed_scope_20261111.json').read_text())
    recovery=json.loads((root/'configs/research/numerical_recovery_20261111.json').read_text())
    source=args.research_source.resolve()
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=source).decode().strip()!=frozen['trained']['research_commit']:
        raise ValueError('research_commit_mismatch')
    if subprocess.check_output(['git','status','--porcelain'],cwd=source).strip():
        raise ValueError('dirty_research_source')
    for name,sha in cfg['trained_files'].items():
        if file_hash(source/name)!=sha:
            raise ValueError('trained_file_mismatch:'+name)
    if file_hash(args.packet)!=frozen['diagnostic']['worker_packet_sha256']:
        raise ValueError('packet_hash_mismatch')
    packet=json.loads(args.packet.read_text())
    if len(packet['queries'])!=1 or set(packet['queries'][0])!={'query_id','atomic_numbers','x_r','b_r','charge','multiplicity'}:
        raise ValueError('input_whitelist_mismatch')
    if file_hash(args.states/'numerical_diagnostics.json')!=recovery['prior_report_sha256']:
        raise ValueError('saved_state_receipt_mismatch')
    for k,sha in recovery['saved_state_sha256'].items():
        if file_hash(args.states/f'state_{k}.pt')!=sha:
            raise ValueError('saved_state_hash_mismatch')
    checkpoint=args.model_root/'generators/joint_s0/ckpt_60000.pt'
    expected=frozen['trained']['checkpoints']['joint']
    if file_hash(checkpoint)!=expected['sha256'] or file_hash(checkpoint.parent/'status.json')!=expected['training_status_sha256']:
        raise ValueError('checkpoint_identity_mismatch')
    # Only the isolated m0/v1 namespace is added; integrated modules retain precedence.
    xtbflow.__path__.append(str(source/'src/xtbflow'))
    vocabulary=types.ModuleType('xtbflow.m0.batching')
    vocabulary.ELEMENT_INDEX={1:1,6:2,7:3,8:4};vocabulary.N_ELEMENT_TYPES=5
    sys.modules[vocabulary.__name__]=vocabulary
    from xtbflow.m0.model import ReactionFlowNet
    from xtbflow.v1.interfaces import query_from_parents
    obj=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=ReactionFlowNet('joint',**obj['meta']['model_cfg'],dual_time=obj['meta']['dual_time'])
    model.load_state_dict(obj['ema']);model.eval().requires_grad_(False).double()
    query=query_from_parents(packet['queries'],'cpu')
    query=replace(query,x_r=query.x_r.double(),b_r=query.b_r.double())
    states={}
    for k in cfg['states']:
        # Only hash-verified local historical tensors; old clocks include NumPy arrays.
        saved=torch.load(args.states/f'state_{k}.pt',map_location='cpu',weights_only=False)
        states[k]=FlowState(*(torch.as_tensor(saved[name],dtype=torch.float64) for name in ('b','x','t_b','t_x')))
        if states[k].b.shape!=(1,16,16) or not torch.equal(query.atom_mask,torch.ones_like(query.atom_mask)):
            raise ValueError('frozen_atom_pool_mismatch')
    return model,query,states,dict(packet_sha256=file_hash(args.packet),checkpoint_sha256=file_hash(checkpoint),
                                 saved_state_sha256=recovery['saved_state_sha256'],trained_commit=frozen['trained']['research_commit'])


def run(args):
    root=Path(__file__).resolve().parents[1]
    config=root/'configs/research/physical_direction_198_20261111.json'
    cfg=json.loads(config.read_text())
    args.out.mkdir(parents=True,exist_ok=False)
    meter=StageMeter(args.out,cfg['caps'])
    resource.setrlimit(resource.RLIMIT_CPU,(cfg['caps']['cpu_seconds'],)*2)
    resource.setrlimit(resource.RLIMIT_FSIZE,(cfg['caps']['output_bytes'],)*2)
    torch.set_num_threads(1)
    report=dict(schema='xtbflow-real-direction-diagnostic/v1',issue=198,host=socket.gethostname(),
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root).decode().strip(),
                config_sha256=file_hash(config),python=platform.python_version(),
                packages={p:importlib.metadata.version(p) for p in ['torch','numpy','scipy','tblite','rdkit']},
                caps=cfg['caps'],job_id=os.environ.get('SLURM_JOB_ID'),states=[dict(step=k,status='not_run',directions={}) for k in cfg['states']],
                status='running',claim_limits=cfg['claim_limits'])
    try:
        if subprocess.check_output(['git','status','--porcelain'],cwd=root).strip():
            raise ValueError('dirty_integrated_source')
        from xtbflow.calculators.xtb_oracle import _package_identity, _adapter_code_hash
        version,_,build,_=_package_identity('tblite')
        if (version,build,_adapter_code_hash())!=(cfg['calculator']['version'],cfg['calculator']['build_sha256'],cfg['calculator']['adapter_sha256']):
            raise ValueError('calculator_preflight_identity_mismatch')
        model,query,states,identities=load_frozen(args,cfg,root)
        report['identities']=identities
        meter.install_model(model)
        for row in report['states']:
            k=row['step'];state=states[k]
            row['status']='started'
            # Match the pinned float64 clock grid, without reconstructing any trajectory.
            next_b=(k+1)/50;next_x=next_b**2
            with torch.no_grad():
                injection=prepare_injection(model,query,state,next_b,next_x)
            distance=torch.cdist(injection.predicted_x,injection.predicted_x)[0]
            distance.fill_diagonal_(float('inf'))
            row.update(old_t_b=float(state.t_b[0]),old_t_x=float(state.t_x[0]),next_t_b=next_b,next_t_x=next_x,
                       dt_x=injection.dt_x,min_distance_angstrom=float(distance.min()))
            if row['min_distance_angstrom']<=cfg['minimum_distinct_distance_angstrom']:
                row.update(status='invalid_coincident_geometry',directions={n:dict(status='unavailable_geometry') for n in ('F','G')})
                continue
            physical=evaluate_physical(injection.predicted_x,query.atomic_numbers,cfg['calculator'],meter,f'k{k}:old_endpoint')
            row['physical_status']=physical['status']
            if physical['status']!='completed':
                row.update(status='physical_signal_unavailable',directions={n:dict(status=physical['status']) for n in ('F','G')})
                continue
            force=torch.tensor(physical['force'],dtype=torch.float64)[None]
            force,reflected,u,valid=physical_fields(force,injection.predicted_b,query.b_r,injection.predicted_x,query.atom_mask)
            row.update(energy_hartree=physical['energy'],force_norm=float(force.norm()),reflected_norm=float(reflected.norm()),
                       event_direction_valid=valid)
            cache=dict(injection={key:asdict(getattr(injection,key)) for key in ('observed','base')},
                       predicted_b=injection.predicted_b,predicted_x=injection.predicted_x,force=force,reflected=reflected,u=u,
                       dt_x=injection.dt_x,directions={})
            for name,direction in (('F',force),('G',reflected)):
                if name=='G' and not valid:
                    row['directions'][name]=dict(status='unavailable_event_direction',qualified=False)
                    continue
                stats,tensors=measure_direction(model,query,injection.base,direction,meter,context=f'k{k}:{name}',scales=cfg['fd_scales_l2_angstrom'])
                row['directions'][name]=stats;cache['directions'][name]=tensors
                if tensors:
                    row['old_to_next_endpoint_drift']=displacement_metrics(tensors['y0']-injection.predicted_x,query.atom_mask)
            row['status']='measured'
            cache_path=args.out/f'k{k}-cache.pt';torch.save(cache,cache_path)
            row['cache_sha256']=file_hash(cache_path)
            write_json(args.out/'report.json',report)
        report['status']='completed'
    except Exception as exc:
        report.update(status='failed_or_blocked',failure_type=type(exc).__name__,failure=str(exc))
        for row in report['states']:
            if row['status'] in ('started','not_run'):
                row['reason']='stage_stopped:'+type(exc).__name__
    work=meter.summary()
    report['work']={k:v for k,v in work.items() if k!='rows'}
    report['work']['attempts']=[{k:v for k,v in r.items() if k not in ('worker',)} for r in work['rows']]
    report['work_sha256']=file_hash(args.out/'work.json')
    report['raw_artifacts']=[dict(name=p.name,sha256=file_hash(p),bytes=p.stat().st_size) for p in sorted(args.out.iterdir()) if p.is_file() and p.name!='report.json']
    write_json(args.out/'report.json',report)
    print(json.dumps(dict(status=report['status'],counts=report['work']['counts'],states=[(r['step'],r['status']) for r in report['states']])))
    return int(report['status']!='completed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('research-source','model-root','packet','states','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(run(parser.parse_args()))
