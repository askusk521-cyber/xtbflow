#!/usr/bin/env python3
"""#188 bounded four-arm generation, candidate freeze and fake/cached new handoff."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
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

import numpy as np
import torch

from diagnose_physical_direction import load_frozen
from xtbflow.sampling.integrated_guidance import FlowState, candidate_from_state
from xtbflow.sampling.mechanism_four_arms import generate_four_arms
from xtbflow.sampling.physical_diagnostic import StageMeter, evaluate_physical
from xtbflow.sampling.guidance_fixture import FixtureReference
from xtbflow.runtime import RunLedger, StageBudget
from xtbflow.validation.certification import CertificationService, file_hash, identity, write_json
from xtbflow.validation.reference_protocol import canonical_event


def run(args):
    root=Path(__file__).resolve().parents[1]
    cfg_path=root/'configs/research/mechanism_188_20261111.json';cfg=json.loads(cfg_path.read_text())
    args.out.mkdir(exist_ok=False,parents=True)
    meter=StageMeter(args.out,cfg['guidance_caps'])
    resource.setrlimit(resource.RLIMIT_CPU,(1800,1800));resource.setrlimit(resource.RLIMIT_FSIZE,(64*1024**2,)*2)
    torch.set_num_threads(1)
    report=dict(schema='xtbflow-mechanism-generation/v1',issue=188,status='preflight',host=socket.gethostname(),
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root).decode().strip(),
                config_sha256=file_hash(cfg_path),python=platform.python_version(),job_id=os.environ.get('SLURM_JOB_ID'),
                packages={p:importlib.metadata.version(p) for p in ['torch','numpy','scipy','tblite','rdkit']},
                candidates=[dict(arm=a,status='not_run') for a in cfg['arms']],claim_limits=cfg['claim_limits'])
    try:
        if subprocess.check_output(['git','status','--porcelain'],cwd=root).strip():raise ValueError('dirty_source')
        if not cfg['reference']['identity_sha256']:raise ValueError('reference_protocol_not_frozen')
        for name,sha in cfg['decoder_integrated_files'].items():
            if file_hash(root/name)!=sha:raise ValueError('integrated_decoder_identity_mismatch')
        amplitude=root/'docs/evidence/amplitude_199_20261111/results_61e8d70.json'
        if file_hash(amplitude)!=cfg['amplitude_report_sha256']:raise ValueError('amplitude_report_mismatch')
        prior=json.loads(amplitude.read_text())
        if prior['s5_gate']!='eligible' or prior['work']['unresolved_started']:raise ValueError('amplitude_gate_closed')
        physical_cfg=json.loads((root/'configs/research/physical_direction_198_20261111.json').read_text())
        from xtbflow.calculators.xtb_oracle import _package_identity,_adapter_code_hash
        version,_,build,_=_package_identity('tblite');spec=physical_cfg['calculator']
        if (version,build,_adapter_code_hash())!=(spec['version'],spec['build_sha256'],spec['adapter_sha256']):
            raise ValueError('calculator_identity_mismatch')
        model,query,_,identities=load_frozen(args,physical_cfg,root)
        report['identities']=identities
        from xtbflow.m0.sampler import decode_be
        from xtbflow.v1.sampler import initial_state
        s0=json.loads((root/'configs/research/narrowed_scope_20261111.json').read_text())
        if file_hash(args.reactants)!=s0['diagnostic']['reactant_packet_sha256']:raise ValueError('reactant_context_hash_mismatch')
        packet=json.loads(args.reactants.read_text())['queries'][0]
        from xtbflow.governance.generation_bridge import legacy_query
        if legacy_query(packet)!=json.loads(args.packet.read_text())['queries'][0]:raise ValueError('reactant_worker_mapping_mismatch')
        float_query=replace(query,x_r=query.x_r.float(),b_r=query.b_r.float())
        initial=initial_state(float_query,[0],0,sampling_seed=0,namespace=s0['noise']['namespace'],group='common',sigma_b=1.,sigma_x=.5)
        initial=FlowState(*(getattr(initial,k).double() for k in ('b','x','t_b','t_x')))
        report['initial_noise_state_sha256']=identity({k:getattr(initial,k).tolist() for k in ('b','x','t_b','t_x')})
        model.eval().requires_grad_(False);meter.install_model(model)
        counts={a:0 for a in cfg['arms']}
        def physical(arm,y,label):
            if counts[arm]>=cfg['arm_ef_caps'][arm]:raise RuntimeError('arm_EF_cap_exhausted')
            counts[arm]+=1
            return evaluate_physical(y,query.atomic_numbers,spec,meter,label)
        result=generate_four_arms(model,query,initial,physical,meter)
        if meter.summary()['unresolved_started']:raise ValueError('unsettled_guidance')
        tensors={k:v for k,v in result.items() if k!='states'}
        tensors['states']={k:asdict(v) for k,v in result['states'].items()}
        raw=args.out/'four-arm-tensors.pt';torch.save(tensors,raw)
        report.update(raw_trajectory_sha256=file_hash(raw),slots=result['slots'],arm_work=result['work'])
        report['F_nonzero_updates']=sum(float(d.norm())>0 for d in result['pushes'].values())
        protocol=dict(config_sha256=file_hash(cfg_path),source_commit=report['source_commit'],
                      initial_state_sha256=report['initial_noise_state_sha256'],checkpoint_sha256=identities['checkpoint_sha256'])
        plan=dict(schema='xtbflow-mechanism-candidate-freeze/v1',source_commit=report['source_commit'],
                  config_sha256=file_hash(cfg_path),protocol=protocol,candidates=[])
        upstream=dict(status='completed',termination='confirmed',settlement='settled',reservations={})
        for row in report['candidates']:
            arm=row['arm'];state=result['states'][arm]
            decoded,_=decode_be(state.b[0],query.element_index[0],16)
            row['geometry_sha256']=identity(state.x[0].tolist());row['continuous_BE_sha256']=identity(state.b[0].tolist())
            if decoded is None:
                row.update(status='invalid_decode',reason='final_BE_decoder_rejected');continue
            candidate_id='mechanism:'+identity(dict(x=state.x[0].tolist(),b=decoded.tolist(),protocol=protocol))
            candidate=candidate_from_state(packet,state,lambda b:decoded,candidate_id=candidate_id,protocol=protocol,upstream=upstream)
            event=canonical_event(np.asarray(packet['b_r']),decoded,np.asarray(packet['reactant_automorphisms']))
            row.update(status='generated',candidate_id=candidate_id,candidate_sha256=identity(candidate.content()),canonical_event=event)
            plan['candidates'].append(dict(arm=arm,candidate=asdict(candidate),candidate_sha256=identity(candidate.content())))
        write_json(args.out/'plan.json',plan)
        report['candidate_plan_sha256']=file_hash(args.out/'plan.json')
        # Freeze every candidate before ANY reference result, including the software handoff check.
        write_json(args.out/'generation-freeze.json',report)
        software=CertificationService(args.out/'fake-handoff',StageBudget('mechanism-fake-handoff',max_calculator_calls=4,
                    max_gpu_hours=0,max_cpu_core_hours=.1,max_storage_gb=.02,max_wall_minutes_per_job=1,max_retries_per_job=0))
        handoff=[]
        from xtbflow.validation.certification import CertificationCandidate
        for item in plan['candidates']:
            candidate=CertificationCandidate(**item['candidate'])
            first=software.run(candidate,FixtureReference(),max_calls=1,max_wall_seconds=10,cpu_cores=1,max_raw_bytes=1024**2)
            replay=software.run(candidate,FixtureReference(),max_calls=1,max_wall_seconds=10,cpu_cores=1,max_raw_bytes=1024**2)
            if not replay['cache_hit'] or replay['new_calculator_calls'] or first['record']!=replay['record']:
                raise ValueError('new_intervention_handoff_replay_failure')
            if first['record']['execution_status']!='completed':raise ValueError('fake_handoff_did_not_execute')
            handoff.append(dict(arm=item['arm'],candidate_sha256=item['candidate_sha256'],cache_hit=replay['cache_hit'],new_replay_calls=0))
        report['new_candidate_handoff']=handoff
        report['fake_reference_ledger']=RunLedger.load(software.ledger_path).summary() if plan['candidates'] else None
        report['status']='generation_completed'
        report['settlement']='settled';report['termination']='confirmed'
        report['reference_gate']='eligible' if report['F_nonzero_updates'] and plan['candidates'] else 'do_not_launch'
    except Exception as exc:
        report.update(status='failed_or_blocked',failure_type=type(exc).__name__,
                      failure=str(exc) if not isinstance(exc,OSError) else 'asset_or_report_io_failure',reference_gate='do_not_launch')
    work=meter.summary();report['work']={k:v for k,v in work.items() if k!='rows'}
    report['work']['attempts']=[{k:v for k,v in r.items() if k!='worker'} for r in work['rows']]
    report['work_sha256']=file_hash(args.out/'work.json')
    report['raw_artifacts']=[dict(name=p.name,sha256=file_hash(p),bytes=p.stat().st_size) for p in sorted(args.out.iterdir()) if p.is_file() and p.name!='report.json']
    write_json(args.out/'report.json',report)
    print(json.dumps(dict(status=report['status'],counts=report['work']['counts'],reference_gate=report['reference_gate'],
                         candidates=[(r['arm'],r['status']) for r in report['candidates']])))
    return int(report['status']!='generation_completed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('research-source','model-root','packet','states','reactants','out'):parser.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(run(parser.parse_args()))
