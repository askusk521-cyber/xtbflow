#!/usr/bin/env python3
"""#199 frozen single-factor step backtracking; reuse exact #198 derivative caches."""
from __future__ import annotations

import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''

import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import socket
import subprocess

import torch

from diagnose_physical_direction import load_frozen
from xtbflow.sampling.amplitude_backtracking import bounded_backtrack
from xtbflow.sampling.integrated_guidance import FlowState, Injection
from xtbflow.sampling.physical_diagnostic import StageMeter, evaluate_physical
from xtbflow.validation.certification import file_hash, write_json


def run(args):
    root=Path(__file__).resolve().parents[1]
    path=root/'configs/research/amplitude_backtracking_199_20261111.json'
    cfg=json.loads(path.read_text())
    parent_cfg=json.loads((root/'configs/research/physical_direction_198_20261111.json').read_text())
    args.out.mkdir(parents=True,exist_ok=False)
    meter=StageMeter(args.out,cfg['caps'])
    resource.setrlimit(resource.RLIMIT_CPU,(cfg['caps']['cpu_seconds'],)*2)
    resource.setrlimit(resource.RLIMIT_FSIZE,(cfg['caps']['output_bytes'],)*2)
    torch.set_num_threads(1)
    report=dict(schema='xtbflow-amplitude-backtracking-result/v1',issue=199,status='running',host=socket.gethostname(),
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root).decode().strip(),
                config_sha256=file_hash(path),job_id=os.environ.get('SLURM_JOB_ID'),python=platform.python_version(),
                packages={p:importlib.metadata.version(p) for p in ['torch','numpy','scipy','tblite','rdkit']},
                diagnostic_report_sha256=cfg['diagnostic_report_sha256'],caps=cfg['caps'],
                states=[dict(step=k,status='not_run') for k in (10,25,40)],claim_limits=cfg['claim_limits'])
    try:
        if subprocess.check_output(['git','status','--porcelain'],cwd=root).strip():
            raise ValueError('dirty_integrated_source')
        if file_hash(args.diagnostic/'report.json')!=cfg['diagnostic_report_sha256']:
            raise ValueError('diagnostic_report_hash_mismatch')
        prior=json.loads((args.diagnostic/'report.json').read_text())
        if prior['status']!='completed' or prior['work']['unresolved_started']:
            raise ValueError('incomplete_diagnostic')
        from xtbflow.calculators.xtb_oracle import _package_identity, _adapter_code_hash
        version,_,build,_=_package_identity('tblite')
        spec=parent_cfg['calculator']
        if (version,build,_adapter_code_hash())!=(spec['version'],spec['build_sha256'],spec['adapter_sha256']):
            raise ValueError('calculator_preflight_identity_mismatch')
        model,query,_,identities=load_frozen(args,parent_cfg,root)
        report['identities']=identities
        meter.install_model(model)
        for row,previous in zip(report['states'],prior['states']):
            k=row['step'];row['status']='started'
            if previous['step']!=k:
                raise ValueError('state_order_mismatch')
            if not previous.get('cache_sha256'):
                row.update(status='no_push',reason='no_qualified_physical_cache')
                continue
            cache_path=args.diagnostic/f'k{k}-cache.pt'
            if file_hash(cache_path)!=previous['cache_sha256']:
                raise ValueError('diagnostic_cache_hash_mismatch')
            cache=torch.load(cache_path,map_location='cpu',weights_only=True)
            injection=Injection(FlowState(**cache['injection']['observed']),FlowState(**cache['injection']['base']),
                                cache['predicted_b'],cache['predicted_x'],cache['dt_x'])
            def physical(y,label):
                return evaluate_physical(y,query.atomic_numbers,spec,meter,label)
            before=meter.summary()['counts']
            state,result=bounded_backtrack(model,query,injection,cache['reflected'],cache['directions'].get('G',{}),
                                           previous['directions']['G'],physical,context=f'k{k}')
            row.update(result,source_cache_sha256=previous['cache_sha256'])
            after=meter.summary()['counts']
            row['new_work']={name:after[name]-before[name] for name in before}
            row['fixed_amplitude_baseline']=row['trials'][0] if row.get('trials') else None
            selected=args.out/f'k{k}-selected.pt';torch.save(asdict(state),selected)
            row['selected_state_sha256']=file_hash(selected)
            write_json(args.out/'report.json',report)
        report['status']='completed'
    except Exception as exc:
        report.update(status='failed_or_blocked',failure_type=type(exc).__name__,
                      failure=str(exc) if not isinstance(exc,OSError) else 'asset_or_report_io_failure')
        for row in report['states']:
            if row['status'] in ('started','not_run'):
                row['reason']='stage_stopped:'+type(exc).__name__
    work=meter.summary()
    report['work']={k:v for k,v in work.items() if k!='rows'}
    report['work']['attempts']=[{k:v for k,v in r.items() if k!='worker'} for r in work['rows']]
    report['work_sha256']=file_hash(args.out/'work.json')
    report['qualified_nonzero_updates']=sum(row['status']=='accepted' for row in report['states'])
    report['s5_gate']='eligible' if report['status']=='completed' and report['qualified_nonzero_updates'] else 'do_not_launch'
    report['raw_artifacts']=[dict(name=p.name,sha256=file_hash(p),bytes=p.stat().st_size) for p in sorted(args.out.iterdir()) if p.is_file() and p.name!='report.json']
    write_json(args.out/'report.json',report)
    print(json.dumps(dict(status=report['status'],counts=report['work']['counts'],s5_gate=report['s5_gate'],
                         selected=[(r['step'],r.get('selected_lambda')) for r in report['states']])))
    return int(report['status']!='completed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('research-source','model-root','packet','states','diagnostic','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(run(parser.parse_args()))
