#!/usr/bin/env python3
"""Sequential fixed-cap independent certification of all frozen valid #188 candidates."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

from xtbflow.governance.mechanism_handoff import load_mechanism_candidates
from xtbflow.runtime import RunLedger, StageBudget
from xtbflow.validation.certification import CertificationService, file_hash, identity, write_json
from xtbflow.validation.reference_backend import ReferenceBackend


def run(args):
    root=Path(__file__).resolve().parents[1]
    cfg_path=root/'configs/research/mechanism_188_20261111.json';cfg=json.loads(cfg_path.read_text())
    if socket.gethostname()!='node2' or not os.environ.get('SLURM_JOB_ID'):raise ValueError('finite_n2_Slurm_allocation_required')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root).strip():raise ValueError('dirty_reference_source')
    args.out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    report=dict(schema='xtbflow-mechanism-reference/v1',host=socket.gethostname(),python=platform.python_version(),
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root).decode().strip(),
                config_sha256=file_hash(cfg_path),generation_report_sha256=args.expected_generation_sha,
                job_id=os.environ['SLURM_JOB_ID'],status='preflight',arms=[dict(arm=a,status='not_started') for a in cfg['arms']],
                caps=cfg['reference'],claim_limits=cfg['claim_limits'],scheduler_cpu_seconds=None,scheduler_max_rss=None)
    write_json(args.out/'report.json',report)
    service=None
    try:
        generation,candidates=load_mechanism_candidates(args.generation,cfg_path,args.expected_generation_sha)
        if generation['source_commit']!=report['source_commit']:raise ValueError('cross_source_reference_handoff')
        probe=subprocess.check_output([sys.executable,'-c',
            "import cupy as c,json; r=c.cuda.runtime;p=r.getDeviceProperties(0);a=c.zeros(1);c.cuda.get_current_stream().synchronize();print(json.dumps(dict(gpu=p['name'].decode(),driver=r.driverGetVersion(),runtime=r.runtimeGetVersion())))"],text=True,timeout=30)
        backend=ReferenceBackend(device='gpu',execution_build=json.loads(probe))
        if identity(backend.identity)!=cfg['reference']['identity_sha256']:raise ValueError('reference_build_protocol_mismatch')
        report['protocol_sha256']=identity(backend.identity)
        guidance_bytes=sum(p.stat().st_size for p in args.generation.rglob('*') if p.is_file())
        budget=dict(cfg['reference']['stage_budget'])
        budget['max_storage_gb']=min(budget['max_storage_gb'],
            (cfg['global_caps']['storage_bytes']-guidance_bytes-1024**2)/1024**3)
        report['effective_reference_budget']=budget
        service=CertificationService(args.out/'service',StageBudget(**budget))
        by_arm=dict(candidates)
        for row in report['arms']:
            arm=row['arm']
            if arm not in by_arm:
                original=next(r for r in generation['candidates'] if r['arm']==arm)
                row.update(status='invalid_generated_candidate',reason=original.get('reason'));continue
            if time.monotonic()-started+cfg['reference']['per_candidate']['max_wall_seconds']+5>cfg['reference']['job_wall_seconds']:
                raise RuntimeError('shared_reference_wall_budget_exhausted')
            size=sum(p.stat().st_size for p in args.out.rglob('*') if p.is_file())+sum(p.stat().st_size for p in args.generation.rglob('*') if p.is_file())
            if size>=cfg['global_caps']['storage_bytes']:raise RuntimeError('shared_storage_budget_exhausted')
            candidate=by_arm[arm];row.update(status='started',candidate_sha256=identity(candidate.content()))
            write_json(args.out/'report.json',report)
            result=service.run(candidate,backend,**cfg['reference']['per_candidate'])
            record=result['record'];ledger=RunLedger.load(service.ledger_path)
            settlement=service.root/record['cache_key']/f"attempt-{record['attempt']:04d}"/'settlement.json'
            if ledger.reservations or not settlement.exists() or json.loads(settlement.read_text())['status']!='settled':
                raise RuntimeError('reference_settlement_unknown')
            row.update(status=record['execution_status'],feedback={k:record['feedback'].get(k) for k in
                        ('evidence_status','stationary_point','first_order_saddle','bidirectional_minima','planned_event_realized',
                         'planned_event_id','observed_event_id','raw_status','terminal_stage','stage_reason')},
                       cache_key=record['cache_key'],cache_hit=result['cache_hit'],new_calculator_calls=result['new_calculator_calls'],
                       charged_calls=record['charged_calls'],resources=record['resources'],result_sha256=record['content_sha256'],
                       settlement='settled',termination='confirmed',retained_artifact_bytes=sum(a['bytes'] for a in record['artifacts']))
            if result['cache_hit']:
                row['shared_from_arm']=next((p['arm'] for p in report['arms'] if p is not row and p.get('cache_key')==record['cache_key']),None)
            report['ledger']=ledger.summary()
            write_json(args.out/'report.json',report)
        report['status']='completed'
    except Exception as exc:
        report.update(status='stopped',failure_type=type(exc).__name__,
                      reason=str(exc) if not isinstance(exc,OSError) else 'asset_or_report_io_failure')
        for row in report['arms']:
            if row['status']=='started':row.update(status='failed_or_unsettled',reason=type(exc).__name__)
            elif row['status']=='not_started':row['reason']='shared_stop:'+type(exc).__name__
    if service and service.ledger_path.exists():report['ledger']=RunLedger.load(service.ledger_path).summary()
    channels=args.out/'service/channels.json'
    report['unique_certified_channels']=len(json.loads(channels.read_text())) if channels.exists() else 0
    report['wall_seconds']=time.monotonic()-started
    report['GPU_time_basis']='GPU allocation/device-residence wall, not measured active kernel time'
    write_json(args.out/'report.json',report)
    print(json.dumps(dict(status=report['status'],channels=report['unique_certified_channels'],arms=[(r['arm'],r['status']) for r in report['arms']])))
    return int(report['status']!='completed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation',type=Path,required=True)
    parser.add_argument('--expected-generation-sha',required=True)
    parser.add_argument('--out',type=Path,required=True)
    raise SystemExit(run(parser.parse_args()))
