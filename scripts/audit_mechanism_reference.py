#!/usr/bin/env python3
"""Read-only audit of NEW #188 records; never calls a certification backend/service run."""
import argparse
import json
from pathlib import Path
import platform
import resource
import socket
import subprocess
import time

from xtbflow.runtime import RunLedger
from xtbflow.validation.certification import _artifact_index,file_hash,identity,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--expected-report-sha',required=True)
    p.add_argument('--generation',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();start,cpu=time.monotonic(),time.process_time()
    resource.setrlimit(resource.RLIMIT_CPU,(60,60));resource.setrlimit(resource.RLIMIT_FSIZE,(4*1024**2,)*2)
    if a.out.exists():raise FileExistsError(a.out)
    report_path=a.reference/'report.json'
    if file_hash(report_path)!=a.expected_report_sha:raise ValueError('reference_report_hash_mismatch')
    report=json.loads(report_path.read_text());generation=json.loads((a.generation/'report.json').read_text())
    if file_hash(a.generation/'report.json')!=report['generation_report_sha256']:raise ValueError('generation_identity_mismatch')
    if report['status'] not in ('completed','stopped'):raise ValueError('reference_not_terminal')
    original={r['arm']:r for r in generation['candidates']}
    ledger=RunLedger.load(a.reference/'service/ledger.json')
    rows=[];seen=set();charged=0
    for arm in report['arms']:
        if 'cache_key' not in arm:
            rows.append(dict(arm=arm['arm'],status=arm['status'],record_verified=False,reason=arm.get('reason')))
            continue
        folder=a.reference/'service'/arm['cache_key']/'attempt-0001'
        result_path=folder/'result.json';before=file_hash(result_path)
        record=json.loads(result_path.read_text());request=json.loads((folder/'request.json').read_text())
        if identity({k:v for k,v in record.items() if k!='content_sha256'})!=record['content_sha256']:
            raise ValueError('canonical_result_hash_mismatch')
        if record['content_sha256']!=arm['result_sha256'] or record['artifacts']!=_artifact_index(folder):
            raise ValueError('published_result_or_artifact_mismatch')
        if file_hash(folder/'request.json')!=record['request_sha256']:raise ValueError('request_hash_mismatch')
        payload={k:v for k,v in request['candidate'].items() if k!='candidate_id'}
        if identity(payload)!=original[arm['arm']]['candidate_sha256']:raise ValueError('candidate_handoff_mismatch')
        if identity(record['protocol'])!=report['protocol_sha256']:raise ValueError('reference_protocol_mismatch')
        calls=json.loads((folder/'raw/calls.json').read_text())
        n=calls['consumed_calls'];attempts=calls['attempts']
        if n!=record['charged_calls'] or len(attempts)!=n or [r['sequence'] for r in attempts]!=list(range(1,n+1)):
            raise ValueError('started_call_denominator_mismatch')
        if n>800 or request['max_calls']!=800 or request['max_wall_seconds']!=3300:raise ValueError('candidate_cap_changed')
        if json.loads((folder/'settlement.json').read_text())['status']!='settled':raise ValueError('unsettled_candidate')
        if before!=file_hash(result_path):raise ValueError('result_changed_during_audit')
        if arm['cache_key'] not in seen:charged+=n;seen.add(arm['cache_key'])
        rows.append(dict(arm=arm['arm'],record_verified=True,result_content_sha256=record['content_sha256'],
                         result_file_sha256=before,charged_calls=n,cache_hit=arm['cache_hit'],
                         completed_primitives=sum(r['status']=='completed' for r in attempts),
                         started_unfinished_primitives=sum(r['status']=='started' for r in attempts),
                         by_kind={kind:sum(r['kind']==kind for r in attempts) for kind in ('scf','gradient','hessian')},
                         reason=record['feedback']['stage_reason'],unchanged_during_audit=True))
    if ledger.reservations:raise ValueError('unsettled_shared_ledger')
    if charged!=ledger.summary()['totals']['calculator_calls'] or charged>3200:raise ValueError('global_calls_mismatch')
    if file_hash(report_path)!=a.expected_report_sha:raise ValueError('report_changed_during_audit')
    out=dict(schema='xtbflow-mechanism-reference-audit/v1',host=socket.gethostname(),python=platform.python_version(),
             source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1]).decode().strip(),
             driver_sha256=file_hash(Path(__file__)),reference_report_sha256=a.expected_report_sha,
             execution_source_commit=report['source_commit'],rows=rows,unique_executed_candidates=len(seen),
             total_charged_reference_primitives=charged,ledger_reconciled=True,
             new_reference_primitives=0,new_trained_forwards=0,new_energy_force=0,
             wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,
             claim_limits=['Read-only identity/accounting verification; no recalculation or chemistry reclassification.',
                           'Legacy historical records were not opened or refreshed.'])
    write_json(a.out,out);print(json.dumps(dict(verified=True,charged=charged,candidates=len(seen))))


if __name__=='__main__':main()
