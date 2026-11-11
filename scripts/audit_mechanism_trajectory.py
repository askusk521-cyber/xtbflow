#!/usr/bin/env python3
"""Read-only #188 trajectory contrasts; zero model, physical or reference calls."""
import argparse
import json
from pathlib import Path
import platform
import resource
import socket
import subprocess
import time

import torch

from xtbflow.validation.certification import file_hash,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generation',type=Path,required=True);p.add_argument('--expected-report-sha',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU,(60,60));resource.setrlimit(resource.RLIMIT_FSIZE,(4*1024**2,)*2)
    torch.set_num_threads(1);start,cpu=time.monotonic(),time.process_time()
    if a.out.exists():raise FileExistsError(a.out)
    if file_hash(a.generation/'report.json')!=a.expected_report_sha:raise ValueError('generation_report_mismatch')
    report=json.loads((a.generation/'report.json').read_text())
    raw=a.generation/'four-arm-tensors.pt'
    if file_hash(raw)!=report['raw_trajectory_sha256']:raise ValueError('trajectory_hash_mismatch')
    data=torch.load(raw,map_location='cpu',weights_only=True)
    contrasts=[];by_arm={r['arm']:r for r in report['candidates']}
    for first,second in [('F','C'),('F','T'),('T','N'),('C','N')]:
        x=data['states'][first]['x']-data['states'][second]['x']
        b=data['states'][first]['b']-data['states'][second]['b']
        event=(by_arm[first]['canonical_event']!=by_arm[second]['canonical_event']
               if all(by_arm[k]['status']=='generated' for k in (first,second)) else None)
        contrasts.append(dict(contrast=f'{first}-{second}',final_geometry_l2_angstrom=float(x.norm()),
                        final_geometry_rms_atom_angstrom=float(x.square().sum(-1).mean().sqrt()),
                        final_geometry_max_atom_angstrom=float(x.norm(dim=-1).max()),
                        final_continuous_BE_l2=float(b.norm()),final_canonical_event_changed=event))
    result=dict(schema='xtbflow-mechanism-trajectory-audit/v1',host=socket.gethostname(),python=platform.python_version(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1]).decode().strip(),
        driver_sha256=file_hash(Path(__file__)),generation_source_commit=report['source_commit'],
        generation_report_sha256=a.expected_report_sha,trajectory_sha256=file_hash(raw),contrasts=contrasts,
        complete_C_BE_trace_equals_N=bool(torch.equal(data['traces']['C']['b'],data['traces']['N']['b'])),
        T_final_BE_equals_N=bool(torch.equal(data['states']['T']['b'],data['states']['N']['b'])),
        reference_primitives=0,trained_forwards=0,energy_force=0,wall_seconds=time.monotonic()-start,
        process_cpu_seconds=time.process_time()-cpu,
        claim_limits=['Intermediate/continuous response and final event identity are distinct; no independent endpoint conclusion in this audit.',
                      'T inherits N generation: 0 incremental but 50 shared/standalone-attributable forwards.',
                      'C replays F observation/displacement data; zero new E/F does not make it a deployable cheap comparator.'])
    write_json(a.out,result)
    print(json.dumps(result['contrasts']))


if __name__=='__main__':main()
