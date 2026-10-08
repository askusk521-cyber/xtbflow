"""Archive provenance, raw verdict hashes, and the exact frozen sample ledger."""
import argparse
import json
from pathlib import Path
import subprocess
from xtbflow.v1.data import file_hash, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    cfg=a.source/'configs/review/open_world_o1.json';config=json.loads(cfg.read_text())
    files=[];verdicts=[]
    for group,rows in sorted(config['groups'].items()):
        for row in rows:
            ident=row.get('candidate_id',row.get('reference_id'));path=a.run/group/ident/'verdict.json'
            if not path.exists():raise RuntimeError('incomplete frozen chain set: '+str(path))
            files.append(path);verdicts.append(json.loads(path.read_text()))
    for name in ('pip-freeze.txt','positive_control_gate.json','run.log'):
        files.append(a.run/name)
    a.out.mkdir(parents=True,exist_ok=True)
    manifest=dict(source_commit=subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip(),
        dirty=bool(subprocess.check_output(['git','-C',str(a.source),'status','--porcelain']).strip()),
        config_sha256=file_hash(cfg),analysis_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        protocol='review-xtb-o1-1',cpu_direct_authorized=True,gpu_hours=0,slurm_jobs=[],chains=len(verdicts),
        completed_chain_wall_seconds=sum(r['wall_seconds'] for r in verdicts),
        gradient_calls=sum(r['gradient_calls'] for r in verdicts),
        numerical_settings=dict(method='GFN2-xTB',charge=0,uhf=0,accuracy=.1,max_iterations=250,
                                electronic_temperature='tblite default',hessian_step_bohr=.005),
        files=[dict(path=str(f),sha256=file_hash(f)) for f in files])
    write_json(a.out/'manifest.json',manifest)
    (a.out/'SHA256SUMS.remote').write_text(''.join(file_hash(f)+'  '+str(f)+'\n' for f in files))


if __name__=='__main__':main()
