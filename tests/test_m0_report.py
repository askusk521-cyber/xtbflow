"""Synthetic smoke tests only; fixtures are never scientific evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.m0_gate_report import aggregate


def test_parent_macro_not_query_micro():
    r={0:dict(parent_id='p',value=1),1:dict(parent_id='p',value=0),2:dict(parent_id='q',value=1)}
    assert aggregate([r],lambda x:x['value'])=={'p':.5,'q':1.}


def test_report_cli_frozen_fixture(tmp_path):
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    frozen=tmp_path/'frozen.json';frozen.write_text(json.dumps({'params':{'A':100,'B':100}}))
    subprocess.run(['git','-C',str(tmp_path),'add','frozen.json'],check=True)
    subprocess.run(['git','-C',str(tmp_path),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','freeze fixture'],check=True)
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps({'eval':{'bootstrap':100,'bootstrap_seed':0}}))
    aa=[];bb=[];candidates=[]
    ts=int(subprocess.check_output(['git','-C',str(tmp_path),'log','-1','--format=%ct'],text=True))
    for arm in ('a','b'):
        for seed in range(3):
            p=tmp_path/f'{arm}{seed}.jsonl'
            rows=[]
            for q in range(2):
                rows.append(dict(index=q,parent_id=str(q),hit={str(round(k/10,1)):arm=='b' for k in range(1,11)},valid_frac=1.,hit_event=arm=='b',consistency=.8,unique_valid_events=1,best_rmsd=.2 if arm=='b' else None,automorphism_cap_hit=False))
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows));(aa if arm=='a' else bb).append(str(p))
            c=tmp_path/f'{arm}{seed}.npz';c.write_bytes(b'fixture timestamp only');os.utime(c,(ts+2,ts+2));candidates.append(str(c))
    out=tmp_path/'result.json';report=tmp_path/'report.md'
    script=Path(__file__).resolve().parents[1]/'scripts/m0_gate_report.py'
    cmd=[sys.executable,str(script),'--a',*aa,'--b',*bb,'--config',str(cfg),'--frozen-selection','frozen.json','--candidates',*candidates,'--out',str(out),'--report',str(report)]
    env=dict(os.environ, PYTHONPATH=str(script.parents[1]/'src')+os.pathsep+str(script.parents[1]/'vendor/mechai_reusable'))
    executed=subprocess.run(cmd,cwd=tmp_path,env=env,capture_output=True,text=True)
    assert executed.returncode==0, executed.stderr
    result=json.loads(out.read_text())
    assert result['decision']=='GO' and result['primary']['delta']==1.
    assert report.with_suffix('.delta.svg').exists()
    failed=subprocess.run(cmd,cwd=tmp_path,capture_output=True)
    assert failed.returncode!=0
