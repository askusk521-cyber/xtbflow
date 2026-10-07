"""Candidate CSR evaluator integration, using synthetic chemistry only."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from xtbflow.m0.t1x_data import write_cache


def test_evaluate_csr_cli(tmp_path):
    pytest.importorskip('rdkit')
    r=np.array([[4,1,1],[1,0,0],[1,0,0]],dtype=np.int8)
    p=np.array([[6,0,1],[0,0,0],[1,0,0]],dtype=np.int8)
    x=np.array([[0.,0.,0.],[1.5,0.,0.],[-.3,.9,0.]],dtype=np.float32)
    rec=dict(z=np.array([8,1,1],dtype=np.int8),x_r=x,x_ts=x,x_p=x,b_r=r,b_p=p,
             energies=np.array([0.,1.,0.]),split=1,parent_id='fixture',rxn_key='fixture',formula='H2O',smiles_r='O',smiles_p='[OH-].[H+]')
    cache=tmp_path/'cache.npz';write_cache([rec],cache)
    cand=tmp_path/'candidates.npz'
    np.savez_compressed(cand,query_index=np.array([0]),sample_id=np.array([0]),n_atoms=np.array([3]),atom_off=np.array([0,3]),x=x,b_raw=p.astype(np.float32).reshape(-1))
    script=Path(__file__).resolve().parents[1]/'scripts/m0_evaluate.py'
    env=dict(os.environ,PYTHONPATH=str(script.parents[1]/'src')+os.pathsep+str(script.parents[1]/'vendor/mechai_reusable'))
    out=tmp_path/'eval.jsonl'
    cmd=[sys.executable,str(script),'--cache',str(cache),'--candidates',str(cand),'--split','val','--out',str(out)]
    got=subprocess.run(cmd,env=env,cwd=tmp_path,capture_output=True,text=True)
    assert got.returncode==0,got.stderr
    row=json.loads(out.read_text())
    assert row['hit_event'] and row['hit']['0.5'] and row['valid_frac']==1.
    assert row['best_rmsd']<1e-6 and row['parent_id']=='fixture'
