"""Synthetic EMA loading and saved candidate shape integration tests."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from xtbflow.m0.model import ReactionFlowNet
from xtbflow.m0.t1x_data import write_cache


@pytest.mark.parametrize('arm',['joint','cascade'])
def test_sample_cli_ema_and_csr(tmp_path,arm):
    cfg_model=dict(scalar_dim=32,vector_dim=8,edge_dim=16,n_layers=2)
    r=np.array([[4,1,1],[1,0,0],[1,0,0]],dtype=np.int8)
    x=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],dtype=np.float32)
    rec=dict(z=np.array([8,1,1],dtype=np.int8),x_r=x,x_ts=x,x_p=x,b_r=r,b_p=r,
             energies=np.array([0.,1.,0.]),split=1,parent_id='fixture',rxn_key='fixture',formula='H2O',smiles_r='O',smiles_p='O')
    cache=tmp_path/'cache.npz';write_cache([rec],cache)
    roles=['joint'] if arm=='joint' else ['event','geometry']
    flags=[]
    for role in roles:
        net=ReactionFlowNet(role,**cfg_model)
        ckpt=tmp_path/f'{role}.pt'
        torch.save(dict(ema=net.state_dict(),role=role,model_cfg=cfg_model,seed=0,step=20,sigma_b=.5,sigma_x=.5),ckpt)
        flags.extend(['--checkpoint' if role=='joint' else f'--{role}-checkpoint',str(ckpt)])
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(data=dict(cache=str(cache)),sample=dict(nfe=2,n_samples_val=2,n_samples_test=32,queries_per_batch=1))))
    script=Path(__file__).resolve().parents[1]/'scripts/m0_sample.py'
    env=dict(os.environ,PYTHONPATH=str(script.parents[1]/'src')+os.pathsep+str(script.parents[1]/'vendor/mechai_reusable'),OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    out=tmp_path/'candidates.npz'
    cmd=[sys.executable,str(script),'--config',str(config),'--arm',arm,*flags,'--split','val','--seed','0','--out',str(out)]
    got=subprocess.run(cmd,env=env,cwd=tmp_path,capture_output=True,text=True)
    assert got.returncode==0,got.stderr
    with np.load(out,allow_pickle=False) as c:
        assert c['query_index'].tolist()==[0,0]
        assert c['sample_id'].tolist()==[0,1]
        assert c['atom_off'].tolist()==[0,3,6]
        assert c['x'].shape==(6,3) and c['b_raw'].shape==(18,)
        assert np.isfinite(c['x']).all() and np.isfinite(c['b_raw']).all()
        assert not np.array_equal(c['x'][:3],c['x'][3:])
    assert subprocess.run(cmd,env=env,cwd=tmp_path,capture_output=True).returncode!=0
