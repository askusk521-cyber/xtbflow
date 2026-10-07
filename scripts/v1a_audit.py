"""Read actual V1a assets and environment; unmeasured costs remain null."""
import argparse
from collections import Counter
import importlib
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

from xtbflow.m0.model import ReactionFlowNet,count_parameters
from xtbflow.m0.t1x_data import T1xCache
from xtbflow.v1.data import digest,file_hash,write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();cfg=json.loads(a.config.read_text())
    cache=T1xCache(Path(cfg['data']['cache']))
    data=Path(cfg['data']['catalogue_dir'])
    parents=json.loads((data/'parent_catalog.json').read_text())
    split=json.loads((data/'split_manifest.json').read_text())
    versions={}
    for name in ['numpy','scipy','torch','rdkit','h5py','pytest']:
        mod=importlib.import_module(name)
        versions[name]=dict(version=mod.__version__,path=mod.__file__)
    sets={k:set(split[k]['formula_groups']) for k in ['train','development','screen_reserve']}
    overlaps={a+'__'+b:sorted(sets[a]&sets[b]) for a in sets for b in sets if a<b}
    if any(overlaps.values()):
        raise ValueError('formula leakage')
    params={role:count_parameters(ReactionFlowNet('joint' if role=='baseline' else role,
             **(cfg['model'] if role in ['joint','baseline'] else cfg['cascade_model']),
             dual_time=role=='joint')) for role in ['joint','baseline','event','geometry']}
    ratio=abs(params['event']+params['geometry']-params['joint'])/params['joint']
    if ratio>.05:
        raise ValueError('generator capacity mismatch')
    original_manifest=json.loads(Path(cfg['data']['cache']).with_suffix('.manifest.json').read_text())
    if file_hash(cfg['data']['cache'])!=original_manifest['cache_sha256']:
        raise ValueError('source cache hash mismatch')
    report=dict(schema='xtbflow-v1a-audit/2.1',stage0_complete=False,
        remaining_stage0=['development forward/gradient/matcher benchmark',
                          'frozen network call weights','resource forecast'],
        hostname=platform.node(),python=sys.version,executable=sys.executable,versions=versions,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
        config_sha256=file_hash(a.config),cache_sha256=original_manifest['cache_sha256'],
        source_h5_sha256=original_manifest['h5_sha256'],
        energy_provenance='M0 T1x CSR cache; original loader copies wB97x_6-31G(d).energy as eV; '
            'existing raw endpoint audit retained in docs/evidence/m0/diagnostics/replay-2547',
        raw_h5_revalidated_this_run=False,
        raw_h5_limitation='Original HDF5 not located in audited home/data paths; retain existing source hash and audit, never replace native TS.',
        split_hash=split['split_hash'],formula_overlap=overlaps,
        asset_hashes={p.name:file_hash(p) for p in data.iterdir() if p.is_file()},
        split_summary={k:dict(parents=len(split[k]['parent_ids']),groups=len(split[k]['formula_groups']))
                       for k in sets},
        training_reactions=len(split['train']['cache_indices']),
        parent_total=len(parents),cache_records=len(cache),
        atomic_numbers=sorted(int(v) for v in np.unique(cache.arrays['z'])),
        energy_unit='eV',coordinate_unit='angstrom',native_frame_order=['R','TS','P'],
        all_energies_finite=bool(np.isfinite(cache.arrays['energies']).all()),
        params=params,cascade_vs_joint_relative_difference=ratio,
        checkpoint_reuse='none; existing M0 generators saw official train pool',
        official_val_test='excluded from V1a training/development/screen; already used by M0',
        quantum_evaluations_allowed=False,
        cuda_build=torch.version.cuda,cuda_visible=torch.cuda.is_available(),
        cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        free_disk_gb=shutil.disk_usage(data).free/1024**3,
        training_gpu_hours_remaining=None,proxy_run_gpu_hours=None,mechanism_gpu_hours=None,
        evaluation_cpu_hours=None,network_call_cost_table=None,wall_days_forecast=None)
    write_json(a.out,report)
    print(json.dumps({k:report[k] for k in ['source_commit','source_dirty','split_summary',
                                         'training_reactions','params','cuda_visible']}))


if __name__=='__main__':
    main()
