"""Frozen T1b geometry and energy acquisition in the read-only aimnet environment.

Only RDKit geometry construction occurs here, not event decoding/catalogue matching.
Neutral total charge; energies kcal/mol; coordinates Angstrom. One conformer,
one embedding fallback, no energy retries. JSONL outputs retain every failure.
"""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from xtbflow.m0.metrics import be_to_mol

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_oracle_benchmark import energy


def fragment_seed(parent_id, channel_id, index):
    return int(sha256(f'{parent_id}|{channel_id}|{index}'.encode()).hexdigest()[:8], 16) % 2**31


def product_geometry(z, b, parent_id, channel_id):
    """Fragment-local ETKDG/MMFF(UFF), then 10-Angstrom centroid separation."""
    mol = be_to_mol(np.asarray(z), np.asarray(b))
    fragments = Chem.GetMolFrags(mol, asMols=True)
    numbers, positions, states = [], [], []
    for index, fragment in enumerate(fragments):
        seed = fragment_seed(parent_id, channel_id, index)
        embedded = False
        for random_coords in (False, True):
            params = AllChem.ETKDGv3()
            params.randomSeed = seed
            params.useRandomCoords = random_coords
            if AllChem.EmbedMolecule(fragment, params) == 0:
                embedded = True
                break
        if not embedded:
            raise ValueError('embedding_failed')
        state = dict(fragment_index=index, seed=seed, random_coords=random_coords)
        if AllChem.MMFFHasAllMoleculeParams(fragment):
            result = AllChem.MMFFOptimizeMolecule(fragment, mmffVariant='MMFF94', maxIters=1000)
            state.update(forcefield='MMFF94', optimization_code=result)
        elif AllChem.UFFHasAllMoleculeParams(fragment):
            result = AllChem.UFFOptimizeMolecule(fragment, maxIters=1000)
            state.update(forcefield='UFF', optimization_code=result)
        else:
            result = 0
            state.update(forcefield='no_forcefield', optimization_code=None)
        if result != 0:
            raise ValueError('forcefield_not_converged' if result == 1 else 'forcefield_failed')
        x = np.asarray(fragment.GetConformer().GetPositions())
        x = x - x.mean(axis=0) + np.array([10.0*index, 0.0, 0.0])
        numbers.extend(atom.GetAtomicNum() for atom in fragment.GetAtoms())
        positions.extend(x.tolist())
        states.append(state)
    return np.asarray(numbers), np.asarray(positions), states


def single_energy(z, b, parent_id, channel_id, cfg):
    start = time.perf_counter()
    try:
        numbers, x, fragments = product_geometry(z, b, parent_id, channel_id)
        value, status = energy('AIMNet2', numbers, x, cfg)
        out = dict(status=status, energy_kcal=float(value) if status == 'ok' and np.isfinite(value) else None,
                   fragments=fragments)
    except (ValueError, RuntimeError, KeyError) as exc:
        out = dict(status=type(exc).__name__ + ':' + str(exc), energy_kcal=None)
    out['wall_s'] = time.perf_counter()-start
    return out


def file_hash(path):
    h = sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='JSONL parent_id/channel_id/z/b/b_r prepared under xtbflow')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--smoke', action='store_true', help='Only workflow pass/fail; no scientific outputs')
    p.add_argument('--max-cpu-seconds', type=float, required=True, help='Remaining acquisition CPU budget; stop without retry')
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU stage must explicitly disable CUDA')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise RuntimeError('Dirty source')
    if a.out.exists():
        raise FileExistsError(a.out)
    cfgpath = Path('configs/review/oracle_benchmark.json')
    cfg = json.loads(cfgpath.read_text())['xtb']
    model = Path(os.environ['REVIEW_AIMNET_MODELS'])/'aimnet2_wb97m_d3_0.pt'
    if file_hash(model) != 'f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28':
        raise ValueError('Model hash mismatch: stop')
    import rdkit
    manifest = dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    dirty=False, slurm_job_id=None, smoke=a.smoke, rdkit=rdkit.__version__,
                    config_sha256=file_hash('configs/explore/enum_thermo.json'),
                    input_sha256={str(a.input):file_hash(a.input),str(model):file_hash(model),str(cfgpath):file_hash(cfgpath)})
    anchor = {}
    count = 0
    started = time.perf_counter()
    cpu_started = time.process_time()
    with a.out.open('x') as output, a.input.open() as handle:
        for line in handle:
            if time.process_time()-cpu_started >= a.max_cpu_seconds:
                raise RuntimeError('CPU budget reached; stop and inspect partial output')
            row = json.loads(line)
            pid = row['parent_id']
            if pid not in anchor:
                anchor[pid] = single_energy(row['z'],row['b_r'],pid,'reactant',cfg)
            product = single_energy(row['z'],row['b'],pid,row['channel_id'],cfg)
            reactant = anchor[pid]
            ok = product['energy_kcal'] is not None and reactant['energy_kcal'] is not None
            if a.smoke:
                output.write(json.dumps(dict(parent_id=pid,channel_id=row['channel_id'],workflow_completed=True,
                                             product_status=product['status'],reactant_status=reactant['status']),sort_keys=True)+'\n')
            else:
                output.write(json.dumps(dict(parent_id=pid,channel_id=row['channel_id'],product=product,reactant=reactant,
                                             delta_e_kcal=product['energy_kcal']-reactant['energy_kcal'] if ok else None),sort_keys=True)+'\n')
            output.flush()
            count += 1
    manifest.update(rows=count,parents=len(anchor),wall_s=time.perf_counter()-started,cpu_s=time.process_time()-cpu_started,
                    output_sha256=file_hash(a.out))
    a.out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print('Workflow completed; manifest written. No scientific summary printed.')


if __name__ == '__main__':
    main()
