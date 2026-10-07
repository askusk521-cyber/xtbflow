"""Sample reactant-only M0 candidates using frozen EMA checkpoints (coordinates in Å)."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from xtbflow.m0.batching import T1xDataset, collate
from xtbflow.m0.model import ReactionFlowNet
from xtbflow.m0.sampler import reactant_view, sample_cascade, sample_joint
from xtbflow.m0.t1x_data import SPLITS, T1xCache


def load_net(path, role, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt['role'] != role:
        raise ValueError(f'Expected {role}, got {ckpt["role"]}')
    net = ReactionFlowNet(role, **ckpt['model_cfg']).to(device)
    net.load_state_dict(ckpt['ema'])
    return net.eval(), ckpt


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--arm', choices=('joint', 'cascade'), required=True)
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--event-checkpoint', type=Path)
    ap.add_argument('--geometry-checkpoint', type=Path)
    ap.add_argument('--split', choices=('val', 'test'), required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n-samples', type=int)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--frozen-selection', type=Path, default=Path('docs/evidence/m0/frozen_selection.json'))
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text())
    if a.out.exists():
        raise FileExistsError(f'Refusing to overwrite {a.out}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if a.arm == 'joint':
        if a.checkpoint is None:
            ap.error('--checkpoint required for joint')
        net, meta = load_net(a.checkpoint, 'joint', device)
        paths = [a.checkpoint]
        sigma_b, sigma_x = meta['sigma_b'], meta['sigma_x']
    else:
        if a.event_checkpoint is None or a.geometry_checkpoint is None:
            ap.error('both cascade checkpoints required')
        event, em = load_net(a.event_checkpoint, 'event', device)
        geom, gm = load_net(a.geometry_checkpoint, 'geometry', device)
        if em['step'] != gm['step']:
            raise ValueError('Cascade checkpoints must have identical step')
        meta = em
        paths = [a.event_checkpoint, a.geometry_checkpoint]
        sigma_b, sigma_x = em['sigma_b'], gm['sigma_x']
        if gm['seed'] != a.seed:
            raise ValueError('Geometry checkpoint seed mismatch')
    if meta['seed'] != a.seed:
        raise ValueError('Checkpoint seed mismatch')
    if a.split == 'test':
        if a.limit is not None:
            raise ValueError('Partial test sampling is forbidden')
        frozen = json.loads(a.frozen_selection.read_text())
        # Freeze format is defined by the selection producer; require every checkpoint hash.
        frozen_text = json.dumps(frozen)
        for path in paths:
            if sha256(path) not in frozen_text:
                raise ValueError(f'Checkpoint absent from frozen selection: {path}')
        import subprocess
        tracked = subprocess.run(['git', 'log', '-1', '--format=%H', '--', str(a.frozen_selection)],
                                 capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--', str(a.frozen_selection)],
                               capture_output=True, text=True, check=True).stdout.strip()
        if not tracked or dirty:
            raise ValueError('Frozen selection must already be committed and unchanged')
    sc = cfg['sample']
    ns = a.n_samples or sc['n_samples_val' if a.split == 'val' else 'n_samples_test']
    if ns < 1:
        raise ValueError('n_samples must be positive')
    cache = T1xCache(Path(os.path.expandvars(cfg['data']['cache'])))
    ds = T1xDataset(cache, SPLITS[a.split], a.limit)
    loader = DataLoader(ds, batch_size=sc['queries_per_batch'], collate_fn=collate)
    gen = torch.Generator(device=device).manual_seed(10_000+a.seed)
    query, sample, sizes, coords, bs = [], [], [], [], []
    for batch in loader:
        rv = reactant_view({k: v.to(device) for k, v in batch.items()})
        args = (rv, ns, sc['nfe'], sigma_b, sigma_x, gen)
        result = sample_joint(net, *args) if a.arm == 'joint' else sample_cascade(event, geom, *args)
        rr = result['rv']
        for k, n in enumerate(rr['atom_mask'].sum(1).tolist()):
            n = int(n)
            query.append(int(rr['index'][k]))
            sample.append(k % ns)
            sizes.append(n)
            coords.append(result['x'][k, :n].cpu().numpy())
            bs.append(result['b_raw'][k, :n, :n].cpu().numpy().reshape(-1))
    if not sizes:
        raise ValueError('No queries')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('xb') as f:
        np.savez_compressed(f, query_index=np.array(query), sample_id=np.array(sample),
                            n_atoms=np.array(sizes), atom_off=np.r_[0, np.cumsum(sizes)],
                            x=np.concatenate(coords).astype(np.float32),
                            b_raw=np.concatenate(bs).astype(np.float32))
    print(json.dumps(dict(path=str(a.out), queries=len(ds), candidates=len(sizes),
                          arm=a.arm, seed=a.seed, split=a.split)))


if __name__ == '__main__':
    main()
