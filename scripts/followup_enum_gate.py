"""Exact b2f2 subset count and three-parent consistency gate (no physical scores)."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from xtbflow.v1.enumeration import enumerate_events


def b2f2_mask(b, br):
    delta = np.asarray(b, dtype=np.int64) - np.asarray(br, dtype=np.int64)
    i, j = np.triu_indices(len(br), 1)
    edits = delta[:, i, j]
    return (np.maximum(-edits, 0).sum(axis=1) <= 2) & (np.maximum(edits, 0).sum(axis=1) <= 2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--enumdir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Dirty execution source')
    parents = {r['parent_id']: r for r in json.loads((a.root/'data/parent_catalog.json').read_text())}
    ids = sorted(json.loads((a.root/'formal/freeze.json').read_text())['parent_ids'])
    metadata = {pid: json.loads((a.enumdir/(pid+'.json')).read_text()) for pid in ids}
    smallest = sorted(ids, key=lambda pid: (metadata[pid]['n_enum'], pid))[:3]
    counts, gates, inputs = {}, [], {}
    t0 = time.perf_counter()
    for pid in ids:
        path = a.enumdir/(pid+'.npz')
        inputs[str(path)] = sha256(path.read_bytes()).hexdigest()
        with np.load(path) as data:
            mask = b2f2_mask(data['b'], parents[pid]['b_r'])
            counts[pid] = int(mask.sum())
            if pid in smallest:
                filtered = set(data['channels'][mask].tolist())
                parent = parents[pid]
                direct = {channel for channel, _ in enumerate_events(parent['atomic_numbers'], parent['b_r'], parent['permutations'], size=2)}
                gates.append(dict(parent_id=pid, filtered=len(filtered), enumerated=len(direct),
                                  passed=filtered == direct, missing=sorted(direct-filtered), extra=sorted(filtered-direct)))
                if filtered != direct:
                    a.out.write_text(json.dumps(dict(passed=False, gates=gates), indent=2)+'\n')
                    raise RuntimeError('b2f2 consistency gate failed: stop')
    output = dict(passed=True, gates=gates, counts=counts, total=sum(counts.values()),
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  dirty=False, input_sha256=inputs, slurm_job_id=None, wall_s=time.perf_counter()-t0)
    a.out.write_text(json.dumps(output, indent=2, sort_keys=True)+'\n')
    print('PASS: three exact enumeration comparisons; counts recorded; no physical score acquired.')


if __name__ == '__main__':
    main()
