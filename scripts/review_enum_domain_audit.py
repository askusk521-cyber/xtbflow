"""Training-only domain audit, before freezing any evaluation analysis."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

import numpy as np

from xtbflow.v1.data import file_hash, write_json


def event_domain(z, br, bp):
    """Return exact domain constraints, including nonbonding electrons."""
    br, bp = np.asarray(br, dtype=int), np.asarray(bp, dtype=int)
    delta = bp - br
    off = delta.copy()
    np.fill_diagonal(off, 0)
    upper = off[np.triu_indices(len(z), 1)]
    diag = np.diag(delta)
    touched = np.any(off != 0, axis=1)
    valence = np.array([{1: 1, 6: 4, 7: 5, 8: 6}[int(v)] for v in z])
    charges = valence - bp.sum(axis=1)
    return dict(broken=int(-upper[upper < 0].sum()),
                formed=int(upper[upper > 0].sum()),
                diagonal_atoms=int(np.count_nonzero(diag)),
                charged_atoms=int(np.count_nonzero(charges)),
                diagonal_allowed=bool(np.isin(diag, [-2, 0, 2]).all()),
                touched_only=bool(np.all((diag == 0) | touched)),
                conserved=bool(delta.sum() == 0),
                charge_allowed=bool(np.all(np.abs(charges) <= 1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = Path('configs/review/enum_domain_audit.json')
    manifest = json.loads((args.data / 'split_manifest.json').read_text())
    ids = set(manifest['train']['parent_ids'])
    parents = {p['parent_id']: p for p in json.loads(
        (args.data / 'parent_catalog.json').read_text()) if p['parent_id'] in ids}
    events = {}
    with (args.data / 'reference_catalog.jsonl').open() as handle:
        for line in handle:
            # Only training rows are inspected; no evaluation geometry or labels.
            row = json.loads(line)
            if row['parent_id'] not in ids:
                continue
            p = parents[row['parent_id']]
            key = (row['parent_id'], row['channel_id'])
            events[key] = event_domain(p['atomic_numbers'], p['b_r'], row['b_p'])
    coverage = {}
    for size in (2, 3):
        inside = [e for e in events.values() if e['broken'] <= size and
                  e['formed'] <= size and all(e[k] for k in
                  ('diagonal_allowed', 'touched_only', 'conserved', 'charge_allowed'))]
        coverage[f'b{size}f{size}'] = dict(covered=len(inside), total=len(events),
                                         fraction=len(inside) / len(events))
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain']).strip())
    if dirty:
        raise RuntimeError('Formal audit requires a clean source clone')
    write_json(args.output, dict(exploratory=True, training_parents=len(parents),
        unique_events=len(events), coverage=coverage,
        size_counts=dict(sorted(Counter(f"{e['broken']},{e['formed']},{e['diagonal_atoms']},{e['charged_atoms']}"
                                        for e in events.values()).items())),
        constraint_failures={k: sum(not e[k] for e in events.values()) for k in
                             ('diagonal_allowed', 'touched_only', 'conserved', 'charge_allowed')},
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        dirty=False, config_sha256=file_hash(config)))


if __name__ == '__main__':
    main()
