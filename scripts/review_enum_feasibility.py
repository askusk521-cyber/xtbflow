"""Five training-parent feasibility audit; bounded by the Slurm wall clock."""
import argparse
import json
from pathlib import Path
import time

from rdkit import RDLogger

from xtbflow.v1.enumeration import enumerate_events
from xtbflow.v1.data import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    ids = sorted(json.loads((args.data / 'split_manifest.json').read_text())['train']['parent_ids'])[:5]
    parents = {p['parent_id']: p for p in json.loads((args.data / 'parent_catalog.json').read_text())
               if p['parent_id'] in ids}
    RDLogger.DisableLog('rdApp.warning')
    RDLogger.DisableLog('rdApp.error')
    rows = []
    for pid in ids:
        p = parents[pid]
        start = time.monotonic()
        count = 0
        for _, _ in enumerate_events(p['atomic_numbers'], p['b_r'], p['permutations'], 3):
            count += 1
            if count % 1000 == 0:
                print(json.dumps(dict(parent_id=pid, events=count, seconds=time.monotonic()-start)), flush=True)
        rows.append(dict(parent_id=pid, legal_events=count, seconds=time.monotonic()-start))
        write_json(args.out, dict(complete=len(rows) == 5, parents=rows))
        print(json.dumps(rows[-1]), flush=True)


if __name__ == '__main__':
    main()
