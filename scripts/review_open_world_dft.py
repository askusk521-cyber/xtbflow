"""Frozen exploratory O2 DFT plan and single-item execution.

Inputs: frozen ten-parent config, screen catalogue and immutable O1 verdicts.
Units: Angstrom coordinates, Hartree electronic energies, neutral singlet.
GPU execution is Slurm-only; one attempt per item, protocol built-in retries only.
The serial dispatcher owns the 16 GPU-hour aggregate budget and 120-minute cap.
"""
import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

PROTOCOL = 'v1b-h-gpu4pyscf-3'


def digest(path):
    h = sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def select_sample(results, verdicts):
    """Reproduce selection from O1 evidence, never from DFT outcomes."""
    baselines = {r['parent_id']: r for r in verdicts
                 if r['group'] == 'BASELINE' and r['status'] == 'STRICT_JOINT_GRAPH_VALID'}
    candidates = {r['id']: r for r in verdicts
                  if r['group'] == 'OUTSIDE_CATALOGUE_EVENT' and r['status'] == 'STRICT_JOINT_GRAPH_VALID'}
    pool = []
    for r in results['delta_ea']['rows']:
        pid, cid = r['parent_id'], r['candidate_id']
        if cid not in candidates or pid not in baselines:
            raise ValueError('O1 delta rows do not satisfy strict paired gate')
        pool.append(dict(parent_id=pid, candidate_id=cid,
                         reference_id=baselines[pid]['id'], delta_ea_xtb_kcal=r['delta_ea_kcal']))
    if len({r['parent_id'] for r in pool}) != 12:
        raise ValueError('Expected twelve eligible O1 parents')
    selected, seen = [], set()
    for r in sorted(pool, key=lambda r: (r['delta_ea_xtb_kcal'], sha256(r['candidate_id'].encode()).hexdigest())):
        if r['parent_id'] not in seen:
            selected.append(r)
            seen.add(r['parent_id'])
        if len(selected) == 10:
            break
    return selected


def reference_start(parent, reference):
    """Exactly the V1b D1 reference atom-order mapping."""
    perm = np.asarray(parent['permutations'][0], dtype=int)
    return (np.asarray(reference['x_ts'])[perm].tolist(),
            np.asarray(reference['b_p'])[np.ix_(perm, perm)].tolist())


def source_state():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    if dirty:
        raise RuntimeError('Execution requires a clean source checkout')
    return dict(source_commit=commit, dirty=False)


def plan(a):
    state = source_state()
    config = json.loads(a.config.read_text())
    ep = Path('docs/evidence/review_open_world')
    results_path, verdicts_path = ep / 'results.json', ep / 'verdicts.csv'
    with verdicts_path.open() as handle:
        selected = select_sample(json.loads(results_path.read_text()), list(csv.DictReader(handle)))
    if selected != config['sample'] or config['protocol_version'] != PROTOCOL:
        raise ValueError('Frozen sample/protocol mismatch')
    for path, expected in config['sources'].items():
        if digest(path) != expected:
            raise ValueError('O1 selection evidence hash mismatch: ' + path)
    files = [a.config, results_path, verdicts_path, a.root / 'data/parent_catalog.json',
             a.root / 'data/reference_catalog.jsonl', a.root / 'data/split_manifest.json',
             a.root / 'screen/efficiency_s0.candidates.jsonl', Path('src/xtbflow/v1/qc_protocol.py')]
    parents = {r['parent_id']: r for r in json.loads(files[3].read_text())}
    split = json.loads(files[5].read_text())
    screen = set(split['screen_frozen']['parent_ids'])
    refs = {r['reference_id']: r for r in map(json.loads, files[4].open())}
    wanted = {r['candidate_id'] for r in selected}
    candidates = {r['candidate_id']: r for r in map(json.loads, files[6].open()) if r['candidate_id'] in wanted}
    items = []
    for row in selected:
        pid, cid, rid = row['parent_id'], row['candidate_id'], row['reference_id']
        if pid not in screen:
            raise ValueError('Non-screen parent')
        p, c, r = parents[pid], candidates[cid], refs[rid]
        if c['parent_id'] != pid or r['parent_id'] != pid:
            raise ValueError('Parent mapping mismatch')
        cv = a.xtb / 'OUTSIDE_CATALOGUE_EVENT' / cid / 'verdict.json'
        bv = a.xtb / 'BASELINE' / rid / 'verdict.json'
        files.extend([cv, bv])
        v, b = json.loads(cv.read_text()), json.loads(bv.read_text())
        if any(t['status'] != 'STRICT_JOINT_GRAPH_VALID' for t in (v, b)):
            raise ValueError('Strict O1 chain gate failed')
        delta = (v['ts']['energy_hartree'] - b['ts']['energy_hartree']) * 627.509474
        if abs(delta - row['delta_ea_xtb_kcal']) > 1e-8:
            raise ValueError('O1 delta energy mismatch')
        xref, bref = reference_start(p, r)
        common = dict(parent_id=pid, query_id=p['query_id'], z=p['atomic_numbers'],
                      charge=0, multiplicity=1, coordinate_unit='Angstrom')
        for kind, x, bp in [('anchor', p['x_r'], None), ('reference', xref, bref), ('candidate', v['ts']['x'], c['b_dec'])]:
            item = dict(common, kind=kind, item_id=sha256((kind + '/' + pid).encode()).hexdigest()[:16], x_start=x)
            if kind != 'anchor':
                item.update(b_r=p['b_r'], perms=p['permutations'], b_predicted=bp)
            if kind == 'reference':
                item.update(reference_id=rid, catalog_barrier_kcal=r['catalog_barrier_kcal'])
            if kind == 'candidate':
                item.update(candidate_id=cid, delta_ea_xtb_kcal=row['delta_ea_xtb_kcal'])
            items.append(item)
    # Put the smallest parent first, retaining frozen selection order for ties.
    order = sorted(selected, key=lambda row: len(parents[row['parent_id']]['atomic_numbers']))
    rank = {r['parent_id']: j for j, r in enumerate(order)}
    items.sort(key=lambda i: (rank[i['parent_id']], ('anchor', 'reference', 'candidate').index(i['kind'])))
    if a.out.exists():
        raise FileExistsError(a.out)
    dump(a.out, dict(schema='xtbflow-open-world-o2-plan/1', **state, protocol_version=PROTOCOL,
                     config_sha256=digest(a.config), slurm_job_id=None,
                     input_sha256={str(p.resolve()): digest(p) for p in files}, items=items))
    print('Plan written: 30 items; no quantum calculation executed.')


def run(a):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU execution requires Slurm')
    state = source_state()
    from xtbflow.v1 import qc_protocol as qc
    if qc.PROTOCOL_VERSION != PROTOCOL:
        raise ValueError('Unexpected certification protocol')
    planned = json.loads(a.plan.read_text())
    for path, expected in planned['input_sha256'].items():
        if digest(path) != expected:
            raise ValueError('Input hash mismatch: ' + path)
    item = next(i for i in planned['items'] if i['item_id'] == a.item)
    work = a.work.resolve() / (item['kind'] + '_' + item['item_id'])
    work.mkdir(parents=True, exist_ok=False)  # Includes interrupted jobs: no external retry.
    started = time.time()
    manifest = dict(**state, config_sha256=planned['config_sha256'], plan_sha256=digest(a.plan),
                    input_sha256=planned['input_sha256'], slurm_job_id=os.environ['SLURM_JOB_ID'],
                    protocol_version=PROTOCOL, started_unix=started, item=item)
    dump(work / 'manifest.json', manifest)
    meter, t0 = qc.Meter(), time.perf_counter()
    try:
        import gpu4pyscf
        import pyscf
        import geometric
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True, check=True).stdout.strip()
        dump(work / 'environment.json', dict(gpu=gpu, gpu4pyscf=gpu4pyscf.__version__,
                                             pyscf=pyscf.__version__, geometric=geometric.__version__,
                                             slurm_job_id=os.environ['SLURM_JOB_ID']))
        z, x = np.asarray(item['z']), np.asarray(item['x_start'], dtype=float)
        method = qc.Method(device='gpu')
        if item['kind'] == 'anchor':
            out = qc.minimum_stage(z, x, method, meter, work, 'anchor')
            out.update(protocol_version=PROTOCOL, method=method.as_dict())
            qc.finish(out, meter, t0, work)
        else:
            out = qc.certification_chain(z, x, item['b_r'], item['perms'], item['b_predicted'], method, work, meter)
    except Exception as exc:
        # Preserve exception evidence and stop dispatcher; do not invent a chemical verdict.
        dump(work / 'unexpected_failure.json', dict(type=type(exc).__name__, message=str(exc),
                                                   stage_timings=meter.rows, wall_s=time.perf_counter()-t0))
        raise
    finally:
        dump(work / 'execution.json', dict(started_unix=started, ended_unix=time.time(),
                                           wall_s=time.perf_counter()-t0, slurm_job_id=os.environ['SLURM_JOB_ID']))
    print(json.dumps(dict(item=a.item, status=out.get('strict_status', out.get('status')))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--xtb', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('configs/review/open_world_o2.json'))
    p.add_argument('--out', type=Path, required=True)
    r = sub.add_parser('run')
    r.add_argument('--plan', type=Path, required=True)
    r.add_argument('--item', required=True)
    r.add_argument('--work', type=Path, required=True)
    a = parser.parse_args()
    plan(a) if a.cmd == 'plan' else run(a)


if __name__ == '__main__':
    main()
