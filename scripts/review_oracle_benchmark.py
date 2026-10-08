"""Frozen exploratory oracle benchmark, resumable single-point acquisition."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import time
import os
import re
import tempfile

import numpy as np

from xtbflow.m0.t1x_data import T1xCache
from xtbflow.v1.data import file_hash, write_json
from xtbflow.v1.geometry_information import (xtb_energy_kcal, event_table, ranking_metrics,
                                           within_parent_rho, HARTREE_TO_KCAL, min_distance)
from xtbflow.calculators.xtb_oracle import BOHR_IN_ANGSTROM, KELVIN_TO_HARTREE
from xtbflow.v1.metrics import cluster_summary


_AIMNET = {}


def energy(method, z, x, cfg):
    if method in ('AIMNet2', 'AIMNet2-rxn'):
        import torch
        from aimnet.calculators import AIMNet2Calculator
        if method not in _AIMNET:
            filename = 'aimnet2_wb97m_d3_0.pt' if method == 'AIMNet2' else 'aimnet2_rxn_0.pt'
            _AIMNET[method] = AIMNet2Calculator(str(Path(os.environ['REVIEW_AIMNET_MODELS'])/filename), device='cpu')
        if min_distance(x) < cfg['min_distance_angstrom']:
            return float('nan'), 'collapsed'
        try:
            result = _AIMNET[method]({'coord':torch.tensor(np.asarray(x),dtype=torch.float32),
                'numbers':torch.tensor(z,dtype=torch.int64), 'charge':torch.tensor([0.])}, forces=False)
            value = float(result['energy'].detach().reshape(-1)[0])*23.060547830619
            return (value,'ok') if np.isfinite(value) else (float('nan'),'nonfinite')
        except (RuntimeError,ValueError) as exc:
            return float('nan'), 'aimnet_failed:'+type(exc).__name__
    if method == 'g-xTB':
        symbols = {1:'H',6:'C',7:'N',8:'O'}
        if min_distance(x) < cfg['min_distance_angstrom']:
            return float('nan'), 'collapsed'
        with tempfile.TemporaryDirectory(prefix='review-gxtb-') as tmp:
            xyz = Path(tmp)/'input.xyz'
            xyz.write_text(str(len(z))+'\n\n'+''.join(f'{symbols[int(a)]} {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}\n' for a,p in zip(z,x)))
            try:
                run = subprocess.run([os.environ['REVIEW_GXTB_BINARY'], str(xyz), '--gxtb', '--chrg', '0', '--acc', '1.0'], cwd=tmp, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                return float('nan'), 'timeout'
            matches = re.findall(r'TOTAL ENERGY\s+([-+0-9.Ee]+)', run.stdout)
            if run.returncode or not matches:
                return float('nan'), 'gxtb_failed'
            return float(matches[-1])*HARTREE_TO_KCAL, 'ok'
    if method == 'GFN2-xTB':
        return xtb_energy_kcal(z, x, cfg)
    if min_distance(x) < cfg['min_distance_angstrom']:
        return float('nan'), 'collapsed'
    from tblite.interface import Calculator
    try:
        calc = Calculator(method, np.asarray(z, dtype=np.int32),
                          np.asarray(x, dtype=float) / BOHR_IN_ANGSTROM, charge=0, uhf=0)
        calc.set('verbosity', 0)
        calc.set('accuracy', cfg['accuracy'])
        calc.set('max-iter', cfg['max_iterations'])
        calc.set('temperature', cfg['electronic_temperature_kelvin'] * KELVIN_TO_HARTREE)
        return float(calc.singlepoint().get('energy')) * HARTREE_TO_KCAL, 'ok'
    except (RuntimeError, ValueError) as exc:
        return float('nan'), 'scf_failed:' + type(exc).__name__


def acquire(args):
    cfg = json.loads(args.config.read_text())
    root = args.root
    parents = {p['parent_id']: p for p in json.loads((root/'data/parent_catalog.json').read_text())}
    refs = [json.loads(s) for s in (root/'data/reference_catalog.jsonl').read_text().splitlines()]
    rows_a = [json.loads(s) for s in args.check.read_text().splitlines()]
    cache = T1xCache(args.cache)
    by_channel = {}
    for r in refs:
        key = (r['parent_id'], r['channel_id'])
        by_channel[key] = min(by_channel.get(key, float('inf')), r['catalog_barrier_kcal'])
    selected = defaultdict(list)
    with (root/'screen/efficiency_s0.candidates.jsonl').open() as handle:
        for line in handle:
            r = json.loads(line)
            if r['arm'] == 'B0' and r['proxy_status'] in cfg['T2b']['statuses']:
                selected[r['parent_id']].append(r)
    chosen = {}
    for pid, rows in selected.items():
        for r in sorted(rows, key=lambda r: hashlib.sha256(r['candidate_id'].encode()).hexdigest())[:8]:
            chosen[r['candidate_id']] = r
    geometries = {}
    with (root/'screen/efficiency_s0/streams.jsonl').open() as handle:
        for line in handle:
            r = json.loads(line)
            # Raw stream records contain completed candidate payloads.
            candidates = r.get('candidates', [r])
            for c in candidates:
                if c.get('candidate_id') in chosen:
                    geometries[c['candidate_id']] = c['x']
    if set(chosen) != set(geometries):
        raise RuntimeError(f'geometry join incomplete: {len(geometries)} / {len(chosen)}')
    jobs = []
    for r in rows_a:
        reaction = cache.reaction(r['cache_index'])
        jobs.append(dict(part='T2a', id=r['reference_id'], parent_id=r['parent_id'],
                         channel_id=r['channel_id'], target=r['catalog_barrier_kcal'],
                         expected_gfn2=r['xtb_ts'], z=reaction['z'], x=reaction['x_ts']))
    for cid, r in sorted(chosen.items()):
        p = parents[r['parent_id']]
        jobs.append(dict(part='T2b', id=cid, parent_id=r['parent_id'],
                         channel_id=r['predicted_channel_id'],
                         target=by_channel[(r['parent_id'], r['predicted_channel_id'])],
                         z=p['atomic_numbers'], x=geometries[cid]))
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.out/'raw.jsonl'
    done = set()
    if raw.exists():
        for line in raw.read_text().splitlines():
            r = json.loads(line)
            done.add((r['method'], r['part'], r['id']))
    anchors = {}
    with raw.open('a') as handle:
        for method in args.methods.split(','):
            for job in jobs:
                key = (method, job['part'], job['id'])
                if key in done:
                    continue
                pid = job['parent_id']
                anchor_key = (method, pid)
                if anchor_key not in anchors:
                    p = parents[pid]
                    anchors[anchor_key] = energy(method, p['atomic_numbers'], p['x_r'], cfg['xtb'])
                start = time.monotonic()
                value, status = energy(method, job['z'], job['x'], cfg['xtb'])
                er, er_status = anchors[anchor_key]
                barrier = value-er
                row = {k: v for k, v in job.items() if k not in ('z', 'x')}
                row.update(method=method, barrier=barrier if np.isfinite(barrier) else None,
                           status=status, anchor_status=er_status, seconds=time.monotonic()-start,
                           split_group=parents[pid]['split_group'])
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
                handle.flush()
    write_json(args.out/'manifest.json', dict(source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(), dirty=bool(subprocess.check_output(
        ['git', 'status', '--porcelain']).strip()), config_sha256=file_hash(args.config),
        raw_sha256=file_hash(raw), candidates=len(chosen), references=len(rows_a)))


def analyze(args):
    rows = [json.loads(s) for s in (args.out/'raw.jsonl').read_text().splitlines()]
    groups = {r['parent_id']: r['split_group'] for r in rows}
    out = {'exploratory': True, 'metrics': {}, 'comparisons': {}}
    rhos = {}
    methods = sorted({r['method'] for r in rows})
    for part in ('T2a', 'T2b'):
        for method in methods:
            subset = [r for r in rows if r['part'] == part and r['method'] == method]
            good = [r for r in subset if r['barrier'] is not None]
            by = defaultdict(list)
            for r in good:
                by[r['parent_id']].append(dict(r, catalog_barrier_kcal=r['target']))
            if part == 'T2a':
                metrics = {}
                for pid, records in sorted(by.items()):
                    _, target, pred = event_table(records, 'barrier')
                    m = ranking_metrics(target, pred)
                    if m is not None:
                        metrics[pid] = m
                rho = {pid: m['rho'] for pid, m in metrics.items()}
            else:
                rho = within_parent_rho([r['barrier'] for r in good], [r['target'] for r in good],
                                        [r['parent_id'] for r in good])
                metrics = {pid: {'rho': v} for pid, v in rho.items()}
            result = {'calls': len(subset), 'failures': len(subset)-len(good), 'parents': metrics}
            for metric in ('rho', 'top1', 'concordance'):
                ids = sorted(pid for pid, m in metrics.items() if m.get(metric) is not None)
                if ids:
                    result[metric] = cluster_summary([metrics[p][metric] for p in ids], [groups[p] for p in ids])
            errors = np.array([r['barrier']-r['target'] for r in good])
            result.update(mae=float(np.abs(errors).mean()), median_signed_error=float(np.median(errors)),
                          mean_call_seconds=float(np.mean([r['seconds'] for r in subset])))
            name = part+'/'+method
            out['metrics'][name] = result
            rhos[name] = rho
    for part, method in ((part, method) for part in ('T2a', 'T2b') for method in methods if method != 'GFN2-xTB'):
        a, b = rhos[part+'/'+method], rhos[part+'/GFN2-xTB']
        ids = sorted(a.keys() & b.keys())
        summary = cluster_summary([a[p]-b[p] for p in ids], [groups[p] for p in ids])
        lo, hi = summary['ci_two95']
        summary['reading'] = 'BETTER_THAN_GFN2' if lo > 0 else 'WORSE_THAN_GFN2' if hi < 0 else 'INCONCLUSIVE'
        out['comparisons'][part+'/'+method+'-minus-GFN2'] = summary
    diffs = [abs(r['barrier']-r['expected_gfn2']) for r in rows if r['part']=='T2a' and r['method']=='GFN2-xTB' and r['barrier'] is not None]
    anchor = out['metrics']['T2a/GFN2-xTB']['rho']
    out['reproduction'] = dict(max_barrier_error=max(diffs), rho=anchor['estimate'], ci_two95=anchor['ci_two95'],
        passed=bool(len(diffs)==558 and max(diffs)<=1e-6 and abs(anchor['estimate']-0.7491015704725383)<1e-9 and
        np.max(np.abs(np.array(anchor['ci_two95'])-[0.6524459560799384,0.8457571848651383]))<1e-9))
    write_json(args.out/'results.json', out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['acquire', 'analyze'])
    p.add_argument('--root', type=Path, default=Path('/home/lhshen/xtbflow-runs/v1a-20261007'))
    p.add_argument('--check', type=Path, default=Path('/home/lhshen/xtbflow-runs/explore-geoinfo-20261008/full-68f5733/heads/check_a_rows.jsonl'))
    p.add_argument('--cache', type=Path, default=Path('/home/lhshen/data/t1x/t1x_m0_v1.npz'))
    p.add_argument('--config', type=Path, default=Path('configs/review/oracle_benchmark.json'))
    p.add_argument('--methods', default='GFN2-xTB,GFN1-xTB')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    (acquire if args.mode == 'acquire' else analyze)(args)


if __name__ == '__main__':
    main()
