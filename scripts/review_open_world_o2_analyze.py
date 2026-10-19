"""Deterministic analysis of the O2 open-world DFT run (handoff section 8.1).

Implements the six metrics frozen in configs/review/open_world_o2.json (pushed at
97169c4, before any DFT job) with the operational choices written down in
configs/review/open_world_o2_analysis.json before any barrier was computed.

Inputs are read only:
- the frozen plan o2-plan.json (items, delta_Ea_xTB, catalogue barriers);
- each item's work directory o2-run/<kind>_<item_id>/: verdict.json from the
  unchanged qc_protocol, or unexpected_failure.json for a chain exception;
- the dispatcher ledgers o2-jobs/ledger.json, ledger_v2.json, ledger_v4.json and
  o2-run-preqc-failed/quarantine.json (GPU accounting);
- the V1a parent catalogue (split_group, anchor_reference_id).

Energies are electronic Hartree (omegaB97X/6-31G(d), protocol v1b-h-gpu4pyscf-3);
barriers and differences are kcal/mol. Nothing here runs quantum chemistry.

  analyze  --run-root R --parents P --out results.json
  plot     --results results.json --out delta_ea_scatter.svg
  manifest --run-root R --parents P --out-dir D   (manifest.json, SHA256SUMS.remote)
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_open_world_analyze import wilson  # noqa: E402
from xtbflow.v1.data import file_hash, write_json  # noqa: E402
from xtbflow.v1.metrics import cluster_summary  # noqa: E402

ANALYSIS_CONFIG = Path('configs/review/open_world_o2_analysis.json')
LEDGERS = (('v1', 'ledger.json'), ('v2', 'ledger_v2.json'), ('v4', 'ledger_v4.json'))
QC_KINDS = ('scf', 'gradient', 'hessian')


def load_item(work):
    """Status and raw record of one work directory.

    Anchors report verdict['status'], chains verdict['strict_status']. A chain that
    raised has no verdict; its status is CHAIN_EXCEPTION:<type>:<message>.
    """
    verdict, failure = work / 'verdict.json', work / 'unexpected_failure.json'
    if verdict.exists():
        v = json.loads(verdict.read_text())
        return dict(status=v.get('strict_status', v.get('status')), verdict=v, failure=None)
    if failure.exists():
        f = json.loads(failure.read_text())
        return dict(status=f"CHAIN_EXCEPTION:{f['type']}:{f['message']}", verdict=None, failure=f)
    return dict(status='MISSING', verdict=None, failure=None)


def stage_summary(rows):
    """Call counts and wall seconds from qc_protocol Meter rows.

    'geometric' rows wrap the scf/gradient rows of the same optimisation, so the
    two are reported separately and never added together.
    """
    calls = Counter(r['kind'] for r in rows if r['kind'] in QC_KINDS)
    qc_wall, geo_wall = {}, {}
    for r in rows:
        target = qc_wall if r['kind'] in QC_KINDS else geo_wall if r['kind'] == 'geometric' else None
        if target is not None:
            target[r['stage']] = target.get(r['stage'], 0.0) + float(r['wall_s'])
    return dict(calls={k: calls.get(k, 0) for k in QC_KINDS}, qc_kernel_wall_s=sum(qc_wall.values()),
                stage_qc_kernel_wall_s=dict(sorted(qc_wall.items())), stage_geometric_wall_s=dict(sorted(geo_wall.items())))


def ts_energy(record):
    v = record['verdict']
    return None if v is None or 'ts' not in v else v['ts'].get('energy_hartree')


def gpu_jobs(jobs_dir, quarantine):
    """Every Slurm job the O2 dispatchers submitted, from their ledgers (no duplicates)."""
    quarantined = {q['job_id'] for q in quarantine}
    jobs = {}
    for version, name in LEDGERS:
        path = jobs_dir / name
        if not path.exists():
            continue
        for r in json.loads(path.read_text()):
            if not r.get('job_id'):
                continue  # PRIOR_ATTEMPT rows point back to jobs counted in an earlier ledger
            jobs[r['job_id']] = dict(job_id=r['job_id'], dispatcher=version, item_id=r['item_id'], kind=r['kind'],
                                     slurm_state=r.get('slurm_state', r.get('state')), runtime_s=int(r['runtime_s']),
                                     outcome=r.get('outcome', r.get('state')), detail=r.get('detail'),
                                     category='pre_qc_infrastructure' if r['job_id'] in quarantined else 'item_attempt')
    return [jobs[k] for k in sorted(jobs, key=int)]


def paired_rows(plan_items, records, anchors, parents, cfg):
    k = cfg['hartree_to_kcal']
    by_parent = {}
    for item in plan_items:
        by_parent.setdefault(item['parent_id'], {})[item['kind']] = item
    rows, excluded = [], []
    for pid in sorted(by_parent):
        items = by_parent[pid]
        cand, ref = records[items['candidate']['item_id']], records[items['reference']['item_id']]
        passed = {kind: records[items[kind]['item_id']]['status'] == cfg['pass_status'][kind] for kind in ('candidate', 'reference')}
        if not all(passed.values()):
            excluded.append(dict(parent_id=pid, candidate_status=cand['status'], reference_status=ref['status']))
            continue
        anchor = anchors.get(pid)
        ec, er = ts_energy(cand), ts_energy(ref)
        delta = (ec - er) * k
        xtb = items['candidate']['delta_ea_xtb_kcal']
        rows.append(dict(parent_id=pid, split_group=parents[pid]['split_group'], formula=parents[pid]['formula'],
                         candidate_id=items['candidate']['candidate_id'], reference_id=items['reference']['reference_id'],
                         delta_ea_dft_kcal=delta, delta_ea_xtb_kcal=xtb,
                         sign_agree=bool(np.sign(delta) == np.sign(xtb) and delta != 0 and xtb != 0),
                         candidate_barrier_vs_anchor_kcal=None if anchor is None else (ec - anchor) * k,
                         reference_barrier_vs_anchor_kcal=None if anchor is None else (er - anchor) * k,
                         candidate_local_barrier_kcal=cand['verdict'].get('local_barrier_kcal'),
                         reference_local_barrier_kcal=ref['verdict'].get('local_barrier_kcal')))
    return rows, excluded


def catalogue_rows(plan_items, records, anchors, parents, cfg):
    k, flag = cfg['hartree_to_kcal'], cfg['catalogue_check']['flag_abs_kcal']
    out = []
    for item in plan_items:
        if item['kind'] != 'reference' or records[item['item_id']]['status'] != cfg['pass_status']['reference']:
            continue
        rec, pid, cat = records[item['item_id']], item['parent_id'], item['catalog_barrier_kcal']
        anchor = anchors.get(pid)
        vs_anchor = None if anchor is None else (ts_energy(rec) - anchor) * k
        local = rec['verdict'].get('local_barrier_kcal')
        out.append(dict(parent_id=pid, reference_id=item['reference_id'],
                        same_reaction_as_anchor=item['reference_id'] == parents[pid]['anchor_reference_id'],
                        catalog_barrier_kcal=cat, barrier_vs_anchor_kcal=vs_anchor,
                        diff_vs_anchor_kcal=None if vs_anchor is None else vs_anchor - cat,
                        local_barrier_kcal=local, diff_local_kcal=None if local is None else local - cat,
                        flag_abs_gt=None if vs_anchor is None else abs(vs_anchor - cat) > flag))
    return out


def delta_summary(rows):
    groups = [r['split_group'] for r in rows]
    if len(rows) < 2 or len(set(groups)) < 2:
        return dict(status='NOT_ESTIMABLE', reason='fewer than two paired parents or split groups', n_parents=len(rows))
    return cluster_summary([r['delta_ea_dft_kcal'] for r in rows], groups)


def analyze(run_root, parents_path, analysis_config=ANALYSIS_CONFIG):
    cfg = json.loads(Path(analysis_config).read_text())
    plan = json.loads((run_root / 'o2-plan.json').read_text())
    parents = {p['parent_id']: p for p in json.loads(Path(parents_path).read_text())}
    quarantine_path = run_root / 'o2-run-preqc-failed' / 'quarantine.json'
    quarantine = json.loads(quarantine_path.read_text()) if quarantine_path.exists() else []
    jobs = gpu_jobs(run_root / 'o2-jobs', quarantine)
    records, items_out, anchors = {}, [], {}
    for item in plan['items']:
        work = run_root / 'o2-run' / f"{item['kind']}_{item['item_id']}"
        rec = load_item(work)
        records[item['item_id']] = rec
        raw = rec['verdict'] or rec['failure'] or {}
        manifest = json.loads((work / 'manifest.json').read_text()) if (work / 'manifest.json').exists() else {}
        job = manifest.get('slurm_job_id')
        runtime = next((j['runtime_s'] for j in jobs if j['job_id'] == job), None)
        if item['kind'] == 'anchor' and rec['status'] == cfg['pass_status']['anchor']:
            anchors[item['parent_id']] = rec['verdict']['energy_hartree']
        energy = rec['verdict'].get('energy_hartree') if item['kind'] == 'anchor' and rec['verdict'] else ts_energy(rec)
        items_out.append(dict(parent_id=item['parent_id'], kind=item['kind'], item_id=item['item_id'], n_atoms=len(item['z']),
                              status=rec['status'], slurm_job_id=job, slurm_runtime_s=runtime, wall_s=raw.get('wall_s'),
                              energy_hartree=energy, local_barrier_kcal=(rec['verdict'] or {}).get('local_barrier_kcal'),
                              irc_status=((rec['verdict'] or {}).get('irc') or {}).get('status'),
                              **stage_summary(raw.get('stage_timings', []))))
    certification = {}
    for kind in ('anchor', 'reference', 'candidate'):
        statuses = [r['status'] for r in items_out if r['kind'] == kind]
        certification[kind] = dict(planned=len(statuses), statuses=dict(sorted(Counter(statuses).items())),
                                   passed=wilson(sum(s == cfg['pass_status'][kind] for s in statuses), len(statuses)))
    rows, excluded = paired_rows(plan['items'], records, anchors, parents, cfg)
    deltas = [r['delta_ea_dft_kcal'] for r in rows]
    catalogue = catalogue_rows(plan['items'], records, anchors, parents, cfg)
    used = sum(j['runtime_s'] for j in jobs)
    by_cat = {}
    for j in jobs:
        c = by_cat.setdefault(j['category'], dict(jobs=0, runtime_s=0))
        c['jobs'] += 1
        c['runtime_s'] += j['runtime_s']
    return dict(
        schema='xtbflow-open-world-o2-results/1', exploratory=True, split='screen',
        protocol_version=plan['protocol_version'], frozen_config_sha256=plan['config_sha256'],
        analysis_config_sha256=file_hash(analysis_config), plan_sha256=file_hash(run_root / 'o2-plan.json'),
        items=items_out, certification=certification,
        paired=dict(rows=rows, excluded=excluded, delta_ea_dft_summary=delta_summary(rows),
                    below_zero=wilson(sum(d < 0 for d in deltas), len(deltas)),
                    at_most_plus5=wilson(sum(d <= 5.0 for d in deltas), len(deltas)),
                    sign_agreement=wilson(sum(r['sign_agree'] for r in rows), len(rows))),
        catalogue_check=dict(rows=catalogue, flag_abs_kcal=cfg['catalogue_check']['flag_abs_kcal'],
                             n_flagged=sum(bool(r['flag_abs_gt']) for r in catalogue)),
        gpu=dict(jobs=jobs, total_runtime_s=used, by_category=by_cat, budget_s=16 * 3600, within_budget=used <= 16 * 3600))


def plot(results, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.hashsalt'] = 'xtbflow-o2'
    rows = results['paired']['rows']
    x = [r['delta_ea_xtb_kcal'] for r in rows]
    y = [r['delta_ea_dft_kcal'] for r in rows]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    lo, hi = min(x + y + [0.0]) - 2, max(x + y + [0.0]) + 2
    ax.plot([lo, hi], [lo, hi], color='0.6', lw=0.8, ls='--', label='y = x')
    ax.axhline(0, color='0.8', lw=0.8)
    ax.axvline(0, color='0.8', lw=0.8)
    ax.scatter(x, y, color='#2b6cb0', zorder=3)
    for r, xi, yi in zip(rows, x, y):
        ax.annotate(r['formula'], (xi, yi), textcoords='offset points', xytext=(4, 4), fontsize=8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('ΔEa xTB (kcal/mol)')
    ax.set_ylabel('ΔEa DFT (kcal/mol)')
    ax.set_title(f'O2 paired parents (n = {len(rows)})', fontsize=10)
    ax.legend(loc='upper left', fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, format='svg', metadata={'Date': None})


def manifest(run_root, parents_path, out_dir):
    source = Path.cwd()
    quarantine = run_root / 'o2-run-preqc-failed' / 'quarantine.json'
    files = [run_root / 'o2-plan.json', Path(parents_path), quarantine]
    files += sorted(p for p in (run_root / 'o2-run').glob('*/*.json'))
    files += sorted((run_root / 'o2-jobs').glob('ledger*.json')) + sorted((run_root / 'o2-jobs').glob('slurm-*-terminal.txt'))
    files = [f for f in files if f.exists()]
    plan = json.loads((run_root / 'o2-plan.json').read_text())
    jobs = gpu_jobs(run_root / 'o2-jobs', json.loads(quarantine.read_text()) if quarantine.exists() else [])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / 'manifest.json', dict(
        schema='xtbflow-open-world-o2-manifest/1',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip(),
        dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=source, text=True).strip()),
        frozen_config_sha256=plan['config_sha256'], analysis_config_sha256=file_hash(ANALYSIS_CONFIG),
        execution_sources=dict(planning=plan['source_commit'], runner_v1='bb09c34', runner_v2_dispatcher_v4='1823534'),
        analysis_slurm_job_id=None, analysis='CPU only on the n2 login shell (owner authorisation 2026-10-09)',
        slurm_jobs=[j['job_id'] for j in jobs], files=[dict(path=str(f), sha256=file_hash(f)) for f in files]))
    (out_dir / 'SHA256SUMS.remote').write_text(''.join(f'{file_hash(f)}  {f}\n' for f in files))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('analyze')
    a.add_argument('--run-root', type=Path, required=True)
    a.add_argument('--parents', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    g = sub.add_parser('plot')
    g.add_argument('--results', type=Path, required=True)
    g.add_argument('--out', type=Path, required=True)
    m = sub.add_parser('manifest')
    m.add_argument('--run-root', type=Path, required=True)
    m.add_argument('--parents', type=Path, required=True)
    m.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args()
    if args.cmd == 'analyze':
        write_json(args.out, analyze(args.run_root.resolve(), args.parents))
    elif args.cmd == 'plot':
        plot(json.loads(args.results.read_text()), args.out)
    else:
        manifest(args.run_root.resolve(), args.parents, args.out_dir)


if __name__ == '__main__':
    main()
