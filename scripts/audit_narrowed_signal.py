#!/usr/bin/env python3
"""S4: reaggregate cached rows only; no model/calculator imports or fitting."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import resource
import socket
import subprocess
import time

import numpy as np
from scipy.stats import spearmanr, t


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clustered(values: dict[str, float], groups: dict[str, str]) -> dict:
    keys = sorted(values)
    data = np.array([values[k] for k in keys])
    labels = np.array([groups[k] for k in keys])
    unique = np.unique(labels)
    if len(unique) < 2 or not np.isfinite(data).all():
        raise ValueError('inadequate complete parent/family support')
    mean = float(data.mean())
    scores = np.array([(data[labels == group]-mean).sum() for group in unique])
    se = float(np.sqrt(len(unique)/(len(unique)-1)*np.square(scores).sum())/len(data))
    half = float(t.ppf(.975, len(unique)-1)*se)
    return dict(estimate=mean, ci_two95=[mean-half,mean+half], n_parents=len(data), n_formula_groups=len(unique))


def audit(a_rows: list[dict], b_rows: list[dict]) -> dict:
    groups = {r['parent_id']: r['formula'] for r in a_rows}
    by_parent = defaultdict(list)
    for row in a_rows:
        by_parent[row['parent_id']].append(row)
    predictions = {key:{} for key in ('E_frozen','X_frozen','N_clean','G_clean')}
    for parent, rows in by_parent.items():
        if len({r['channel_id'] for r in rows}) != len(rows):
            raise ValueError('this audit requires one retained reference per channel')
        for key in predictions:
            predictions[key][parent] = float(spearmanr([r[key] for r in rows], [r['catalog_barrier_kcal'] for r in rows]).statistic)
    comparisons = {f'{x}-{y}':clustered({p:predictions[x][p]-predictions[y][p] for p in groups}, groups)
                   for x,y in [('G_clean','N_clean'),('X_frozen','E_frozen')]}
    valid = [r for r in b_rows if r['commit_step'] is not None]
    commit = np.array([r['commit_step']/50 for r in valid])
    grid = np.linspace(0,1,11)
    curves = []
    b_groups = {r['parent_id']:r['formula'] for r in b_rows}
    target = np.array([np.nan if r['event_barrier_kcal'] is None else r['event_barrier_kcal'] for r in b_rows])
    parent_ids = np.array([r['parent_id'] for r in b_rows])
    signal = np.asarray([r['xtb_delta_kcal'] for r in b_rows])
    for index in range(11):
        values, eligible = {}, 0
        for parent in sorted(b_groups):
            use = (parent_ids==parent)&np.isfinite(target)&np.isfinite(signal[:,index])
            eligible += int(use.sum())
            if use.sum()>=4 and len(np.unique(target[use]))>=2 and len(np.unique(signal[use,index]))>=2:
                values[parent] = float(spearmanr(signal[use,index],target[use]).statistic)
        summary = clustered(values,b_groups)
        curves.append(dict(time=float(grid[index]), eligible_rows=eligible, **summary))
    final = curves[-1]
    informative = next((r['time'] for r in curves if final['ci_two95'][0]>0 and r['estimate']>=.5*final['estimate']),None)
    return dict(head_records=len(a_rows),head_parents=len(groups),head_comparisons=comparisons,
                head_predictors={k:clustered(v,groups) for k,v in predictions.items()},
                total_trajectories=len(b_rows),valid_final=len(valid),invalid_final=len(b_rows)-len(valid),
                final_status=dict(Counter(r['proxy_status'] for r in b_rows)),
                catalogue_labelled_rows=int(np.isfinite(target).sum()),out_of_catalogue_or_invalid=int(np.isnan(target).sum()),
                median_commit_time=float(np.median(commit)),xtb_curve=curves,t_geo=informative,
                offline_uncommitted_fraction=None if informative is None else float((commit>informative+1e-9).mean()),
                offline_uncommitted_by_time=[float((commit>x+1e-9).mean()) for x in grid],
                physical_status_by_time=[dict(Counter(r['xtb_status'][i] for r in b_rows)) for i in range(11)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--heads',type=Path,required=True)
    parser.add_argument('--trajectories',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    config_path = root/'configs/research/incremental_signal_audit_20261111.json'
    cfg = json.loads(config_path.read_text())
    resource.setrlimit(resource.RLIMIT_CPU,(600,600))
    resource.setrlimit(resource.RLIMIT_FSIZE,(16*1024**2,)*2)
    start,cpu = time.monotonic(),time.process_time()
    report = dict(schema='xtbflow-narrowed-incremental-signal/v1',host=socket.gethostname(),python=platform.python_version(),
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root).decode().strip(),
                  driver_sha256=digest(Path(__file__)),config_sha256=digest(config_path),caps=cfg['caps'],
                  new_model_forwards=0,training_steps=0,new_calculator_calls=0,gpu_seconds=0,status='executing')
    try:
        if args.heads.stat().st_size+args.trajectories.stat().st_size>cfg['caps']['max_input_bytes']:
            raise ValueError('input byte cap exceeded')
        for key,path in [('heads',args.heads),('trajectories',args.trajectories)]:
            if digest(path)!=cfg['input_sha256'][key]:
                raise ValueError('input hash mismatch: '+key)
        report['input_sha256']=cfg['input_sha256']
        result=audit([json.loads(s) for s in args.heads.read_text().splitlines()],
                     [json.loads(s) for s in args.trajectories.read_text().splitlines()])
        report['reaggregation']=result
        report.update(status='completed',decision='inconclusive_for_declared_strict_channel_endpoint',
                      s5_launch=False,independent_confirmation_available=False,
                      reasons=cfg['gate_reasons'],conditional_shuffle='not_validated; no exchangeability/support proof, not executed',
                      strict_channel_effect=dict(estimate=None,confidence_interval=None,logical_probability_difference_bounds=[-1,1],
                                                 practical_threshold=.10,practical_effect_excluded=False))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes=plt.subplots(1,3,figsize=(12,3.4),constrained_layout=True)
        labels=list(result['head_comparisons']);means=[result['head_comparisons'][k]['estimate'] for k in labels]
        intervals=np.array([result['head_comparisons'][k]['ci_two95'] for k in labels])
        axes[0].errorbar(means,[0,1],xerr=np.array([means-intervals[:,0],intervals[:,1]-means]),fmt='o',capsize=4)
        axes[0].axvline(0,color='grey',lw=1);axes[0].set_yticks([0,1],labels);axes[0].set_xlabel('Delta Spearman (95% clustered CI)')
        axes[0].set_title('Exposed catalogue ranking')
        x=[r['time'] for r in result['xtb_curve']]
        axes[1].plot(x,[r['estimate'] for r in result['xtb_curve']],label='xTB catalogue correlation')
        axes[1].plot(x,result['offline_uncommitted_by_time'],label='Offline uncommitted share')
        axes[1].axvline(result['t_geo'],color='grey',linestyle=':');axes[1].set_xlabel('Generation time');axes[1].legend(fontsize=7)
        axes[1].set_title('Timing is retrospective')
        statuses=result['final_status'];axes[2].bar(range(len(statuses)),list(statuses.values()),color='#617f99')
        axes[2].set_xticks(range(len(statuses)),list(statuses),rotation=35,ha='right',fontsize=7)
        axes[2].set_title('All 5,952 trajectories');axes[2].set_ylabel('Retained attempts')
        fig.savefig(args.out/'incremental_signal.png',dpi=160)
        plt.close(fig)
        report['figure_sha256']=digest(args.out/'incremental_signal.png')
    except Exception as exc:
        report.update(status='audit_failed',failure_type=type(exc).__name__,failure=str(exc),s5_launch=False)
    report.update(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,
                  claim_limits=['Existing exposed frozen-head/trajectory outputs only; no new causal sample.',
                                'True TS geometry and eventual event labels are retrospective, not legal online inputs.',
                                'Information, numerical coupling and connected-channel efficacy are different endpoints.'])
    with (args.out/'incremental_signal.json').open('x') as f:
        json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(report['status'])
    return int(report['status']!='completed')


if __name__=='__main__':
    raise SystemExit(main())
