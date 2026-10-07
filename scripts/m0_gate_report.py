"""Generate the M0 paired-parent gate from real frozen test artifacts."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
import numpy as np
from xtbflow.m0.metrics import DELTAS, decide, paired_bootstrap, parent_means


def load(path):
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    result={r['index']:r for r in rows}
    if len(rows)!=len(result):raise ValueError('Duplicate query IDs')
    return result


def aggregate(runs, field):
    keys=set(runs[0])
    if any(set(r)!=keys for r in runs):raise ValueError('Seed query coverage differs')
    parents={q:runs[0][q]['parent_id'] for q in keys}
    if any(r[q]['parent_id']!=parents[q] for r in runs for q in keys):raise ValueError('Parent identity differs')
    return parent_means({q:float(np.mean([field(r[q]) for r in runs])) for q in keys},parents)


NEXT={'GO':'Enter M1.','NO-GO':'Retain negative result and move to direction B.',
      'INCONCLUSIVE':'Request the predefined extension; do not tune from test results.'}
DEVIATIONS=[
    'Data gate: the owner authorized continuing with unchanged filtering and splits despite 84.58% retention '
    '(required >= 85%) and 35 test parents (required >= 40). This limits statistical coverage.',
    'Numerical stop: the first formal attempt had non-finite loss in two of three seeds per arm. With owner approval '
    'the PaiNN update block of both arms was pre-normalized (layer_norm(s) without affine parameters, per-atom '
    'rescaled vectors, zero new parameters), and noise selection and all nine formal runs were redone. Failed runs '
    'are retained; see docs/evidence/m0/stabilization_decision.json.',
    'CPU pytest needs the repository root on PYTHONPATH.',
    'No xTB or DFT validation of generated transition states is claimed.',
]


def render_report(result, plot_name):
    """Markdown in the order of guide §8.2: verdict sentence, Table 1, Figure 1, Table 2, deviations, next step."""
    p,v,sec=result['primary'],result['validity'],result['secondary']
    ci=lambda d:f"[{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}]"
    fmt=lambda x:'n/a' if x is None else f'{x:.4f}'
    out=[f"# M0 gate report",'',
         f"**{result['decision']}**: M_A = {p['M_A']:.4f}, M_B = {p['M_B']:.4f}, Δ = {p['delta']:.4f}, "
         f"95% CI {ci(p)} over {p['n_parents']} test parents.",'',
         '## Table 1. Primary metric and validity','',
         '| Quantity | A (cascade) | B (joint) | Δ = B − A | 95% CI |','|---|---:|---:|---:|---|',
         f"| Primary M (δ = 0.5 Å) | {p['M_A']:.4f} | {p['M_B']:.4f} | {p['delta']:.4f} | {ci(p)} |",
         f"| Validity | {v['M_A']:.4f} | {v['M_B']:.4f} | {v['delta']:.4f} | {ci(v)} |",'',
         'Per-seed Δ (no bootstrap): '+', '.join(f'seed {k}: {d:.4f}' for k,d in enumerate(result['seed_deltas']))+'.','',
         'Sensitivity (reported only; the decision stays at δ = 0.5 Å): '
         +', '.join(f'δ = {d} Å → {x}' for d,x in result['sensitivity'].items())+'.','',
         '## Figure 1. Primary metric versus δ','',f'![Primary metric versus δ]({plot_name})','',
         '## Table 2. Secondary metrics (seed-averaged) and capacity','',
         '| Metric | A (cascade) | B (joint) |','|---|---:|---:|']
    for key in ('event_recall','median_rmsd_on_hits','consistency','unique_valid_events'):
        out.append(f"| {key} | {fmt(sec['A'][key])} | {fmt(sec['B'][key])} |")
    out+=[f"| parameters | {result['params']['A']} | {result['params']['B']} |",'',
          f"Automorphism cap reached on {result['automorphism_cap_hits']} test reactions. A2x was not run.",'',
          '## Deviations from the guide','']
    out+=[f'{k}. {d}' for k,d in enumerate(DEVIATIONS,1)]
    out+=['','## Next','',NEXT[result['decision']],'']
    return '\n'.join(out)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--a',nargs='+',type=Path,required=True)
    ap.add_argument('--b',nargs='+',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--frozen-selection',type=Path,default=Path('docs/evidence/m0/frozen_selection.json'))
    ap.add_argument('--candidates',nargs='+',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=Path('docs/evidence/m0/gate_result.json'))
    ap.add_argument('--report',type=Path,default=Path('docs/m0/M0_GATE_REPORT.md'))
    a=ap.parse_args()
    if len(a.a)!=3 or len(a.b)!=3:raise ValueError('Initial gate requires three seeds per arm')
    if len(a.candidates)!=6:raise ValueError('Six test candidate artifacts required')
    if a.out.exists() or a.report.exists():raise FileExistsError('Refusing to overwrite report')
    frozen=json.loads(a.frozen_selection.read_text())
    ts=subprocess.check_output(['git','log','-1','--format=%ct','--',str(a.frozen_selection)],text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--',str(a.frozen_selection)],text=True).strip()
    if not ts or dirty:raise ValueError('Selection must be committed and unchanged')
    if any(p.stat().st_mtime<=int(ts) for p in a.candidates):raise ValueError('Candidates predate frozen selection commit')
    def sha256(path):
        digest=hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda:handle.read(1 << 20),b''):
                digest.update(chunk)
        return digest.hexdigest()
    artifact_provenance=dict(
        report_git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        config_sha256=sha256(a.config),
        frozen_selection_sha256=sha256(a.frozen_selection),
        candidate_sha256=[sha256(p) for p in a.candidates],
        evaluation_sha256=[sha256(p) for p in a.a+a.b],
    )
    cfg=json.loads(a.config.read_text());ec=cfg['eval'];aa=list(map(load,a.a));bb=list(map(load,a.b))
    if 'data' in cfg:  # guide §7.4 provenance includes the cache hash; synthetic fixtures have no cache
        artifact_provenance['cache_sha256']=sha256(Path(os.path.expandvars(cfg['data']['cache'])))
    if any(set(r)!=set(aa[0]) for r in aa+bb):raise ValueError('Arm query coverage differs')
    boot=lambda f:paired_bootstrap(aggregate(aa,f),aggregate(bb,f),ec['bootstrap'],ec['bootstrap_seed'])
    primary=boot(lambda r:r['hit']['0.5']);valid=boot(lambda r:r['valid_frac'])
    def deltas_at(d):
        return [paired_bootstrap(aggregate([x],lambda r:r['hit'][str(d)]),aggregate([y],lambda r:r['hit'][str(d)]),1,0)['delta'] for x,y in zip(aa,bb)]
    sd=deltas_at(.5);secondary={}
    for arm,runs in [('A',aa),('B',bb)]:
        vals={}
        for name,key in [('event_recall','hit_event'),('consistency','consistency'),('unique_valid_events','unique_valid_events')]:
            per=[]
            for run in runs:
                available=[r[key] for r in run.values() if r[key] is not None]
                per.append(float(np.mean(available)) if available else None)
            vals[name]=float(np.mean([v for v in per if v is not None])) if any(v is not None for v in per) else None
        meds=[]
        for run in runs:
            v=[r['best_rmsd'] for r in run.values() if r['hit']['0.5']]
            if v:meds.append(float(np.median(v)))
        vals['median_rmsd_on_hits']=float(np.mean(meds)) if meds else None
        secondary[arm]=vals
    curve={str(d):boot(lambda r,d=d:r['hit'][str(d)]) for d in DELTAS}
    result=dict(decision=decide(primary,sd,valid),primary=primary,seed_deltas=sd,validity=valid,
                secondary=secondary,delta_curve=curve,sensitivity={str(d):decide(curve[str(d)],deltas_at(d),valid) for d in (.3,1.)},
                a2x=None,params=frozen['params'],automorphism_cap_hits=sum(any(r[q]['automorphism_cap_hit'] for r in aa+bb) for q in aa[0]),
                provenance=dict(selection=frozen,**artifact_provenance), data_gate_exceptions={'retention':0.8458254740395116,'test_parents':35})
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n', encoding='utf-8')
    # Dependency-free curve plot: values come exclusively from the computed bootstrap.
    a.report.parent.mkdir(parents=True,exist_ok=True)
    plot = a.report.with_suffix('.delta.svg')
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="600" height="340" viewBox="0 0 600 340">',
             '<rect width="600" height="340" fill="white"/>',
             '<path d="M50 20 V290 H570" stroke="black" fill="none"/>',
             '<text x="220" y="330">TS RMSD threshold (Å)</text>',
             '<text x="50" y="15">Parent macro hit rate (0–1)</text>']
    for arm,color in [('A','#2563eb'),('B','#dc2626')]:
        points=' '.join(f'{50+(d-.1)/.9*520:.2f},{290-curve[str(d)]["M_"+arm]*260:.2f}' for d in DELTAS)
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<text x="{430 if arm=="A" else 500}" y="35" fill="{color}">Arm {arm}</text>')
    lines.append('</svg>')
    plot.write_text('\n'.join(lines), encoding='utf-8')
    a.report.write_text(render_report(result, plot.name), encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
