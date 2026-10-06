"""Generate the M0 paired-parent gate from real frozen test artifacts."""
import argparse
import hashlib
import json
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
    cfg=json.loads(a.config.read_text());ec=cfg['eval'];aa=list(map(load,a.a));bb=list(map(load,a.b))
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
                provenance=frozen, data_gate_exceptions={'retention':0.8458254740395116,'test_parents':35})
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    p=primary
    text=f"# M0 gate report\n\n{result['decision']}: M_A={p['M_A']:.6f}, M_B={p['M_B']:.6f}, Δ={p['delta']:.6f}, 95% CI={p['ci95']}.\n\n"
    text+='## Main and validity\n\n```json\n'+json.dumps({'primary':primary,'validity':valid,'seed_deltas':sd},indent=2)+'\n```\n'
    text+='## Secondary metrics and capacity\n\n```json\n'+json.dumps({'secondary':secondary,'params':result['params'],'sensitivity':result['sensitivity']},indent=2)+'\n```\n'
    text+='## Guide exceptions\n\nOwner authorized continuation with unchanged filtering/splits despite 84.58% retention and 35 test parents. This limits statistical coverage. CPU pytest required repository root on PYTHONPATH. No xTB TS validation is claimed.\n'
    text+='\n## Next\n\n'+{'GO':'Enter M1.','NO-GO':'Retain negative result and move to direction B.','INCONCLUSIVE':'Request the predefined extension; do not tune from test results.'}[result['decision']]+'\n'
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(text)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
