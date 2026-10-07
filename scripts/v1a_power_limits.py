"""Report identifiable effects and hypothetical larger designs after HOLD_POWER."""
import argparse
import json
from pathlib import Path

import numpy as np

from xtbflow.v1.data import file_hash,write_json
from xtbflow.v1.power import nuisance,simulate_gate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    report=json.loads(a.plan.read_text());rows=report['development_values']
    x=np.array([[r['AUC'],r['M0'],r['MR']] for r in rows]);groups=np.array([r['split_group'] for r in rows])
    sizes=report['candidate_inventory']['sizes'];guard=report['planning_guard']
    results=[]
    for effect in (.05,.055,.06,.065,.075):
        for distribution in ('normal','empirical_skew'):
            r=simulate_gate(x,groups,sizes,[effect,.02,.02],repeats=20000,
                distribution=distribution,**guard)
            results.append(r)
    hypothetical=[]
    # Whole new independent groups with approximately the existing size mixture;
    # these parents do NOT exist in the reserved inventory and are NOT selected.
    for target in (400,450,500):
        rng=np.random.default_rng(20261007);ss=[]
        while sum(ss)<target:ss.append(int(rng.choice(sizes)))
        hypothetical.append(simulate_gate(x,groups,ss,[.05,.02,.02],repeats=20000,**guard))
    # Whole-formula bootstrap exposes uncertainty in the development nuisance.
    rng=np.random.default_rng(20261007);keys=np.unique(groups);sd=[];icc=[]
    for _ in range(2000):
        selected=rng.choice(keys,len(keys),replace=True)
        xx=np.concatenate([x[groups==g] for g in selected]);gg=np.concatenate([
            np.full((groups==g).sum(),j) for j,g in enumerate(selected)])
        try:nu=nuisance(xx,gg)
        except ValueError:continue
        sd.append(nu['sd']);icc.append(nu['icc_raw'])
    candidates=sorted({r['effects'][0] for r in results})
    qualifying=[e for e in candidates if all(r['GO']-1.645*r['mcse_GO']>=.8 for r in results if r['effects'][0]==e)]
    out=dict(source_plan_sha256=file_hash(a.plan),effect_grid=results,hypothetical_designs=hypothetical,
             smallest_tested_auc_effect_with_80pct_guarded_power=min(qualifying) if qualifying else None,
             bootstrap_nuisance=dict(repeats=len(sd),quantiles=[.05,.5,.95],
                 sd=np.quantile(sd,[.05,.5,.95],axis=0).tolist(),icc_raw=np.quantile(icc,[.05,.5,.95],axis=0).tolist()),
             guard_note='SD x1.10 and ICC +0.05 were specified before the first power scan and before any screen results. They are analyst planning choices, not literal numeric requirements in the supplied guide.',
             scope='Planning diagnostics only. Larger designs require new independent data or a newly designed retraining/OOF study; none are authorized or executed by this report.')
    write_json(a.out,out)
    print(json.dumps(dict(smallest_tested_effect=out['smallest_tested_auc_effect_with_80pct_guarded_power'],
        effect_power=[dict(effect=r['effects'][0],distribution=r['distribution'],GO=r['GO']) for r in results],
        hypothetical=[dict(N=r['n_parents'],G=r['n_formula_groups'],GO=r['GO']) for r in hypothetical]),indent=2))


if __name__=='__main__':main()
