"""Independent formula-cluster interval audit; never imports project metrics."""
import hashlib
import json
import math
from pathlib import Path
from scipy.stats import t

ROOT=Path('/home/lhshen/xtbflow-runs/explore-smc-20261009')

def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    rp=ROOT/'raw-ranking-726912a/results.json';dp=ROOT/'decomposition-8c98eac/results.json'
    pp=Path('/home/lhshen/xtbflow-runs/v1a-20261007/data/parent_catalog.json')
    raw=load(rp);dec=load(dp);parents={p['parent_id']:p for p in load(pp)}
    assert sha(pp)==raw['provenance']['inputs'][str(pp)]
    checks=0;max_error=0.;degenerate=0
    def verify(values,reported):
        nonlocal checks,max_error,degenerate
        keys=sorted(values);n=len(keys);groups=sorted({parents[p]['split_group'] for p in keys});g=len(groups)
        assert n==62 and g==21
        mean=math.fsum(values[p] for p in keys)/n
        totals=[math.fsum(values[p]-mean for p in keys if parents[p]['split_group']==group) for group in groups]
        se=math.sqrt(g/(g-1)*math.fsum(v*v for v in totals))/n
        assert reported['n_parents']==n and reported['n_formula_groups']==g
        targets=[(mean,reported['estimate']),(se,reported['se_cluster'])]
        if se<=1e-12:
            degenerate+=1
            assert reported['status']=='DEGENERATE_VARIANCE' and reported['ci_two95']==[-1.,1.]
        else:
            q=float(t.ppf(.975,g-1));qone=float(t.ppf(.95,g-1))
            targets.extend([(mean-q*se,reported['ci_two95'][0]),(mean+q*se,reported['ci_two95'][1]),
                            (mean-qone*se,reported['lower_one95']),(mean+qone*se,reported['upper_one95'])])
        for expected,actual in targets:
            error=abs(expected-actual);max_error=max(max_error,error);assert error<1e-12
        checks+=1
    def diff(x,y):
        assert set(x)==set(y)
        return {p:x[p]-y[p] for p in x}
    pm=raw['parent_metrics']
    for ranker,arms in pm.items():
        for arm,metrics in arms.items():
            for m,v in metrics.items():verify(v,raw['arms'][ranker][arm][m])
        for name,metrics in raw['comparisons'][ranker].items():
            a,b=name.split('-')
            for m,s in metrics.items():verify(diff(arms[a][m],arms[b][m]),s)
    shifts={arm:diff(pm['raw'][arm]['H'],pm['original_relaxed'][arm]['H']) for arm in pm['raw']}
    for arm,v in shifts.items():verify(v,raw['within_arm_ranker_shift'][arm])
    for name,s in raw['difference_in_differences'].items():
        a,b=name.split('-');verify(diff(shifts[a],shifts[b]),s)
    for ranker,arms in dec['parent_values'].items():
        for arm,metrics in arms.items():
            for m,v in metrics.items():verify(v,dec['absolute'][ranker][arm][m])
        for name,metrics in dec['comparisons'][ranker].items():
            a,b=name.split('-')
            for m,s in metrics.items():verify(diff(arms[a][m],arms[b][m]),s)
    result=dict(intervals_checked=checks,max_absolute_error=max_error,degenerate_intervals=degenerate,
                inputs={str(p):sha(p) for p in (rp,dp,pp)},
                method='Independent Python math.fsum cluster-score sandwich SE and scipy t quantiles, no project metric imports',
                scope='Verifies reported inference against frozen formula, not whether development reuse or exploratory multiplicity permits confirmatory claims.')
    out=ROOT/'supplement-interval-audit.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
