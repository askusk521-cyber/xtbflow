"""Planning numbers for the V1b v2.2 amendment (no quantum calculation).

Reads only frozen sources: the sealed V1a budget-16 source population, the
task R candidate order (for the S3 truncated sets) and the V1a parent
catalogue; D1 chain wall times are read as a dated snapshot. Writes one JSON
with the v2.2 parent strata, |C*| distributions, Delta^proxy of every
estimand, resource scenarios and the guide Section 13.3 precision model.
"""
import argparse
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path
import time

import numpy as np

from xtbflow.v1.data import file_hash

CERT_ARMS=('A1','A2','B0','B1')
# Estimand -> (X set, Y set); sets are full budget-16 bundles or S3 truncations.
ESTIMANDS={'primary':('B1','A2'),'S1':('B1','B0'),'S2':('B1','A1'),
           'S3a':('B1','B0|S3a'),'S3b':('B1|S3b','A1|S3b')}
# Which estimands each plan level keeps (handoff cut order: S1 first, then S2).
LEVELS={'v2.1_primary_only':('primary',),'v2.2_full':('primary','S1','S2','S3a','S3b'),
        'v2.2_drop_S1':('primary','S2','S3a','S3b'),'v2.2_drop_S1_S2':('primary','S3a','S3b')}
STRATA=('B1_only','A2_only','both','neither_with_best_event','primary_empty_secondary_nonempty','all_empty')


def jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def stratum(p):
    """v2.1 strata on the B1/A2 proxy pattern, with neither_empty split for v2.2."""
    yb,ya=p['proxy']['B1'],p['proxy']['A2']
    if yb and not ya:return 'B1_only'
    if ya and not yb:return 'A2_only'
    if ya and yb:return 'both'
    if p['C']['A2'] or p['C']['B1']:return 'neither_with_best_event'
    if any(p['C'][k] for k in p['C']):return 'primary_empty_secondary_nonempty'
    return 'all_empty'


def chains(p,level):
    """Candidate chains a sampled parent needs: every C* candidate of the sets in use."""
    used=set()
    for e in LEVELS[level]:used.update(ESTIMANDS[e])
    # Truncated sets are subsets of their arm's bundle; count each arm once at its widest use.
    total=0
    for arm in CERT_ARMS:
        sets=[s for s in used if s.split('|')[0]==arm]
        if sets:total+=max(p['C'][s] for s in sets)
    return total


def half_width(parents,alloc,est,r,z=1.6448536269514722):
    """Guide 13.3: one-sided 95% half-width of an estimand's HT mean under correction rate r.

    Per arm with a non-empty C* the correction is -1 (proxy hit not certified) or +1
    (proxy miss certified) with probability r, independently across arms.
    """
    x,y=ESTIMANDS[est];M=len(parents);var=0.
    by=defaultdict(list)
    for p in parents:by[p['stratum']].append(p)
    for h,rows in by.items():
        m=alloc[h];N=len(rows)
        if m>=N:continue
        def sign(p,s):return 0 if not p['C'][s] else (-1 if p['proxy'][s] else 1)
        mean_d=np.array([r*(sign(p,x)-sign(p,y)) for p in rows])
        var_d=np.array([r*(1-r)*(bool(p['C'][x])+bool(p['C'][y])) for p in rows])
        s2=var_d.mean()+(mean_d.var(ddof=1) if N>1 else 0.)
        var+=N*N*(1-m/N)*s2/m
    return z*np.sqrt(var)/M


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--v1a-root',type=Path,required=True)
    ap.add_argument('--sealed',type=Path,required=True,help='docs/evidence/v1a/formal/SHA256SUMS')
    ap.add_argument('--r-streams',type=Path,required=True)
    ap.add_argument('--r-manifest',type=Path,required=True,help='docs/evidence/v1a_r/extract_manifest.json')
    ap.add_argument('--d1',type=Path,required=True,help='V1b D1 run directory (read-only snapshot)')
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    sealed={n:d for d,n in (l.split() for l in a.sealed.read_text().splitlines() if l.strip())}
    src=a.v1a_root/'final/v1b_source_population.jsonl'
    if file_hash(src)!=sealed['final/v1b_source_population.jsonl']:raise ValueError('source population changed')
    rman=json.loads(a.r_manifest.read_text())
    if file_hash(a.r_streams)!=rman['streams_gz_sha256']:raise ValueError('task R extract changed')
    order={}
    with gzip.open(a.r_streams,'rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['training_seed']==0:order[r['parent_id'],r['arm']]=r
    bundles=defaultdict(dict)
    for b in jsonl(src):bundles[b['parent_id']][b['arm']]=b
    parents=[]
    for pid in sorted(bundles):
        arms=bundles[pid];p=dict(parent_id=pid,C={},proxy={},n={})
        for arm in CERT_ARMS:
            b=arms[arm];st=order[pid,arm];n=len(b['candidate_ids'])
            # The bundle is the 800-unit prefix of the stream; check it against the R order.
            if (sum(c<=800+1e-8 for c in st['completion_units'])!=n or sum(st['event'][:n])!=len(b['best_event_candidate_ids'])
                    or any(st['joint'][:n])!=bool(b['proxy_best_reference_hit'])):
                raise ValueError(f'bundle and stream order disagree: {pid} {arm}')
            p['C'][arm]=len(b['best_event_candidate_ids']);p['proxy'][arm]=bool(b['proxy_best_reference_hit']);p['n'][arm]=n
        n3a=p['n']['B1'];n3b=min(p['n']['B1'],p['n']['A1'])
        for name,arm,k in (('B0|S3a','B0',n3a),('A1|S3b','A1',n3b),('B1|S3b','B1',n3b)):
            st=order[pid,arm];p['C'][name]=int(sum(st['event'][:k]));p['proxy'][name]=bool(any(st['joint'][:k]))
        meta=rman['parents'][pid];p['n_atoms']=meta['n_atoms']
        p['stratum']=stratum(p);parents.append(p)
    catalog={c['parent_id']:c for c in json.loads((a.v1a_root/'data/parent_catalog.json').read_text())}
    for p in parents:p['n_best_refs']=len(catalog[p['parent_id']]['best_reference_ids'])
    M=len(parents);sizes=Counter(p['stratum'] for p in parents)
    delta_proxy={e:float(np.mean([p['proxy'][x]-p['proxy'][y] for p in parents])) for e,(x,y) in ESTIMANDS.items()}
    c_star={s:dict(sorted(Counter(p['C'][s] for p in parents).items())) for s in
            ('A1','A2','B0','B1','B0|S3a','A1|S3b','B1|S3b')}
    by_stratum={h:dict(M_h=sizes.get(h,0),**{f'mean_C_{s}':float(np.mean([p['C'][s] for p in parents if p['stratum']==h] or [0]))
                for s in ('A1','A2','B0','B1','B0|S3a','A1|S3b')}) for h in STRATA}

    verdicts=[]
    for f in sorted(a.d1.glob('*/verdict.json')):
        v=json.loads(f.read_text());kind=f.parent.name.split('_')[0]
        verdicts.append(dict(item=f.parent.name,kind=kind,wall_s=v.get('wall_s'),n_atoms=v.get('n_atoms'),
                             status=v.get('strict_status') or v.get('status')))
    cand=[v['wall_s'] for v in verdicts if v['kind']=='candidate' and v['wall_s']]
    ref=[v['wall_s'] for v in verdicts if v['kind']=='reference' and v['wall_s']]
    anchor=[v['wall_s'] for v in verdicts if v['kind']=='anchor' and v['wall_s']]
    unit=dict(candidate_median_s=float(np.median(cand)),candidate_max_s=float(max(cand)),n_candidate_chains=len(cand),
              reference_median_s=float(np.median(ref)),reference_max_s=float(max(ref)),n_reference_chains=len(ref),
              anchor_median_s=float(np.median(anchor)),anchor_max_s=float(max(anchor)),n_anchor_chains=len(anchor))

    scenarios={'census_discordant_sample_20':{h:(sizes[h] if h in ('B1_only','A2_only','all_empty') else min(20,sizes[h]))
                                              for h in STRATA if sizes.get(h)},
               'census_discordant_sample_10':{h:(sizes[h] if h in ('B1_only','A2_only','all_empty') else min(10,sizes[h]))
                                              for h in STRATA if sizes.get(h)},
               'census_all':{h:sizes[h] for h in STRATA if sizes.get(h)}}
    resources={}
    for name,alloc in scenarios.items():
        resources[name]={}
        for level in LEVELS:
            exp=Counter();up=Counter()
            for h,m in alloc.items():
                rows=[p for p in parents if p['stratum']==h];N=len(rows)
                k=[chains(p,level) for p in rows]
                # A parent with no C* candidate in any used set needs no QC, anchor or reference;
                # otherwise one reactant anchor plus every native best reference chain.
                jobs=dict(candidate=k,anchor=[int(bool(x)) for x in k],
                          reference=[p['n_best_refs'] if x else 0 for p,x in zip(rows,k)])
                for kind,v in jobs.items():
                    exp[kind]+=m/N*sum(v);up[kind]+=sum(sorted(v,reverse=True)[:m])
            resources[name][level]=dict(
                expected_jobs=dict(exp),upper_jobs=dict(up),
                median_gpu_h=sum(exp[k]*unit[f'{k}_median_s'] for k in exp)/3600,
                conservative_upper_gpu_h=sum(up[k]*unit[f'{k}_max_s'] for k in up)/3600,
                precision_one_sided_halfwidth={e:{str(r):float(half_width(parents,alloc,e,r)) for r in (.05,.10,.20)}
                                               for e in LEVELS[level]})
    out=dict(schema='xtbflow-v1b-v22-plan/1',generated_unix=time.time(),M=M,strata=dict(sizes),
             strata_v21=dict(Counter('neither_empty' if p['stratum'] in ('primary_empty_secondary_nonempty','all_empty')
                                     else p['stratum'] for p in parents)),
             delta_proxy=delta_proxy,c_star_size_distribution=c_star,by_stratum=by_stratum,
             d1_snapshot=dict(unit_costs=unit,verdicts=verdicts,
                              note='Snapshot of a D1 run still in progress; wall seconds on one GPU (gpu4pyscf).'),
             scenarios=scenarios,resources=resources,
             inputs=dict(source_population_sha256=file_hash(src),r_streams_sha256=rman['streams_gz_sha256'],
                         parent_catalog_sha256=file_hash(a.v1a_root/'data/parent_catalog.json')),
             model=('Resources: no witness stopping or energy short-cut, so every C* candidate of a sampled parent '
                    'runs a full chain; upper = largest-|C*| parents of each stratum at the largest D1 chain time. '
                    'Precision: guide 13.3 independent per-arm correction rate r.'))
    a.out.write_text(json.dumps(out,indent=1,sort_keys=True)+'\n')
    print(json.dumps(dict(strata=out['strata'],delta_proxy=delta_proxy,c_star=c_star,unit=unit),indent=1))


if __name__=='__main__':main()
