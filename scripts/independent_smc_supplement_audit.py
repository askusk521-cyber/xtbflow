"""Independent endpoint-level audit; no SMC metric/selection imports."""
import hashlib
import json
from pathlib import Path
from collections import defaultdict

ROOT=Path('/home/lhshen/xtbflow-runs/explore-smc-20261009')

def load(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def event_hash(event):return hashlib.sha256(event.encode()).hexdigest()


def main():
    rawpath=ROOT/'raw-ranking-726912a/results.json'
    decpath=ROOT/'decomposition-8c98eac/results.json'
    endpointpath=ROOT/'smc-2116d43/endpoint_rows.json'
    parentpath=Path('/home/lhshen/xtbflow-runs/v1a-20261007/data/parent_catalog.json')
    raw=load(rawpath);dec=load(decpath)
    assert digest(endpointpath)==raw['provenance']['inputs'][str(endpointpath)]
    assert digest(parentpath)==raw['provenance']['inputs'][str(parentpath)]
    parents={p['parent_id']:p for p in load(parentpath)}
    populations=defaultdict(list)
    for r in load(endpointpath):populations[r['arm'],r['parent_id'],r['seed']].append(r)
    assert len(populations)==2418
    values=defaultdict(list)
    for (arm,pid,seed),rr in sorted(populations.items()):
        assert len(rr)==32 and {r['proposal'] for r in rr}==set(range(32))
        best=set(parents[pid]['best_channel_ids']);scores={}
        for r in rr:
            c=r['channel_id']
            if c is None:continue
            e=r['score']['raw_barrier']
            key=(1,float('inf')) if e is None else (0,e)
            if c not in scores or key<scores[c]:scores[c]=key
        ranked=sorted(scores,key=lambda c:(*scores[c],event_hash(c)))
        hits=[float(bool(set(ranked[:k])&best)) for k in (1,2,4)]
        metrics={'hit@1':hits[0],'hit@2':hits[1],'hit@4':hits[2],
                 'H':sum(hits)/3,'hit_infinity':float(bool(set(scores)&best))}
        for m,v in metrics.items():values[arm,pid,m].append((seed,v))
    error=0.
    for (arm,pid,m),vv in values.items():
        assert sorted(s for s,v in vv)==[0,1,2]
        observed=sum(v for s,v in vv)/3
        error=max(error,abs(observed-raw['parent_metrics']['raw'][arm][m][pid]))
    assert error<=1e-12
    identity_error=0.
    availability_error=0.
    for ranker,arms in dec['parent_values'].items():
        for arm,metrics in arms.items():
            for pid,h in metrics['H'].items():
                available=metrics['availability'][pid];loss=metrics['ranking_loss'][pid]
                identity_error=max(identity_error,abs(h-available+loss))
                availability_error=max(availability_error,abs(available-raw['parent_metrics'][ranker][arm]['hit_infinity'][pid]))
                assert abs(h-raw['parent_metrics'][ranker][arm]['H'][pid])<=1e-12
    assert identity_error<=1e-12 and availability_error<=1e-12
    report=dict(independent_of_smc_metric_functions=True,populations=len(populations),
                candidates=sum(map(len,populations.values())),raw_parent_metrics_checked=len(values),
                raw_max_error=error,decomposition_identity_max_error=identity_error,
                availability_max_error=availability_error,
                inputs={str(p):digest(p) for p in (rawpath,decpath,endpointpath,parentpath)},
                scope='Independent raw event deduplication, ranking, best-event hits and seed aggregation; decomposition identities. Does not independently validate quantum chemistry or confidence-interval formula.')
    dest=ROOT/'supplement-independent-audit.json'
    if dest.exists():raise FileExistsError(dest)
    dest.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps(report,sort_keys=True,indent=2))


if __name__=='__main__':main()
