"""Deterministic sample selection, without quantum calculations."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def select(root):
    by = defaultdict(list)
    with (root/'screen/efficiency_s0.candidates.jsonl').open() as handle:
        for line in handle:
            r = json.loads(line)
            if r['arm']=='B0' and r['training_seed']==0 and r['decode_status']=='VALID':
                by[r['proxy_status']].append(r)
    selected = {}
    for status, n in [('OUTSIDE_CATALOGUE_EVENT',40), ('PROXY_MATCH',20), ('KNOWN_EVENT_GEOMETRY_MISS',20)]:
        rows, seen = [], set()
        for r in sorted(by[status], key=lambda r: hashlib.sha256(r['candidate_id'].encode()).hexdigest()):
            if r['parent_id'] in seen:
                continue
            seen.add(r['parent_id'])
            rows.append({'candidate_id':r['candidate_id'], 'parent_id':r['parent_id']})
            if len(rows)==n:
                break
        if len(rows)!=n:
            raise ValueError(f'insufficient eligible candidates in {status}')
        selected[status] = rows
    pids = {r['parent_id'] for r in selected['OUTSIDE_CATALOGUE_EVENT']}
    refs = defaultdict(list)
    with (root/'data/reference_catalog.jsonl').open() as handle:
        for line in handle:
            r = json.loads(line)
            if r['parent_id'] in pids:
                refs[r['parent_id']].append(r)
    selected['BASELINE'] = [{'parent_id':pid, 'reference_id':min(refs[pid],key=lambda r:(r['catalog_barrier_kcal'],r['reference_id']))['reference_id']}
                            for pid in sorted(pids)]
    return dict(exploratory=True, version=1, selection='sha256 candidate_id ascending; distinct parents within each group',
                groups=selected, engine='tblite GFN2-xTB; neutral singlet', hessian_step_bohr=0.005,
                ts_maxiter=200, irc_maxiter=300, irc_trust=0.2, minimum_maxiter=200,
                thresholds='qc_protocol.TIGHT; harmonic; classify_saddle; classify_minimum; endpoint_identity imported unchanged',
                intervals='Wilson 95% for certification proportions',
                positive_control_gate='PROXY_MATCH certification < 0.5: do not interpret OUTSIDE results',
                baseline_gate='delta TS energy only if both candidate and same-parent baseline are STRICT_JOINT_GRAPH_VALID',
                second_stage='proposal only, at most 10 candidates, no DFT execution',
                cpu_core_hours=50, gpu_hours=0)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/home/lhshen/xtbflow-runs/v1a-20261007'))
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(select(a.root),sort_keys=True,indent=2)+'\n')
