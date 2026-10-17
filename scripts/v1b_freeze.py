"""V1b D0: verify the V1a GO source and freeze populations, then (after D1) samples.

--check-source-only    build/verify populations, Delta^proxy and native best references
--freeze-after-calibration --allocation alloc.json
                       draw Stage A/B samples once, before any target H calculation
Outputs go to a new run directory; nothing is overwritten.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from xtbflow.v1.audit_sampling import B_STRATA,build_populations,frame_hash,stratified_srswor
from xtbflow.v1.data import file_hash,write_json

SEED_A='v1b_candidate_A_v2_1'
SEED_B='v1b_parent_B_v2_1'


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write_jsonl(path,rows):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        for r in rows:f.write(json.dumps(r,sort_keys=True)+'\n')


def check_source(a):
    gate=json.loads(a.v1a_gate.read_text())
    if gate['decision']!='GO_V1b':raise ValueError('V1b requires a frozen V1a GO_V1b')
    bundles=jsonl(a.bundles)
    candidates={}
    for line in a.candidates.read_text().splitlines():
        r=json.loads(line)
        if r['training_seed']==0 and r['sampling_seed']==0:candidates[r['candidate_id']]=r
    frame,parents,delta_proxy,summary=build_populations(bundles,candidates)
    if len(parents)!=gate['n_parents']:raise ValueError('parent population differs from the V1a gate')
    # Delta^proxy must reproduce the V1a budget-16 binary contrast at training seed 0.
    v1a=gate['secondary']['per_seed_primary']
    best={p['parent_id']:p for p in json.loads((a.catalogue/'parent_catalog.json').read_text())}
    native=[]
    refs=defaultdict(list)
    for line in (a.catalogue/'reference_catalog.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['parent_id'] in {p['unit_id'] for p in parents}:refs[r['parent_id']].append(r)
    for p in parents:
        meta=best[p['unit_id']]
        for r in refs[p['unit_id']]:
            if r['reference_id'] in meta['best_reference_ids']:
                native.append(dict(parent_id=p['unit_id'],query_id=p['query_id'],reference_id=r['reference_id'],
                                   channel_id=r['channel_id'],is_original_best_reference=True,
                                   native_catalogue_barrier_kcal=r['catalog_barrier_kcal'],
                                   native_ts_energy_ev=r['energy_ts_ev'],x_ts=r['x_ts'],
                                   threshold_source='refined_original_catalog_best'))
    if {r['parent_id'] for r in native}!={p['unit_id'] for p in parents}:
        raise ValueError('a parent lacks its native best reference')
    return gate,bundles,frame,parents,delta_proxy,summary,native,v1a


def main():
    ap=argparse.ArgumentParser()
    for key in ('v1a-gate','bundles','candidates','catalogue','out'):ap.add_argument('--'+key,type=Path,required=True)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-source-only',action='store_true')
    mode.add_argument('--freeze-after-calibration',action='store_true')
    ap.add_argument('--allocation',type=Path,help='{"A":{stratum:n},"B":{stratum:m}} frozen after D1')
    a=ap.parse_args()
    gate,bundles,frame,parents,delta_proxy,summary,native,_=check_source(a)
    a.out.mkdir(parents=True,exist_ok=True)
    manifest_path=a.out/'manifest.json'
    if a.check_source_only:
        if manifest_path.exists():raise FileExistsError('D0 already frozen')
        write_jsonl(a.out/'source_bundles.jsonl',bundles)
        write_jsonl(a.out/'source_candidates.jsonl',frame)
        write_jsonl(a.out/'source_parents.jsonl',parents)
        write_jsonl(a.out/'native_references.jsonl',native)
        manifest=dict(schema='xtbflow-v1b-manifest/1',stage='D0_source_frozen',required_v1a_decision='GO_V1b',
                      v1a_gate_sha256=file_hash(a.v1a_gate),v1a_freeze_sha256=gate['freeze_sha256'],
                      bundles_sha256=file_hash(a.bundles),candidates_sha256=file_hash(a.candidates),
                      candidate_frame_hash=frame_hash(frame),parent_frame_hash=frame_hash(parents),
                      native_references_sha256=file_hash(a.out/'native_references.jsonl'),
                      delta_proxy=delta_proxy,summary=summary,budget_equiv=16,budget_units=800,
                      training_seed=0,sampling_seed=0,h_results_present=False,
                      samples_frozen=False,seeds=dict(A=SEED_A,B=SEED_B))
        write_json(manifest_path,manifest)
        print(json.dumps(dict(M=summary['M'],candidates=summary['n_candidates'],delta_proxy=delta_proxy,
                              parent_strata=summary['parent_strata'],
                              candidate_strata=summary['candidate_strata']),indent=2))
        return
    manifest=json.loads(manifest_path.read_text())
    if manifest['samples_frozen'] or manifest['candidate_frame_hash']!=frame_hash(frame):
        raise ValueError('samples already frozen or source changed since D0')
    alloc=json.loads(a.allocation.read_text())
    sample_a=stratified_srswor(frame,alloc['A'],SEED_A)
    if set(alloc['B'])!={p['stratum'] for p in parents} or not set(alloc['B'])<=set(B_STRATA):
        raise ValueError('B allocation must cover every nonempty parent stratum')
    sizes=defaultdict(int)
    for p in parents:sizes[p['stratum']]+=1
    if 'neither_empty' in alloc['B'] and alloc['B']['neither_empty']!=sizes['neither_empty']:
        raise ValueError('neither_empty is a census')
    for h,m in alloc['B'].items():
        if m<sizes[h] and m<2:raise ValueError('sampled parent strata need m_h >= 2')
    sample_b=stratified_srswor(parents,alloc['B'],SEED_B)
    write_jsonl(a.out/'sample_A.jsonl',sample_a)
    write_jsonl(a.out/'sample_B_parents.jsonl',sample_b)
    manifest.update(stage='samples_frozen',samples_frozen=True,allocation=alloc,allocation_sha256=file_hash(a.allocation),
                    sample_A_sha256=file_hash(a.out/'sample_A.jsonl'),sample_B_sha256=file_hash(a.out/'sample_B_parents.jsonl'))
    write_json(manifest_path,manifest)
    print(json.dumps(dict(n_A=len(sample_a),n_B=len(sample_b)),indent=2))


if __name__=='__main__':main()
