"""Task R step R0: verify the sealed V1a screen, re-match its raw streams, reproduce V1a.

Reads only sealed files under the V1a run root and writes a new directory:
  streams.jsonl.gz      one compact row per (parent, arm, training seed) stream
  extract_manifest.json hash checks, provenance, rarity labels, reproduction result
Exits non-zero (and still writes the manifest) when any hash or reproduced
number differs; the handoff requires stopping for the owner in that case.
"""
import argparse
from collections import Counter,defaultdict
import gzip
import json
from multiprocessing import Pool
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from rdkit import RDLogger

from xtbflow.v1.costs import BUDGETS,prefixes
from xtbflow.v1.data import file_hash
from xtbflow.v1.formal import ARMS,load_freeze,rarity_stratum
from xtbflow.v1.metrics import prefix_auc
from xtbflow.v1.proxy import CatalogueMatcher
from xtbflow.v1.reanalysis import mode_key,same_event_rmsd,verification_flags

RAW_FIELDS=('candidate_id','completion_units','b_dec','x','is_fallback','decode_status')
CURVE_KEYS=('hits','event_hits','any_reference_hits','completed_candidates','valid_candidates','unique_events')
ROW_KEYS=CURVE_KEYS+('auc','event_auc','best_event_count','best_reference_count','n_proposals','spent_units')
STATE={}


def sealed_hashes(path):
    rows=[line.split() for line in Path(path).read_text().splitlines() if line.strip()]
    return {name:digest for digest,name in rows}


def git(*args,cwd):
    return subprocess.run(['git',*args],cwd=cwd,check=True,capture_output=True,text=True).stdout.strip()


def cost_bin(table,n):
    for row in table['bins']:
        if row['min_atoms']<=n<=row['max_atoms']:return row
    raise ValueError(f'no cost bin for {n} atoms')


def init_worker(parents,matchers,specs):
    STATE.update(parents=parents,matchers=matchers,specs=specs)
    RDLogger.DisableLog('rdApp.error')


def process(line):
    row=json.loads(line);qid=row['query_id']
    parent=STATE['parents'][qid];matcher=STATE['matchers'][qid]
    if row['sampling_seed']!=0:raise ValueError('sampling seed must be 0')
    recs=[]
    for i,c in enumerate(row['candidates']):
        missing=[k for k in RAW_FIELDS if k not in c]
        if missing:raise ValueError(f'candidate lacks {missing}')
        m=matcher.match(c['b_dec'],c['x'],c['is_fallback'],c['decode_status'])
        recs.append(dict(i=i,candidate_id=c['candidate_id'],completion_units=c['completion_units'],
                         decode_status=c['decode_status'],is_fallback=c['is_fallback'],b=c['b_dec'],x=c['x'],
                         **{k:m[k] for k in ('predicted_channel_id','proxy_status','hits_best_reference','hits_best_event')}))
    perms=np.asarray(parent['permutations'],dtype=np.int64);cache={}

    def rmsd(u,c):
        key=(u['i'],c['i'])
        if key not in cache:cache[key]=same_event_rmsd(perms,u['b'],u['x'],c['b'],c['x'])
        return cache[key]
    opens={mode_key(s):verification_flags(recs,s,rmsd) for s in STATE['specs']}
    pre=prefixes(recs)
    hits=[int(any(r['hits_best_reference'] for r in pre[str(k)])) for k in BUDGETS]
    event=[int(any(r['hits_best_event'] for r in pre[str(k)])) for k in BUDGETS]
    repro=dict(hits=hits,event_hits=event,
               any_reference_hits=[int(any(r['proxy_status']=='PROXY_MATCH' for r in pre[str(k)])) for k in BUDGETS],
               completed_candidates=[len(pre[str(k)]) for k in BUDGETS],
               valid_candidates=[sum(r['proxy_status']!='INVALID_OUTPUT' for r in pre[str(k)]) for k in BUDGETS],
               unique_events=[len({r['predicted_channel_id'] for r in pre[str(k)] if r['predicted_channel_id'] is not None})
                              for k in BUDGETS],
               auc=float(prefix_auc(hits)),event_auc=float(prefix_auc(event)),
               best_event_count=sum(bool(r['hits_best_event']) for r in recs),
               best_reference_count=sum(bool(r['hits_best_reference']) for r in recs),
               n_proposals=row['n_proposals'],spent_units=row['spent_units'],
               budget16_candidate_ids=[r['candidate_id'] for r in pre['16']],
               budget16_best_event_candidate_ids=[r['candidate_id'] for r in pre['16'] if r['hits_best_event']])
    compact=dict(parent_id=parent['parent_id'],query_id=qid,split_group=parent['split_group'],
                 n_atoms=len(parent['atomic_numbers']),arm=row['arm'],training_seed=row['training_seed'],
                 cap_units=row['cap_units'],spent_units=row['spent_units'],n_proposals=row['n_proposals'],
                 completion_units=[r['completion_units'] for r in recs],
                 joint=[bool(r['hits_best_reference']) for r in recs],event=[bool(r['hits_best_event']) for r in recs],
                 proxy_status=[r['proxy_status'] for r in recs],channel=[r['predicted_channel_id'] for r in recs],
                 opens=opens,n_rmsd_pairs=len(cache))
    check={r['candidate_id']:(r['proxy_status'],r['predicted_channel_id'],bool(r['hits_best_reference']),
                              bool(r['hits_best_event'])) for r in recs}
    return compact,repro,check


def deviation(a,b):
    if isinstance(a,list):
        if len(a)!=len(b):return float('inf')
        return max([deviation(x,y) for x,y in zip(a,b)],default=0.)
    return abs(float(a)-float(b))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-root',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--sealed',type=Path,required=True,help='docs/evidence/v1a/formal/SHA256SUMS')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=16)
    a=ap.parse_args();started=time.monotonic();RDLogger.DisableLog('rdApp.error')
    a.out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(a.config.read_text());op=cfg['operational'];root=a.run_root
    repo=Path(__file__).resolve().parents[1]
    manifest=dict(schema='xtbflow-v1a-r-extract/1',exploratory=True,status='RUNNING',
                  code_commit=git('rev-parse','HEAD',cwd=repo),
                  worktree_clean=git('status','--porcelain',cwd=repo)=='',
                  config_path=str(a.config),config_sha256=file_hash(a.config),
                  config_commit=git('log','-1','--format=%H','--',str(a.config.resolve().relative_to(repo)),cwd=repo),
                  hash_checks={},failures=[])

    def check(name,actual,expected):
        ok=actual==expected;manifest['hash_checks'][name]=dict(actual=actual,expected=expected,ok=ok)
        if not ok:manifest['failures'].append(f'hash mismatch: {name}')

    def finish(code):
        manifest['wall_s']=time.monotonic()-started
        (a.out/'extract_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
        print(json.dumps({k:manifest[k] for k in ('status','failures')},indent=2));sys.exit(code)

    sealed=sealed_hashes(a.sealed)
    for rel in ('formal/freeze.json','final/gate.json','data/screen_queries.jsonl','screen/rarity.json',
                *[f'screen/efficiency_s{s}.json' for s in op['source']['training_seeds']]):
        check(rel,file_hash(root/rel),sealed[rel])
    freeze=load_freeze(root/'formal/freeze.json');gate=json.loads((root/'final/gate.json').read_text())
    check('gate.freeze_sha256',gate['freeze_sha256'],freeze['freeze_sha256'])
    for name in ('parent_catalog.json','reference_catalog.jsonl','split_manifest.json'):
        check(f'data/{name}',file_hash(root/'data'/name),freeze['bindings']['catalogue'][name])
    check('cost_calibration_512.json',file_hash(root/'evidence/cost_calibration_512.json'),
          freeze['bindings']['cost_table_sha256'])
    manifest['unsealed_inputs']={'evidence/cost_calibration.json':file_hash(root/'evidence/cost_calibration.json')}
    reports={}
    for s in op['source']['training_seeds']:
        rep=json.loads((root/f'screen/efficiency_s{s}.json').read_text());reports[s]=rep
        run=json.loads((root/f'screen/efficiency_s{s}/manifest.json').read_text())
        raw=file_hash(root/f'screen/efficiency_s{s}/streams.jsonl')
        check(f'screen/efficiency_s{s}/streams.jsonl',raw,rep['source_sha256'])
        check(f'screen/efficiency_s{s}/manifest.streams_sha256',run['streams_sha256'],rep['source_sha256'])
        check(f'efficiency_s{s}.freeze_sha256',rep['freeze_sha256'],freeze['freeze_sha256'])
        check(f'efficiency_s{s}.training_seed',str(rep['training_seed']),str(s))
    if manifest['failures']:
        manifest['status']='STOP_HASH_MISMATCH';finish(2)

    order=sorted(freeze['parent_ids']);wanted=set(order)
    parents={p['query_id']:p for p in json.loads((root/'data/parent_catalog.json').read_text()) if p['parent_id'] in wanted}
    if sorted(p['parent_id'] for p in parents.values())!=order:raise ValueError('parent catalogue differs from freeze')
    refs=defaultdict(list)
    for line in (root/'data/reference_catalog.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['parent_id'] in wanted:refs[r['parent_id']].append(r)
    matchers={q:CatalogueMatcher(p,refs[p['parent_id']]) for q,p in parents.items()}
    specs=[cfg['verification_units']['main'],*cfg['verification_units']['sensitivity']]

    rarity=json.loads((root/'screen/rarity.json').read_text())
    labels={}
    for r in rarity['summaries']['B0']['parent_rows']:
        if r['n_proposals']!=128:raise ValueError('rarity pilot did not plan 128 proposals')
        labels[r['parent_id']]=rarity_stratum(r['best_event_count']/128)
    sizes=Counter(labels.values());frozen_sizes={k:v['n_parents'] for k,v in gate['secondary']['rarity_strata'].items()}
    manifest['rarity']=dict(labels=labels,sizes=dict(sizes),gate_sizes=frozen_sizes,ok=dict(sizes)==frozen_sizes)
    if not manifest['rarity']['ok']:manifest['failures'].append('rarity strata differ from gate.json')
    dev=json.loads((root/'evidence/cost_calibration.json').read_text())
    big=json.loads((root/'evidence/cost_calibration_512.json').read_text())
    manifest['parents']={p['parent_id']:dict(
        query_id=q,split_group=p['split_group'],n_atoms=len(p['atomic_numbers']),
        f_seconds_per_call_dev=cost_bin(dev,len(p['atomic_numbers']))['seconds_per_batch']['f'],
        dev_batch_size=cost_bin(dev,len(p['atomic_numbers']))['batch_size'],
        f_seconds_per_call_frozen=cost_bin(big,len(p['atomic_numbers']))['seconds_per_batch']['f'],
        frozen_batch_size=cost_bin(big,len(p['atomic_numbers']))['batch_size'],
        frozen_weights=cost_bin(big,len(p['atomic_numbers']))['weights']) for q,p in parents.items()}

    repro=defaultdict(dict);worst=defaultdict(float);mismatch=Counter();n_rmsd=0;side=Counter()
    with gzip.open(a.out/'streams.jsonl.gz','wt',encoding='utf-8') as out, \
         Pool(a.workers,initializer=init_worker,initargs=(parents,matchers,specs)) as pool:
        for s in op['source']['training_seeds']:
            path=root/f'screen/efficiency_s{s}/streams.jsonl';seen=set();checks={}
            with path.open() as f:
                for compact,values,cands in pool.imap(process,f,chunksize=2):
                    key=(compact['parent_id'],compact['arm'])
                    if compact['training_seed']!=s or key in seen:raise ValueError('duplicate or mis-seeded stream')
                    seen.add(key);out.write(json.dumps(compact,allow_nan=False)+'\n')
                    repro[s][key]=values;checks.update(cands);n_rmsd+=compact['n_rmsd_pairs']
            if seen!={(p,arm) for p in order for arm in ARMS}:raise ValueError(f'seed {s}: incomplete streams')
            # Reproduction against the sealed per-parent rows of the V1a report.
            for arm in ARMS:
                for row in reports[s]['summaries'][arm]['parent_rows']:
                    mine=repro[s][row['parent_id'],arm]
                    for k in ROW_KEYS:worst['parent_rows.'+k]=max(worst['parent_rows.'+k],deviation(mine[k],row[k]))
                    v=row['v1b_budget16']
                    mismatch['budget16_candidate_ids']+=mine['budget16_candidate_ids']!=v['candidate_ids']
                    mismatch['budget16_best_event_candidate_ids']+=(
                        mine['budget16_best_event_candidate_ids']!=v['best_event_candidate_ids'])
            # Unhashed V1a side file: cross-check only.
            side_path=root/f'screen/efficiency_s{s}.candidates.jsonl'
            if side_path.exists():
                ids=set()
                for line in side_path.open():
                    r=json.loads(line);mine=checks.get(r['candidate_id']);ids.add(r['candidate_id'])
                    side['compared']+=1
                    side['differ']+=mine!=(r['proxy_status'],r['predicted_channel_id'],bool(r['hits_best_reference']),
                                           bool(r['hits_best_event']))
                side['missing_or_extra']+=len(ids^set(checks))
    tol=op['reproduction_tolerance'];curves={}
    for arm in ARMS:
        rows=[[repro[s][p,arm] for p in order] for s in op['source']['training_seeds']]
        mine={k:np.mean([[r[k] for r in rr] for rr in rows],axis=(0,1)).tolist() for k in CURVE_KEYS}
        mine.update(auc=float(np.mean([[r['auc'] for r in rr] for rr in rows])),
                    event_auc=float(np.mean([[r['event_auc'] for r in rr] for rr in rows])))
        curves[arm]=mine
        for k,v in mine.items():worst['gate.curves.'+k]=max(worst['gate.curves.'+k],deviation(v,gate['curves'][arm][k]))
    manifest['reproduction']=dict(tolerance=tol,max_abs_deviation=dict(worst),list_mismatches=dict(mismatch),
                                  recomputed_curves=curves,side_file_cross_check=dict(side),
                                  passed=max(worst.values())<=tol and not any(mismatch.values()))
    manifest['n_streams']=sum(len(v) for v in repro.values());manifest['n_rmsd_pairs']=n_rmsd
    manifest['coordinates_available']=True;manifest['verification_modes']=[mode_key(s) for s in specs]
    manifest['streams_gz_sha256']=file_hash(a.out/'streams.jsonl.gz')
    if not manifest['reproduction']['passed']:manifest['failures'].append('V1a numbers not reproduced')
    if manifest['failures']:
        manifest['status']='STOP_REPRODUCTION_FAILED';finish(3)
    manifest['status']='EXTRACT_REPRODUCTION_PASS';finish(0)


if __name__=='__main__':main()
