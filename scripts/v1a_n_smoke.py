"""Task N smoke check of a step-count control run on a few development parents.

Checks, per stream: every completed attempt used exactly the overridden number
of integration steps per role, the clock increments of each grid sum to one,
endpoints decode and go through the V1a proxy matcher, and the run directory
lies outside the sealed V1a directories. Also projects full-scale GPU hours
(342 formal parents x 3 training seeds) and stops above the handoff's limit.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np
from rdkit import RDLogger

from xtbflow.v1.clocks import increments
from xtbflow.v1.data import write_json
from xtbflow.v1.proxy import CatalogueMatcher

ROLE_OP={'event':'g','geometry':'h','joint':'f'}
PROGRAM_ROLES={'A0':('event','geometry'),'B0':('joint',)}
SEALED=('formal','final','screen','evidence','data','generators','scores')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--catalogue',type=Path,required=True)
    ap.add_argument('--v1a-root',type=Path,required=True)
    ap.add_argument('--full-parents',type=int,default=342)
    ap.add_argument('--seeds',type=int,default=3)
    ap.add_argument('--gpu-hour-limit',type=float,default=20.)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();RDLogger.DisableLog('rdApp.error')
    run=a.run.resolve();manifest=json.loads((run/'manifest.json').read_text())
    steps={k[len('nfe_'):]:v for k,v in manifest['nfe_overrides'].items()}
    checks=dict(manifest_complete=manifest['complete'],
                outside_sealed_dirs=not any(run.is_relative_to((a.v1a_root/d).resolve()) for d in SEALED),
                clock_increments_sum_to_one={role:bool(all(np.isclose(d.sum(),1.) and len(d)==n
                                                             for d in increments(manifest['config']['path'],n)))
                                             for role,n in steps.items()})
    parents={p['query_id']:p for p in json.loads((a.catalogue/'parent_catalog.json').read_text())}
    refs=defaultdict(list)
    for line in (a.catalogue/'reference_catalog.jsonl').read_text().splitlines():
        r=json.loads(line);refs[r['parent_id']].append(r)
    step_errors=[];status=Counter();per_arm=defaultdict(Counter)
    for line in (run/'streams.jsonl').read_text().splitlines():
        row=json.loads(line);arm=row['arm'];p=parents[row['query_id']]
        if row.get('nfe_steps')!={**dict(event=50,geometry=50,joint=50),**steps}:step_errors.append('stream nfe_steps')
        ops=defaultdict(Counter)
        for op in row['operations']:ops[op['attempt_id']][op['operation']]+=op['count']
        for att in row['attempts']:
            if att['status']!='COMPLETE':continue
            for role in PROGRAM_ROLES[arm]:
                n=steps.get(role,50)
                if ops[att['attempt_id']][ROLE_OP[role]]!=n:step_errors.append(f"{att['attempt_id']} {role}")
        matcher=CatalogueMatcher(p,refs[p['parent_id']])
        for c in row['candidates']:
            m=matcher.match(c['b_dec'],c['x'],c['is_fallback'],c['decode_status'])
            status[m['proxy_status']]+=1;per_arm[arm][m['proxy_status']]+=1
            per_arm[arm]['completed']+=1
        per_arm[arm]['streams']+=1
    n_parents=len({json.loads(l)['query_id'] for l in (run/'streams.jsonl').read_text().splitlines()})
    smoke_hours=sum(manifest['gpu_wall_s'].values())/3600
    projected_smoke=smoke_hours/n_parents*a.full_parents*a.seeds
    v1a={}
    for s in range(a.seeds):
        m=json.loads((a.v1a_root/f'screen/efficiency_s{s}/manifest.json').read_text())
        v1a[str(s)]=sum(m['gpu_wall_s'][arm] for arm in manifest['arms'])
    projected_v1a=sum(v1a.values())/3600
    checks.update(steps_exact=not step_errors,step_errors=step_errors[:20],
                  matcher_ran=sum(status.values())>0,any_valid_endpoint=sum(v for k,v in status.items() if k!='INVALID_OUTPUT')>0)
    passed=all(v for k,v in checks.items() if k not in ('step_errors','clock_increments_sum_to_one')) and \
        all(checks['clock_increments_sum_to_one'].values())
    report=dict(schema='xtbflow-v1a-n-smoke/1',exploratory=True,run=str(run),nfe_overrides=manifest['nfe_overrides'],
                n_parents=n_parents,arms=manifest['arms'],checks=checks,proxy_status=dict(status),
                per_arm={k:dict(v) for k,v in per_arm.items()},smoke_gpu_wall_s=manifest['gpu_wall_s'],
                projected_full_gpu_hours=dict(
                    from_smoke_per_parent=projected_smoke,
                    from_v1a_same_arm_wall=projected_v1a,
                    note=('Smoke batches hold only a few parents, so per-parent scaling overstates the full run; '
                          'the new arms make the same number of network calls per parent as V1a A0/B0 under the 3200-unit cap.')),
                gpu_hour_limit=a.gpu_hour_limit,
                status='SMOKE_PASS' if passed and max(projected_smoke,projected_v1a)<=a.gpu_hour_limit else
                       ('STOP_GPU_HOURS' if passed else 'SMOKE_FAIL'))
    write_json(a.out,report)
    print(json.dumps({k:report[k] for k in ('status','checks','projected_full_gpu_hours','per_arm')},indent=2))


if __name__=='__main__':main()
