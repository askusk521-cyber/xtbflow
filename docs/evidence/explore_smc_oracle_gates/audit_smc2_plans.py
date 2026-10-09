"""Independent read-only raw-score ordering audit, no SMC helper shortcut."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
R=Path('/home/lhshen/xtbflow-runs/followup-20261009')
groups=defaultdict(dict)
files=[]
for method in ('aimnet','gxtb'):
    f=R/f'smc-checkpoint-{method}/rows.jsonl';files.append(f)
    for r in map(json.loads,f.open()):
        key=(f"{method}_{r['step']/50:.1f}",r['parent_id'],r['seed'])
        if r['proposal'] in groups[key]:raise ValueError('Duplicate score')
        groups[key][r['proposal']]=r
plans=json.loads((R/'smc2-new/plans.json').read_text())
assert len(plans)==1488 and len(groups)==1488
seen=set()
for p in plans:
    key=(p['arm'],p['parent_id'],p['seed']);assert key not in seen;seen.add(key)
    rows=groups[key];assert set(rows)==set(range(32))
    def order(j):
        s=rows[j]['score']
        h=sha256(f"{p['parent_id']}|{p['seed']}|{j}".encode()).hexdigest()
        return (0,s['barrier'],h) if s['status']=='ok' else (1,float('inf'),h)
    ranked=sorted(range(32),key=order)
    assert p['survivors']==ranked[:16]
    assert p['clones']==[list(pair) for pair in zip(ranked[16:],ranked[:16])]
    assert all(rows[j]['batch']==p['batch'] and rows[j]['start']==p['start'] and rows[j]['step']==p['step'] for j in rows)
    assert set(p['survivors']).isdisjoint(d for d,_ in p['clones'])
assert seen==set(groups)
files += [R/'smc2-new/plans.json',R/'smc2-decoded/manifest.json',R/'smc2-jobs/slurm-2754.txt']
gate=json.loads(files[-2].read_text())
assert gate['gate_passed'] and len(gate['survivor_checks'])==8
assert all(v['agreement']>=.999 for v in gate['survivor_checks'].values())
job=files[-1].read_text();assert 'JobState=COMPLETED' in job and 'RunTime=00:00:24' in job
out=dict(passed=True,n_plans=len(plans),n_scores=sum(map(len,groups.values())),
         source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
         config_sha256=sha256(Path('configs/explore/smc_oracle.json').read_bytes()).hexdigest(),
         input_sha256={str(f):sha256(f.read_bytes()).hexdigest() for f in files},
         survivor_checks=gate['survivor_checks'],slurm_job_id='2754',gpu_runtime_s=24,
         scope='Independent score ordering, slot mapping and survivor gate audit; not endpoint ranking or scientific analysis')
f=R/'smc2-plan-independent-audit.json'
if f.exists():raise FileExistsError(f)
f.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print('Independent audit PASS:1488 plans,47616 scores,8 survivor gates,Slurm2754')
