"""Read-only replay of all saved raw Hessian numerical diagnostics."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path.cwd()/'scripts'))
from followup_u_math import hessian_metrics
R=Path('/home/lhshen/xtbflow-runs/followup-20261009')
out={}
for method in ('gfn2','aimnet'):
    root=R/('u-'+method);rows=list(map(json.loads,(root/'rows.jsonl').open()));n=0;maxerr=0.
    for r in rows:
        if r['status']!='ok':continue
        with np.load(root/r['hessian_file']) as d:
            actual=hessian_metrics(d['z'],d['x'],d['h'],d['u'])
        for key,value in actual.items():
            error=float(np.max(np.abs(np.asarray(value)-np.asarray(r['metrics'][key]))))
            maxerr=max(maxerr,error)
        n+=1
    if maxerr>1e-10:raise RuntimeError('Numerical replay mismatch')
    out[method]=dict(n=n,max_abs_error=maxerr,rows_sha256=sha256((root/'rows.jsonl').read_bytes()).hexdigest())
p=R/'u-npz-replay.json'
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print('Raw Hessian metric replay PASS')
