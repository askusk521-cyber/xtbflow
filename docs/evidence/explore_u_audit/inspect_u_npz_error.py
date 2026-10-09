import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path.cwd()/'scripts'))
from followup_u_math import hessian_metrics
R=Path('/home/lhshen/xtbflow-runs/followup-20261009');out={}
for method in ('gfn2','aimnet'):
    root=R/('u-'+method);errors={};readings_changed=[]
    for r in map(json.loads,(root/'rows.jsonl').open()):
        if r['status']!='ok':continue
        with np.load(root/r['hessian_file']) as d:actual=hessian_metrics(d['z'],d['x'],d['h'],d['u'])
        for key,value in actual.items():
            error=float(np.max(np.abs(np.asarray(value)-np.asarray(r['metrics'][key]))))
            if error>errors.get(key,{}).get('error',-1):errors[key]=dict(error=error,reference_id=r['reference_id'],expected=r['metrics'][key],actual=value)
        if actual['imaginary_count']!=r['metrics']['imaginary_count'] or (actual['c_u']<0)!=(r['metrics']['c_u']<0):readings_changed.append(r['reference_id'])
    out[method]=dict(max_errors=errors,sign_or_frequency_changes=readings_changed)
(R/'u-npz-replay-diagnostic.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
