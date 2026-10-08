"""Read-only strict replication audit, including chronology and preservation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    new=root/'strict-91298b4';old=root/'smc-2116d43';smoke=root/'strict-smoke-91298b4'
    manifest={s:load(new/(s+'_manifest.json')) for s in ('s0','g1','c1','initial_states','g2','c2')}
    proof=load(smoke/'smoke_complete.json');repeat=load(smoke/'cpu_repeat_check.json')
    assert proof['passed'] and repeat['passed'] and repeat['max_state_difference']==0
    assert all(not m['dirty'] and not m['smoke'] and m['source_commit']==proof['source_commit'] for m in manifest.values())
    assert (smoke/'smoke_complete.json').stat().st_mtime<(new/'s0_manifest.json').stat().st_mtime
    assert (new/'initial_states_manifest.json').stat().st_mtime<(new/'g2_manifest.json').stat().st_mtime
    for stage in ('s0','c1','c2'):assert manifest[stage]['gate_passed']
    assert manifest['s0']['n']==558 and manifest['s0']['raw_max_error']<=1e-6
    assert abs(manifest['s0']['raw_rho']['estimate']-.7491015704725383)<1e-9
    assert manifest['c1']['n']==5952 and manifest['c1']['event_status_mismatches']==0 and manifest['c1']['raw_max_error']<=1e-6
    for s in manifest['c2']['survivor_checks'].values():assert s['n']==2976 and s['agreement']>=.999
    initial=manifest['initial_states'];assert initial['clone_count']==35712 and len(initial['files'])==288
    assert initial['plans_sha256']==sha(new/'plans.json')
    assert manifest['g2']['consumed_initial_manifest_sha256']==sha(new/'initial_states_manifest.json')
    for f,h in initial['files'].items():assert sha(new/f)==h
    assert len(manifest['g1']['batches'])==24 and len(manifest['g2']['files'])==288
    r=load(new/'results.json');previous=load(old/'results.json')
    fields=['arms','comparisons','readings','diagnostics','checkpoint_correlations','parent_metrics','limitations']
    equal={f:r[f]==previous[f] for f in fields}
    assert all(equal.values()),equal
    assert (root/'strict-results-first.json').read_bytes()==(new/'results.json').read_bytes()
    assert load(new/'analysis_plumbing_check.json')['passed']
    # Verify all original frozen artifacts, including tensors, remain unchanged.
    sums=root/'dev-clone/docs/evidence/explore_smc/SHA256SUMS';preserved=0
    for line in sums.read_text().splitlines():
        digest,name=line.split('  ',1);assert sha(name)==digest,name;preserved+=1
    endpoints=load(new/'endpoint_rows.json');groups={}
    for row in endpoints:groups.setdefault((row['arm'],row['parent_id'],row['seed']),[]).append(row)
    assert len(endpoints)==77376 and len(groups)==2418
    assert all(len(v)==32 and {r['proposal'] for r in v}==set(range(32)) for v in groups.values())
    def cpu_time(path):
        text=path.read_text();vals={}
        for line in text.splitlines():
            for key in ('User time (seconds)','System time (seconds)'):
                if line.strip().startswith(key+':'):vals[key]=float(line.split(':')[-1])
        return sum(vals.values())
    timing={s:cpu_time(root/f'strict-{s}-time.txt') for s in ('s0','c1','c2','analyze')}
    timing['successful_cpu_smoke']=cpu_time(root/'strict-smoke-91298b4-time.txt')
    gpu={}
    for job in (2716,2717):
        text=(root/f'strict-slurm-{job}.txt').read_text();assert 'JobState=COMPLETED' in text and 'ExitCode=0:0' in text
        runtime=next(w.split('=')[1] for w in text.split() if w.startswith('RunTime='));h,m,s=map(int,runtime.split(':'));gpu[str(job)]=h*3600+m*60+s
    result=dict(passed=True,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                formal_source_commit=proof['source_commit'],scientific_fields_equal_to_original=equal,
                formal_analysis_byte_identical=True,original_files_preserved=preserved,
                n_candidates=len(endpoints),n_populations=len(groups),n_initial_batches=len(initial['files']),
                n_clones=initial['clone_count'],smoke_before_formal=True,initial_states_before_g2=True,
                cpu_process_seconds=timing,cpu_measured_core_hours=sum(timing.values())/3600,gpu_wall_seconds=gpu,
                scope='Chronology checked by original artifact mtimes and consumed manifest hash; mtimes alone are not cryptographic timestamps. CPU sum excludes tests, GPU-host CPU and the second analysis run.',
                inputs={str(p):sha(p) for p in [new/'results.json',old/'results.json',smoke/'smoke_complete.json',new/'initial_states_manifest.json',sums]})
    out=new/'strict_audit.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps(result,sort_keys=True,indent=2))


if __name__=='__main__':main()
