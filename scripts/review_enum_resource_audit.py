"""Conservative CPU wall-allocation ledger including interrupted owned runs."""
import argparse
import json
from pathlib import Path
from xtbflow.v1.data import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    # Bounds for interrupted launches, rounded UP from recorded launch/stop times.
    interrupted=[
        dict(run='slurm2691',processes=1,wall_seconds_upper=660),
        dict(run='feasibility-cpu-1d6e895',processes=1,wall_seconds_upper=1800),
        dict(run='feasibility-cpu-4f0102a',processes=1,wall_seconds_upper=1800),
        dict(run='feasibility-shards-7d4bec3/index2',processes=1,wall_seconds_upper=900),
        dict(run='excluded344-screen-pilot',processes=8,wall_seconds_upper=300),
        dict(run='initial-screen-9403d6c',processes=8,wall_seconds_upper=1200),
        dict(run='initial-dev-7c28436',processes=2,wall_seconds_upper=1200),
        dict(run='initial-training-6f8ab54',processes=4,wall_seconds_upper=1800),
        dict(run='training-d414174-shard0',processes=1,wall_seconds_upper=1500)]
    completed=[]
    import re
    patterns=['feasibility-shards-7d4bec3/parent-1.log','feasibility-shards-7d4bec3/parent-3.log',
              'feasibility-shards-7d4bec3/parent-4.log','feasibility-fast-1f02afa/*.log',
              'enum-screen-9403d6c/resume*.log','enum-dev-7c28436/resume*.log',
              'training-b3-6f8ab54/resume-d414174-[123].log','training-b3-6f8ab54/resume-15a65e8-0.log']
    for pattern in patterns:
        for path in sorted(a.run.glob(pattern)):
            text=path.read_text()
            if 'Exit status: 0' not in text:raise RuntimeError('unfinished resource log '+str(path))
            user=re.search(r'User time \(seconds\): ([\d.]+)',text);system=re.search(r'System time \(seconds\): ([\d.]+)',text)
            completed.append(dict(path=str(path),cpu_seconds=float(user[1])+float(system[1])))
    upper=sum(r['processes']*r['wall_seconds_upper'] for r in interrupted)+sum(r['cpu_seconds'] for r in completed)
    write_json(a.out,dict(interrupted_conservative_bounds=interrupted,completed=completed,
        cpu_seconds_upper_before_preparation=upper,cpu_core_hours_upper_before_preparation=upper/3600,
        preparation_allowance_core_hours=1.0,cpu_core_hours_conservative_total=upper/3600+1,
        cpu_budget_core_hours=50,within_cpu_budget=upper/3600+1<=50,
        note='Interrupted values are upper bounds, not measured exact CPU use. Preparation allowance covers tests, audits and un-timed short feasibility checks. GPU allocations accounted separately.'))


if __name__=='__main__':main()
