"""O2 serial dispatcher, version 2: record chain exceptions as failures and continue.

Version 1 (run-root dispatch_o2.py) stopped at the first protocol-external
exception (Slurm 2753, IRC SCF_LIMIT wrapped as geomeTRIC EngineError). The
owner delegated the decision on 2026-10-09; the chosen handling is: keep that
attempt as a computational failure, never retry it, and continue the untouched
items. Nothing here imports or changes qc_protocol; the per-item runner is the
unchanged scripts/review_open_world_dft.py of the frozen execution clone.

Rules, all deterministic and outcome-independent:
- items whose work directory already exists are prior attempts and are skipped;
- one Slurm job at a time for the whole user (squeue must be empty to submit);
- a 2-hour reservation per job against 16 GPU hours, counting prior attempts;
- FAILED with the runner's unexpected_failure.json -> CHAIN_EXCEPTION, continue;
- TIMEOUT -> JOB_TIMEOUT, continue; any other non-COMPLETED state, or FAILED
  without the runner's exception record, is infrastructure and stops;
- three consecutive CHAIN_EXCEPTION/JOB_TIMEOUT outcomes stop dispatching.
Work directories are never written by this dispatcher; outcomes go to the
version-2 ledger only.
"""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

BUDGET_S = 16 * 3600
RESERVE_S = 7200
ACTIVE = ('RUNNING', 'PENDING', 'COMPLETING', 'CONFIGURING')
MAX_CONSECUTIVE_FAILURES = 3


def command(args):
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout


def parse_slurm(text):
    return dict(re.findall(r'(\w+)=([^\s]+)', text))


def seconds(value):
    """Slurm [D-]HH:MM:SS -> seconds."""
    days, sep, clock = value.partition('-')
    if not sep:
        clock, days = days, '0'
    parts = [int(p) for p in clock.split(':')]
    if len(parts) != 3:
        raise ValueError('Unsupported Slurm runtime: ' + value)
    return int(days) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def classify(state, work):
    """Map a terminal Slurm record and the runner's files to a ledger outcome."""
    job_state = state['JobState']
    if job_state == 'COMPLETED' and state.get('ExitCode') == '0:0':
        verdict = work / 'verdict.json'
        if not verdict.exists() or not (work / 'environment.json').exists():
            return 'MISSING_RECORDS', None, True
        v = json.loads(verdict.read_text())
        return 'VERDICT', v.get('strict_status', v.get('status')), False
    if job_state == 'TIMEOUT':
        return 'JOB_TIMEOUT', None, False
    failure = work / 'unexpected_failure.json'
    if job_state == 'FAILED' and failure.exists():
        f = json.loads(failure.read_text())
        return 'CHAIN_EXCEPTION', f"{f.get('type')}:{f.get('message')}", False
    return 'INFRASTRUCTURE', job_state, True


def wait_terminal(job, jobs_dir):
    first_pending = None
    while True:
        proc = subprocess.run(['scontrol', 'show', 'job', str(job)], capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError('Lost Slurm terminal state for ' + str(job))
        (jobs_dir / f'slurm-{job}-latest.txt').write_text(proc.stdout)
        state = parse_slurm(proc.stdout)
        if state['JobState'] not in ACTIVE:
            (jobs_dir / f'slurm-{job}-terminal.txt').write_text(proc.stdout)
            return state
        if state['JobState'] == 'PENDING':
            first_pending = first_pending or time.monotonic()
            if time.monotonic() - first_pending > 7200:
                raise RuntimeError('Own job queued beyond two hours; stop and ask owner')
        time.sleep(5)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True, help='follow-up run root holding o2-plan.json')
    p.add_argument('--source', type=Path, required=True, help='clean execution clone of the frozen runner')
    a = p.parse_args()
    root, source = a.root.resolve(), a.source.resolve()
    jobs_dir = root / 'o2-jobs'
    plan = json.loads((root / 'o2-plan.json').read_text())
    prior = json.loads((jobs_dir / 'ledger.json').read_text())
    ledger_path = jobs_dir / 'ledger_v2.json'
    if ledger_path.exists():
        raise FileExistsError('Version-2 dispatcher is single-use; inspect ledger_v2.json')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=source, text=True).strip():
        raise RuntimeError('Execution clone is dirty')
    used = sum(r['runtime_s'] for r in prior)
    prior_items = {r['item_id']: r for r in prior}
    ledger, consecutive = [], 0

    def save():
        ledger_path.write_text(json.dumps(ledger, indent=2) + '\n')

    for item in plan['items']:
        work = root / 'o2-run' / (item['kind'] + '_' + item['item_id'])
        if work.exists():
            ledger.append(dict(item_id=item['item_id'], kind=item['kind'], parent_id=item['parent_id'],
                               outcome='PRIOR_ATTEMPT', prior=prior_items.get(item['item_id'])))
            save()
            continue
        if used + RESERVE_S > BUDGET_S:
            raise RuntimeError('Budget reservation would exceed 16 GPU hours; stop before submitting')
        waiting = time.monotonic()
        while command(['squeue', '-h', '-u', 'lhshen', '-o', '%i']).strip():
            if time.monotonic() - waiting > 7200:
                raise RuntimeError('Queue conflict persisted two hours')
            time.sleep(10)
        wrap = ('cd ' + shlex.quote(str(source)) + '; export PYTHONPATH=src:vendor/mechai_reusable OMP_NUM_THREADS=8; '
                '/home/lhshen/miniconda3/envs/xtbflow-qc/bin/python scripts/review_open_world_dft.py run '
                '--plan ' + shlex.quote(str(root / 'o2-plan.json')) + ' --item ' + item['item_id'] +
                ' --work ' + shlex.quote(str(root / 'o2-run')))
        job = command(['sbatch', '--parsable', '--partition=main', '--gres=gpu:pro6000:1',
                       '--cpus-per-task=8', '--mem=64G', '--time=02:00:00', '--job-name=fu-dft-' + item['kind'],
                       '--output=' + str(jobs_dir / 'dft-%j.log'), '--wrap', wrap]).strip()
        ledger.append(dict(item_id=item['item_id'], kind=item['kind'], parent_id=item['parent_id'],
                           job_id=job, outcome='SUBMITTED', submitted_unix=time.time()))
        save()
        state = wait_terminal(job, jobs_dir)
        runtime = seconds(state['RunTime'])
        used += runtime
        outcome, detail, stop = classify(state, work)
        ledger[-1].update(outcome=outcome, detail=detail, slurm_state=state['JobState'],
                          exit_code=state.get('ExitCode'), runtime_s=runtime, cumulative_gpu_s=used)
        save()
        print(json.dumps(ledger[-1]), flush=True)
        if stop:
            raise RuntimeError(f'{outcome} for job {job}; stop and inspect')
        if used > BUDGET_S:
            raise RuntimeError('Aggregate GPU budget exceeded')
        consecutive = consecutive + 1 if outcome in ('CHAIN_EXCEPTION', 'JOB_TIMEOUT') else 0
        if consecutive >= MAX_CONSECUTIVE_FAILURES:
            raise RuntimeError('Three consecutive chain failures; stop and inspect')
    print('All planned items reached terminal states; analysis still required.', flush=True)


if __name__ == '__main__':
    main()
