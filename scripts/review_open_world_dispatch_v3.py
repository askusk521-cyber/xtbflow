"""O2 serial dispatcher, version 3: separate pre-QC infrastructure failures.

Version 2 (scripts/review_open_world_dispatch_v2.py, ledger o2-jobs/ledger_v2.json)
counted any runner exception as a chain failure. On 2026-10-10 the node's NVIDIA
user-space library (595.99) no longer matched the loaded kernel module (595.91.07);
nvidia-smi exited 18 inside Slurm jobs 2825-2827 before any quantum-chemistry stage
ran, and v2 stopped after three such items. Those were environment failures, not
attempts of the frozen protocol.

Version 3 changes only the classification and bookkeeping:
- a runner exception with no recorded QC stage (empty stage_timings) is
  INFRASTRUCTURE and stops dispatching at once;
- `--quarantine-preqc` moves work directories of earlier pre-QC failures (empty
  stage_timings, no verdict) into o2-run-preqc-failed/ with a JSON record, so the
  unchanged runner can attempt those items for the first time. Directories with
  any QC stage are never moved and stay prior attempts (e.g. Slurm 2753);
- GPU seconds of all earlier ledgers count against the 16-hour budget.
Everything else (serial Slurm, 2-hour reservation, continue after recorded chain
exceptions or timeouts, stop after three consecutive) is as in version 2.
"""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_open_world_dispatch_v2 import (BUDGET_S, MAX_CONSECUTIVE_FAILURES, RESERVE_S, command,  # noqa: E402
                                          seconds, wait_terminal)


def preqc_failure(work):
    """True when the runner failed before any QC stage and left no verdict."""
    failure = work / 'unexpected_failure.json'
    if not failure.exists() or (work / 'verdict.json').exists():
        return False
    return not json.loads(failure.read_text()).get('stage_timings')


def classify(state, work):
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
        detail = f"{f.get('type')}:{f.get('message')}"
        if not f.get('stage_timings'):
            return 'INFRASTRUCTURE', 'pre-QC ' + detail, True
        return 'CHAIN_EXCEPTION', detail, False
    return 'INFRASTRUCTURE', job_state, True


def quarantine(root, ledgers):
    """Move pre-QC failed work dirs aside, recording where each came from."""
    jobs = {r['item_id']: r.get('job_id') for ledger in ledgers for r in ledger if r.get('job_id')}
    target = root / 'o2-run-preqc-failed'
    moved = []
    for work in sorted((root / 'o2-run').iterdir()):
        if work.is_dir() and preqc_failure(work):
            item_id = work.name.split('_', 1)[1]
            dest = target / f'{work.name}.job{jobs.get(item_id)}'
            if dest.exists():
                raise FileExistsError(dest)
            target.mkdir(exist_ok=True)
            shutil.move(str(work), str(dest))
            moved.append(dict(item_id=item_id, job_id=jobs.get(item_id), from_=str(work), to=str(dest),
                              moved_unix=time.time()))
    record = target / 'quarantine.json'
    previous = json.loads(record.read_text()) if record.exists() else []
    record.write_text(json.dumps(previous + moved, indent=2) + '\n')
    return moved


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True, help='clean execution clone of the frozen runner')
    p.add_argument('--quarantine-preqc', action='store_true')
    a = p.parse_args()
    root, source = a.root.resolve(), a.source.resolve()
    jobs_dir = root / 'o2-jobs'
    plan = json.loads((root / 'o2-plan.json').read_text())
    ledger_path = jobs_dir / 'ledger_v3.json'
    if ledger_path.exists():
        raise FileExistsError('Version-3 dispatcher is single-use; inspect ledger_v3.json')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=source, text=True).strip():
        raise RuntimeError('Execution clone is dirty')
    gpu = subprocess.run(['srun', '--partition=main', '--gres=gpu:pro6000:1', '--time=00:02:00',
                          '--job-name=fu-gpucheck', 'nvidia-smi', '--query-gpu=name,driver_version',
                          '--format=csv,noheader'], capture_output=True, text=True)
    (jobs_dir / 'gpu_precheck_v3.txt').write_text(gpu.stdout + gpu.stderr)
    if gpu.returncode:
        raise RuntimeError('GPU precheck failed inside Slurm; fix the node before dispatching')
    earlier = [json.loads((jobs_dir / name).read_text()) for name in ('ledger.json', 'ledger_v2.json')
               if (jobs_dir / name).exists()]
    if a.quarantine_preqc:
        print(json.dumps(dict(quarantined=quarantine(root, earlier))), flush=True)
    used = sum(r.get('runtime_s', 0) for ledger in earlier for r in ledger if r.get('job_id'))
    ledger, consecutive = [], 0

    def save():
        ledger_path.write_text(json.dumps(ledger, indent=2) + '\n')

    for item in plan['items']:
        work = root / 'o2-run' / (item['kind'] + '_' + item['item_id'])
        if work.exists():
            ledger.append(dict(item_id=item['item_id'], kind=item['kind'], parent_id=item['parent_id'],
                               outcome='PRIOR_ATTEMPT'))
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
