"""O2 serial dispatcher, version 4: run the version-2 runner after a CUDA precheck.

Version 3 (scripts/review_open_world_dispatch_v3.py) never ran: its GPU precheck
calls nvidia-smi, which has failed on n2 since 2026-10-10 (NVML user-space
library 595.99 vs loaded kernel module 595.91.07) although CUDA itself works.
The owner approved on 2026-10-17 a versioned change limited to how the GPU is
checked and recorded; see docs/evidence/review_open_world_o2/DECISION_NVML_ZH.md.

Version 4 differs from version 3 only in:
- the precheck runs `scripts/review_open_world_dft_v2.py gpu-check` inside Slurm
  (CUDA runtime via cupy) instead of nvidia-smi; output in gpu_precheck_v4.txt;
- each item runs scripts/review_open_world_dft_v2.py, i.e. the unchanged
  version-1 runner with only its nvidia-smi query answered from the CUDA runtime;
- ledger o2-jobs/ledger_v4.json; ledger.json, ledger_v2.json and ledger_v3.json
  (if present) all count against the 16 GPU-hour budget and quarantine lookup.
Classification, quarantine of pre-QC failures, serial submission, the 2-hour
reservation, continue-after-recorded-exception and the three-in-a-row stop are
imported from versions 2 and 3 unchanged.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_open_world_dispatch_v2 import (BUDGET_S, MAX_CONSECUTIVE_FAILURES, RESERVE_S, command,  # noqa: E402
                                          seconds, wait_terminal)
from review_open_world_dispatch_v3 import classify, quarantine  # noqa: E402

PYTHON = '/home/lhshen/miniconda3/envs/xtbflow-qc/bin/python'
RUNNER = 'scripts/review_open_world_dft_v2.py'
EARLIER_LEDGERS = ('ledger.json', 'ledger_v2.json', 'ledger_v3.json')


def _in_source(source, args):
    return ('cd ' + shlex.quote(str(source)) + '; export PYTHONPATH=src:vendor/mechai_reusable OMP_NUM_THREADS=8; '
            + PYTHON + ' ' + RUNNER + ' ' + args)


def precheck_command(source):
    """Slurm one-GPU job that must print the CUDA-runtime GPU record and exit 0."""
    return ['srun', '--partition=main', '--gres=gpu:pro6000:1', '--cpus-per-task=1', '--mem=4G',
            '--time=00:05:00', '--job-name=fu-gpucheck', 'bash', '-c', _in_source(source, 'gpu-check')]


def item_wrap(source, root, item_id):
    """Shell command sbatch runs for one plan item."""
    return _in_source(source, 'run --plan ' + shlex.quote(str(root / 'o2-plan.json')) + ' --item ' + item_id +
                      ' --work ' + shlex.quote(str(root / 'o2-run')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True, help='clean execution clone containing the v2 runner')
    p.add_argument('--quarantine-preqc', action='store_true')
    a = p.parse_args()
    root, source = a.root.resolve(), a.source.resolve()
    jobs_dir = root / 'o2-jobs'
    plan = json.loads((root / 'o2-plan.json').read_text())
    ledger_path = jobs_dir / 'ledger_v4.json'
    if ledger_path.exists():
        raise FileExistsError('Version-4 dispatcher is single-use; inspect ledger_v4.json')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=source, text=True).strip():
        raise RuntimeError('Execution clone is dirty')
    gpu = subprocess.run(precheck_command(source), capture_output=True, text=True)
    (jobs_dir / 'gpu_precheck_v4.txt').write_text(gpu.stdout + gpu.stderr)
    if gpu.returncode or 'source=cuda-runtime' not in gpu.stdout:
        raise RuntimeError('CUDA precheck failed inside Slurm; inspect gpu_precheck_v4.txt')
    earlier = [json.loads((jobs_dir / name).read_text()) for name in EARLIER_LEDGERS
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
        job = command(['sbatch', '--parsable', '--partition=main', '--gres=gpu:pro6000:1',
                       '--cpus-per-task=8', '--mem=64G', '--time=02:00:00', '--job-name=fu-dft-' + item['kind'],
                       '--output=' + str(jobs_dir / 'dft-%j.log'), '--wrap',
                       item_wrap(source, root, item['item_id'])]).strip()
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
