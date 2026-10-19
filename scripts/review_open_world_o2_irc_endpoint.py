"""Supplementary O2 diagnostic: minimise the two ends of the IRC_LIMIT chains.

Owner approval 2026-10-19; rules in configs/review/open_world_o2_irc_endpoint.json,
pushed before any calculation. Two O2 candidate chains (parents 4f55..., 7dc6...)
passed TS Opt+Freq but stopped with IRC_LIMIT: the forward IRC converged, the
backward direction used all 300 iterations. Under v1b-h-gpu4pyscf-3 that is
PROTOCOL_FAILURE and the O2 results keep it. This script asks a narrower
question: minimising the first and last frame of the stored trajectory with the
unchanged qc_protocol.minimum_stage, do the two minima give the predicted event
under the unchanged qc_protocol.endpoint_identity?

geomeTRIC 1.1.1 writes irc/irc_irc.xyz for irc_direction='both' as one path: at
the direction switch it reverses the forward frames, so the file runs forward
end -> TS (the one 'Iteration 0' frame) -> backward end. Its first and last
frames are the two frames certification_chain would have minimised had the IRC
converged. Coordinates are Angstrom, energies electronic Hartree, neutral singlet.

  run       --run-root R --item ITEM                 inside Slurm, clean clone
  submit    --run-root R --source S                  login node; serial sbatch, single use
  summarize --run-root R --out results_irc_endpoint.json   CPU only, deterministic
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_open_world_dft as v1  # noqa: E402
from review_open_world_dft_v2 import cuda_gpu_info  # noqa: E402
from review_open_world_dispatch_v2 import classify, command, seconds, wait_terminal  # noqa: E402
from review_open_world_dispatch_v4 import PYTHON, precheck_command  # noqa: E402
from review_open_world_o2_analyze import stage_summary  # noqa: E402
from xtbflow.v1.data import file_hash, write_json  # noqa: E402

CONFIG = Path('configs/review/open_world_o2_irc_endpoint.json')
OUT = 'o2-irc-endpoint'
SYMBOLS = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}
COMMENT = re.compile(r'Iteration (\d+) Energy\s+(-?\d+\.\d+)')
MINIMUM_OK = ('MINIMUM_PASS', 'MINIMUM_GRAY_ZONE')


def read_frames(path):
    """Frames of a geomeTRIC xyz trajectory: iteration, comment energy (Hartree), symbols, x (Angstrom)."""
    lines, frames, i = Path(path).read_text().splitlines(), [], 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i])
        match = COMMENT.search(lines[i + 1])
        if not match:
            raise ValueError('Unrecognised frame comment: ' + lines[i + 1])
        atoms = [line.split() for line in lines[i + 2:i + 2 + n]]
        frames.append(dict(iteration=int(match.group(1)), energy_hartree=float(match.group(2)),
                           symbols=[a[0] for a in atoms], x=np.array([[float(v) for v in a[1:4]] for a in atoms])))
        i += 2 + n
    return frames


def irc_ends(frames, z, ts, checks):
    """First and last frames after the frozen frame checks; raises before any QC call."""
    symbols = [SYMBOLS[int(a)] for a in z]
    if any(f['symbols'] != symbols for f in frames):
        raise ValueError('Frame symbols differ from the plan item')
    zero = [k for k, f in enumerate(frames) if f['iteration'] == 0]
    if len(zero) != 1:
        raise ValueError(f'Expected one TS frame, found {len(zero)}')
    t = frames[zero[0]]
    if abs(t['energy_hartree'] - ts['energy_hartree']) > checks['ts_energy_abs_hartree']:
        raise ValueError('TS frame energy differs from the O2 TS')
    if np.abs(t['x'] - np.asarray(ts['x_ts'])).max() > checks['ts_coordinates_max_abs_angstrom']:
        raise ValueError('TS frame coordinates differ from the O2 TS')
    first, last = frames[0], frames[-1]
    info = dict(n_frames=len(frames), ts_index=zero[0],
                first=dict(iteration=first['iteration'], energy_hartree=first['energy_hartree']),
                last=dict(iteration=last['iteration'], energy_hartree=last['energy_hartree']))
    return first['x'], last['x'], info


def endpoint_diagnostic(qc, z, x_first, x_last, item, ts_energy, method, meter, work):
    """certification_chain lines 326-336 (unchanged qc_protocol calls) on two given frames."""
    out = {}
    try:
        ends = [qc.minimum_stage(z, x, method, meter, Path(work) / k, f'end_{k}')
                for k, x in (('first', x_first), ('last', x_last))]
        out['endpoints'] = ends
        if any(e['status'] not in MINIMUM_OK for e in ends):
            out['diagnostic_status'] = 'ENDPOINT_FAILURE'
            return out
        ident = qc.endpoint_identity(z, item['b_r'], item['perms'], item['b_predicted'],
                                     np.asarray(ends[0]['x']), np.asarray(ends[1]['x']))
        out['identity'], out['diagnostic_status'] = ident, ident['status']
        if 'reactant_end' in ident:
            out['local_barrier_kcal'] = (ts_energy - ends[ident['reactant_end']]['energy_hartree']) * qc.HARTREE_TO_KCAL
    except qc.SCFError as err:
        out['diagnostic_status'], out['failure'] = 'PROTOCOL_FAILURE', str(err)
    return out


def checked_inputs(root, cfg, item_id):
    """Plan item, O2 verdict and IRC ends, after every frozen hash and frame check."""
    plan_path = root / 'o2-plan.json'
    if file_hash(plan_path) != cfg['plan_sha256']:
        raise ValueError('o2-plan.json hash mismatch')
    plan = json.loads(plan_path.read_text())
    entry = next(e for e in cfg['items'] if e['item_id'] == item_id)
    item = next(i for i in plan['items'] if i['item_id'] == item_id)
    src = root / 'o2-run' / f"{item['kind']}_{item_id}"
    files = {'verdict.json': entry['o2_verdict_sha256'], 'irc/irc_irc.xyz': entry['irc_xyz_sha256']}
    for name, expected in files.items():
        if file_hash(src / name) != expected:
            raise ValueError('Input hash mismatch: ' + str(src / name))
    verdict = json.loads((src / 'verdict.json').read_text())
    if verdict['ts']['status'] != 'TS_OPTFREQ_PASS' or verdict['irc']['status'] != 'IRC_LIMIT':
        raise ValueError('Item is not a TS-pass IRC_LIMIT chain')
    x_first, x_last, info = irc_ends(read_frames(src / 'irc/irc_irc.xyz'), item['z'], verdict['ts'], cfg['frame_checks'])
    sha = {str(plan_path): cfg['plan_sha256'], **{str(src / k): v for k, v in files.items()}}
    return plan, item, verdict, x_first, x_last, info, sha


def run(a):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU execution requires Slurm')
    state = v1.source_state()
    from xtbflow.v1 import qc_protocol as qc
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    if qc.PROTOCOL_VERSION != cfg['protocol_version']:
        raise ValueError('Unexpected certification protocol')
    root = a.run_root.resolve()
    plan, item, verdict, x_first, x_last, info, sha = checked_inputs(root, cfg, a.item)
    frozen = next(v for k, v in plan['input_sha256'].items() if k.endswith('src/xtbflow/v1/qc_protocol.py'))
    if file_hash('src/xtbflow/v1/qc_protocol.py') != frozen:
        raise ValueError('qc_protocol.py differs from the O2 plan')
    work = root / OUT / f"{item['kind']}_{item['item_id']}"
    work.mkdir(parents=True, exist_ok=False)  # one attempt per item
    started = time.time()
    v1.dump(work / 'manifest.json', dict(**state, config_sha256=file_hash(CONFIG), input_sha256=sha,
                                         slurm_job_id=os.environ['SLURM_JOB_ID'], protocol_version=qc.PROTOCOL_VERSION,
                                         started_unix=started, item_id=item['item_id'], parent_id=item['parent_id'],
                                         irc_frames=info))
    meter, t0 = qc.Meter(), time.perf_counter()
    try:
        import geometric
        import gpu4pyscf
        import pyscf
        v1.dump(work / 'environment.json', dict(gpu=cuda_gpu_info(), gpu4pyscf=gpu4pyscf.__version__,
                                                pyscf=pyscf.__version__, geometric=geometric.__version__,
                                                slurm_job_id=os.environ['SLURM_JOB_ID']))
        z, method = np.asarray(item['z']), qc.Method(device='gpu')
        out = dict(schema='xtbflow-open-world-o2-irc-endpoint-verdict/1', supplementary=True,
                   protocol_version=qc.PROTOCOL_VERSION, method=method.as_dict(), n_atoms=len(z),
                   item_id=item['item_id'], parent_id=item['parent_id'], o2_strict_status=verdict['strict_status'],
                   o2_irc_status=verdict['irc']['status'], irc_frames=info)
        out.update(endpoint_diagnostic(qc, z, x_first, x_last, item, verdict['ts']['energy_hartree'], method, meter, work))
        qc.finish(out, meter, t0, work)
    except Exception as exc:
        v1.dump(work / 'unexpected_failure.json', dict(type=type(exc).__name__, message=str(exc),
                                                      stage_timings=meter.rows, wall_s=time.perf_counter() - t0))
        raise
    finally:
        v1.dump(work / 'execution.json', dict(started_unix=started, ended_unix=time.time(),
                                              wall_s=time.perf_counter() - t0, slurm_job_id=os.environ['SLURM_JOB_ID']))
    print(json.dumps(dict(item=a.item, diagnostic_status=out['diagnostic_status'])), flush=True)


def irc_limit_items(root, plan):
    """Item ids of every O2 chain whose verdict reports IRC_LIMIT (the frozen selection rule)."""
    found = []
    for item in plan['items']:
        verdict = root / 'o2-run' / f"{item['kind']}_{item['item_id']}" / 'verdict.json'
        if item['kind'] != 'anchor' and verdict.exists():
            if (json.loads(verdict.read_text()).get('irc') or {}).get('status') == 'IRC_LIMIT':
                found.append(item['item_id'])
    return found


def item_wrap(source, root, item_id):
    """Shell command sbatch runs for one item."""
    return ('cd ' + shlex.quote(str(source)) + '; export PYTHONPATH=src:vendor/mechai_reusable OMP_NUM_THREADS=8; '
            + PYTHON + ' scripts/review_open_world_o2_irc_endpoint.py run --run-root ' + shlex.quote(str(root))
            + ' --item ' + item_id)


def submit(a):
    root, source = a.run_root.resolve(), a.source.resolve()
    cfg = json.loads((source / CONFIG).read_text(encoding='utf-8'))
    out = root / OUT
    ledger_path = out / 'ledger.json'
    if ledger_path.exists():
        raise FileExistsError('Single use; inspect ' + str(ledger_path))
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=source, text=True).strip():
        raise RuntimeError('Execution clone is dirty')
    plan = json.loads((root / 'o2-plan.json').read_text())
    wanted = [e['item_id'] for e in cfg['items']]
    if sorted(irc_limit_items(root, plan)) != sorted(wanted):
        raise ValueError('IRC_LIMIT items in the run root differ from the frozen list')
    out.mkdir(exist_ok=True)
    gpu = subprocess.run(precheck_command(source), capture_output=True, text=True)
    (out / 'gpu_precheck.txt').write_text(gpu.stdout + gpu.stderr)
    if gpu.returncode or 'source=cuda-runtime' not in gpu.stdout:
        raise RuntimeError('CUDA precheck failed inside Slurm; inspect gpu_precheck.txt')
    execution, ledger, used = cfg['execution'], [], 0

    def save():
        ledger_path.write_text(json.dumps(ledger, indent=2) + '\n')

    for entry in cfg['items']:
        if used + execution['reserve_s'] > execution['budget_s']:
            raise RuntimeError('Reservation would exceed the supplementary GPU budget')
        waiting = time.monotonic()
        while command(['squeue', '-h', '-u', 'lhshen', '-o', '%i']).strip():
            if time.monotonic() - waiting > 7200:
                raise RuntimeError('Queue conflict persisted two hours')
            time.sleep(10)
        job = command(['sbatch', '--parsable', '--partition=main', '--gres=gpu:pro6000:1', '--cpus-per-task=8',
                       '--mem=64G', '--time=01:00:00', '--job-name=fu-ircend',
                       '--output=' + str(out / 'ircend-%j.log'), '--wrap',
                       item_wrap(source, root, entry['item_id'])]).strip()
        ledger.append(dict(item_id=entry['item_id'], kind=entry['kind'], parent_id=entry['parent_id'], job_id=job,
                           outcome='SUBMITTED', submitted_unix=time.time()))
        save()
        state = wait_terminal(job, out)
        runtime = seconds(state['RunTime'])
        used += runtime
        work = out / f"{entry['kind']}_{entry['item_id']}"
        outcome, detail, stop = classify(state, work)
        if outcome == 'VERDICT':
            detail = json.loads((work / 'verdict.json').read_text())['diagnostic_status']
        ledger[-1].update(outcome=outcome, detail=detail, slurm_state=state['JobState'],
                          exit_code=state.get('ExitCode'), runtime_s=runtime, cumulative_gpu_s=used)
        save()
        print(json.dumps(ledger[-1]), flush=True)
        if stop:
            raise RuntimeError(f'{outcome} for job {job}; stop and inspect')
    print('Both items reached terminal states; summarize next.', flush=True)


def kabsch_rmsd(a, b):
    """RMSD (Angstrom) after optimal rotation; same atom order, no permutation."""
    a = np.asarray(a, dtype=float) - np.mean(a, axis=0)
    b = np.asarray(b, dtype=float) - np.mean(b, axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return float(np.sqrt(np.mean(np.sum((a @ r.T - b) ** 2, axis=1))))


def lowest(freq, n=3):
    return sorted(freq)[:n]


def summarize(root, config=CONFIG):
    cfg = json.loads(Path(config).read_text(encoding='utf-8'))
    k = cfg['reported']['hartree_to_kcal']
    ledger_path = root / OUT / 'ledger.json'
    ledger = {r['item_id']: r for r in json.loads(ledger_path.read_text())}
    sha, rows = {str(ledger_path): file_hash(ledger_path)}, []
    for entry in cfg['items']:
        plan, item, verdict, x_first, x_last, info, input_sha = checked_inputs(root, cfg, entry['item_id'])
        sha.update(input_sha)
        by_kind = {i['kind']: i for i in plan['items'] if i['parent_id'] == item['parent_id']}
        o2 = {}
        for kind in ('anchor', 'reference'):
            path = root / 'o2-run' / f"{kind}_{by_kind[kind]['item_id']}" / 'verdict.json'
            sha[str(path)] = file_hash(path)
            o2[kind] = json.loads(path.read_text())
        e_ts = verdict['ts']['energy_hartree']
        ref_ok = o2['reference'].get('strict_status') == 'STRICT_JOINT_GRAPH_VALID'
        anchor_ok = o2['anchor'].get('status') == 'MINIMUM_PASS'
        work = root / OUT / f"{item['kind']}_{item['item_id']}"
        record = None
        for name in ('verdict.json', 'unexpected_failure.json'):
            if (work / name).exists():
                sha[str(work / name)] = file_hash(work / name)
                record = json.loads((work / name).read_text())
                status = record['diagnostic_status'] if name == 'verdict.json' else \
                    f"CHAIN_EXCEPTION:{record['type']}:{record['message']}"
                break
        else:
            status = 'MISSING'
        ends = []
        for label, x0, start in (('first', x_first, info['first']), ('last', x_last, info['last'])):
            end = dict(label=label, frame_iteration=start['iteration'], frame_energy_hartree=start['energy_hartree'])
            got = ((record or {}).get('endpoints') or [None, None])[0 if label == 'first' else 1]
            if got is not None:
                end.update(status=got['status'], retried='first_attempt' in got)
                if 'x' in got:
                    end.update(energy_hartree=got['energy_hartree'],
                               drop_from_frame_kcal=(start['energy_hartree'] - got['energy_hartree']) * k,
                               rmsd_frame_to_minimum_angstrom=kabsch_rmsd(x0, got['x']),
                               lowest_frequencies_cm1=lowest(got['frequencies_cm1']))
            ends.append(end)
        delta = (e_ts - o2['reference']['ts']['energy_hartree']) * k if ref_ok else None
        rows.append(dict(
            item_id=item['item_id'], parent_id=item['parent_id'], n_atoms=len(item['z']),
            delta_ea_xtb_kcal=item['delta_ea_xtb_kcal'], o2_strict_status=verdict['strict_status'],
            diagnostic_status=status, passes=status == cfg['classification']['pass'],
            identity=(record or {}).get('identity'), local_barrier_kcal=(record or {}).get('local_barrier_kcal'),
            irc_frames=info, ends=ends, reference_status=o2['reference'].get('strict_status'),
            ts_energy_difference_vs_reference_kcal=delta,
            candidate_ts_vs_anchor_kcal=(e_ts - o2['anchor']['energy_hartree']) * k if anchor_ok else None,
            reference_ts_vs_anchor_kcal=(o2['reference']['ts']['energy_hartree'] - o2['anchor']['energy_hartree']) * k
            if anchor_ok and ref_ok else None,
            ts_rmsd_xtb_to_dft_angstrom=kabsch_rmsd(item['x_start'], verdict['ts']['x_ts']),
            ts_imaginary_cm1=min(verdict['ts']['frequencies_cm1']),
            slurm=ledger.get(item['item_id']), wall_s=(record or {}).get('wall_s'),
            **stage_summary((record or {}).get('stage_timings', []))))
    used = sum(r.get('runtime_s', 0) for r in ledger.values())
    return dict(schema='xtbflow-open-world-o2-irc-endpoint-results/1', exploratory=True, supplementary=True,
                config_sha256=file_hash(config), protocol_version=cfg['protocol_version'], rows=rows,
                gpu=dict(total_runtime_s=used, budget_s=cfg['execution']['budget_s'],
                         within_budget=used <= cfg['execution']['budget_s']),
                input_sha256=dict(sorted(sha.items())))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--run-root', type=Path, required=True)
    r.add_argument('--item', required=True)
    s = sub.add_parser('submit')
    s.add_argument('--run-root', type=Path, required=True)
    s.add_argument('--source', type=Path, required=True, help='clean execution clone')
    m = sub.add_parser('summarize')
    m.add_argument('--run-root', type=Path, required=True)
    m.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.cmd == 'run':
        run(a)
    elif a.cmd == 'submit':
        submit(a)
    else:
        write_json(a.out, summarize(a.run_root.resolve()))


if __name__ == '__main__':
    main()
