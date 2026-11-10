#!/usr/bin/env python3
"""#184 zero-QC retained-row, analytic and bounded frozen-model diagnostics."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import resource
import socket
import subprocess
import sys
import time
import types

import numpy as np


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: dict) -> None:
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    temporary = path.with_suffix('.writing')
    with temporary.open('w') as stream:
        stream.write(data)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    temporary.replace(path)


def retained_steps(rows: list[dict]) -> dict:
    pairs = {}
    for row in rows:
        key = (row['parent_id'], row['seed'])
        arm = row['arm']
        if arm in pairs.setdefault(key, {}):
            raise ValueError('duplicate retained row')
        pairs[key][arm] = row
    ratios, standard, normalized = [], [], []
    counts = dict(standard_energy_improved=0, normalized_energy_worsened=0,
                  standard_force_improved=0, normalized_force_worsened=0)
    for arms in pairs.values():
        if set(arms) != {'no_op', 'standard_force_step', 'normalized_force_step'}:
            raise ValueError('incomplete matched denominator')
        a, b = arms['standard_force_step'], arms['normalized_force_step']
        if a['input_fingerprint'] != b['input_fingerprint']:
            raise ValueError('unpaired inputs')
        standard.append(a['geometry_displacement_angstrom'])
        normalized.append(b['geometry_displacement_angstrom'])
        ratios.append(normalized[-1] / standard[-1])
        counts['standard_energy_improved'] += a['energy_delta_hartree'] < 0
        counts['normalized_energy_worsened'] += b['energy_delta_hartree'] > 0
        counts['standard_force_improved'] += a['force_rms_reduction'] > 0
        counts['normalized_force_worsened'] += b['force_rms_reduction'] < 0
    return dict(rows=len(rows), pairs=len(pairs), parents=len({p for p, _ in pairs}),
                standard_median_angstrom=float(np.median(standard)),
                normalized_median_angstrom=float(np.median(normalized)),
                ratio_median=float(np.median(ratios)), ratio_range=[min(ratios), max(ratios)],
                **counts, interpretation='Amplitude confounding supports overshoot; no sign-error or TS-success proof.')


def analytic_fixtures() -> list[dict]:
    rows = []
    x = np.array([1., 1.])
    for name, jacobian, mode in [('correct_mode_identity', np.eye(2), [1., 0.]),
                                  ('wrong_mode_identity', np.eye(2), [0., 1.]),
                                  ('correct_mode_reversed_endpoint', np.diag([-1., 1.]), [1., 0.])]:
        y = jacobian @ x
        force = np.array([y[0], -y[1]])
        u = np.asarray(mode)
        direction = force - 2 * u * np.dot(u, force)
        after = jacobian @ (x + .1 * direction)
        rows.append(dict(case=name, before_saddle_distance=float(np.linalg.norm(y)),
                         after_saddle_distance=float(np.linalg.norm(after)),
                         first_unstable_coordinate_before=float(abs(y[0])),
                         first_unstable_coordinate_after=float(abs(after[0])),
                         expected_counterexample=name != 'correct_mode_identity',
                         normalized_unit_rescaling_invariant=bool(np.allclose(
                             direction/np.linalg.norm(direction), 1000*direction/np.linalg.norm(1000*direction))),
                         alpha_zero_equivalent=bool(np.array_equal(jacobian @ (x + 0*direction), y))))
    return rows


def frozen_model(args, cfg, report, start, cpu):
    import torch
    torch.set_num_threads(1)
    source = args.research_source.resolve()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source).decode().strip()
    if commit != cfg['trained']['research_commit'] or subprocess.check_output(['git', 'status', '--porcelain'], cwd=source).strip():
        raise ValueError('research source is not exact and clean')
    packet = json.loads(args.packet.read_text())
    if file_hash(args.packet) != cfg['diagnostic']['worker_packet_sha256']:
        raise ValueError('packet hash mismatch')
    if len(packet['queries']) != 1 or set(packet['queries'][0]) != {'query_id', 'atomic_numbers', 'x_r', 'b_r', 'charge', 'multiplicity'}:
        raise ValueError('inference whitelist mismatch')
    sys.path.insert(0, str(source / 'src'))
    vocabulary = types.ModuleType('xtbflow.m0.batching')
    vocabulary.ELEMENT_INDEX = {1: 1, 6: 2, 7: 3, 8: 4}
    vocabulary.N_ELEMENT_TYPES = 5
    sys.modules[vocabulary.__name__] = vocabulary
    from xtbflow.m0.model import ReactionFlowNet
    from xtbflow.m0.sampler import decode_be
    from xtbflow.v1.interfaces import State, query_from_parents
    from xtbflow.v1.sampler import initial_state, rollout
    from xtbflow.v1.clocks import clock_grid
    from xtbflow.v1.guidance import centered

    path = args.model_root / 'generators/joint_s0/ckpt_60000.pt'
    expected = cfg['trained']['checkpoints']['joint']
    if file_hash(path) != expected['sha256'] or file_hash(path.parent/'status.json') != expected['training_status_sha256']:
        raise ValueError('checkpoint hash mismatch')
    obj = torch.load(path, map_location='cpu', weights_only=True)
    meta = obj['meta']
    model = ReactionFlowNet('joint', **meta['model_cfg'], dual_time=meta['dual_time'])
    model.load_state_dict(obj['ema'])
    model.eval().requires_grad_(False)
    query = query_from_parents(packet['queries'], 'cpu')
    initial = initial_state(query, [0], 0, sampling_seed=0, namespace=cfg['noise']['namespace'], group='common',
                            sigma_b=meta['sigma_b'], sigma_x=meta['sigma_x'])
    # Preserve the historical float32 addressed noise; only subsequent numerics
    # are promoted. This is a diagnostic, not bitwise CPU/GPU reproduction.
    query = replace(query, x_r=query.x_r.double(), b_r=query.b_r.double())
    initial = State(*(getattr(initial, k).double() for k in ('b', 'x', 't_b', 't_x')))
    model.double()
    cap = report['caps']
    journal = dict(forwards=[], jvp_calls=0, backward_calls=0)
    report['model_work'] = journal

    def before(module, values, kwargs):
        if (len(journal['forwards']) >= cap['model_forwards'] or time.monotonic()-start >= cap['wall_seconds']
                or time.process_time()-cpu >= cap['cpu_seconds']):
            raise RuntimeError('frozen diagnostic budget exhausted')
        journal['forwards'].append(dict(sequence=len(journal['forwards'])+1, status='started'))
        save(args.out/'model_calls.json', journal)

    def after(module, values, output):
        if not all(torch.isfinite(v).all() for v in output.values()):
            raise ValueError('nonfinite model output')
        journal['forwards'][-1]['status'] = 'completed'
        save(args.out/'model_calls.json', journal)
    model.register_forward_pre_hook(before, with_kwargs=True)
    model.register_forward_hook(after)
    tb, tx = clock_grid('event_lead2', 50)
    if args.resume_states is not None:
        recovery = json.loads((Path(__file__).resolve().parents[1]/'configs/research/numerical_recovery_20261111.json').read_text())
        if file_hash(args.resume_states/'numerical_diagnostics.json') != recovery['prior_report_sha256']:
            raise ValueError('recovery receipt mismatch')
        states = {}
        for k in (10, 25, 40):
            path = args.resume_states/f'state_{k}.pt'
            if file_hash(path) != recovery['saved_state_sha256'][str(k)]:
                raise ValueError('saved state mismatch')
            # These exact locally produced, hash-pinned files include NumPy
            # clock arrays. No arbitrary checkpoint pickle is admitted.
            saved = torch.load(path, map_location='cpu', weights_only=False)
            states[k] = State(*(torch.as_tensor(saved[name], dtype=torch.float64) for name in ('b','x','t_b','t_x')))
        report['saved_state_sha256'] = recovery['saved_state_sha256']
        report['recovery_of'] = recovery['prior_report_sha256']
    else:
        baseline = rollout(model, query, initial, path='event_lead2', n_steps=50)
        if baseline.failed.any():
            raise ValueError('reconstruction failed')
        states = {k: State(baseline.b_trace[k], baseline.x_trace[k], torch.as_tensor(tb[k:k+1]), torch.as_tensor(tx[k:k+1])) for k in (10,25,40)}
        for k, state in states.items():
            torch.save(dict(b=state.b, x=state.x, t_b=state.t_b, t_x=state.t_x), args.out/f'state_{k}.pt')
        report['saved_state_sha256'] = {str(k): file_hash(args.out/f'state_{k}.pt') for k in states}
    report['response'] = []
    for k, state in states.items():
        d = centered(torch.sin(torch.arange(state.x.numel(), dtype=torch.float64).reshape_as(state.x)), query.atom_mask)
        d = d / torch.linalg.vector_norm(d)
        def values(x):
            out = model(z=query.element_index, atom_mask=query.atom_mask, x_cur=centered(x, query.atom_mask),
                        x_r=query.x_r, b_cur=state.b, b_r=query.b_r, t=state.t_b, t_x=state.t_x)
            return centered(x+(1-state.t_x[:,None,None])*out['x_vel'], query.atom_mask), out['b_vel']
        journal['jvp_calls'] += 1
        journal['backward_calls'] += 2  # torch.autograd.functional.jvp: reverse over reverse
        if journal['jvp_calls'] > cap['jvp_calls'] or journal['backward_calls'] > cap['backward_calls']:
            raise RuntimeError('derivative budget exhausted')
        save(args.out/'model_calls.json', journal)
        _, derivative = torch.autograd.functional.jvp(values, state.x, d, strict=True)
        jx, jb = derivative
        row = dict(step=k, t_b=float(state.t_b[0]), t_x=float(state.t_x[0]),
                   endpoint_gain=float(torch.linalg.vector_norm(jx)),
                   endpoint_direction_cosine=float((jx*d).sum()/torch.linalg.vector_norm(jx)),
                   event_velocity_response_norm=float(torch.linalg.vector_norm(jb)),
                   event_response_symmetry_error=float((jb-jb.transpose(1,2)).abs().max()),
                   event_response_electron_sum=float(jb.sum()),
                   centered_endpoint_derivative_error=float(jx.mean(1).abs().max()), finite_differences=[])
        for epsilon in cfg['state_selection']['fd_scales_angstrom']:
            with torch.no_grad():
                plus, minus = values(state.x+epsilon*d), values(state.x-epsilon*d)
            fd = tuple((a-b)/(2*epsilon) for a,b in zip(plus,minus))
            row['finite_differences'].append(dict(epsilon_angstrom=epsilon,
                endpoint_relative_error=float(torch.linalg.vector_norm(fd[0]-jx)/torch.linalg.vector_norm(jx).clamp_min(1e-15)),
                event_relative_error=float(torch.linalg.vector_norm(fd[1]-jb)/torch.linalg.vector_norm(jb).clamp_min(1e-15))))
        zero = rollout(model, query, state, path='event_lead2', n_steps=50, start=k, end=k+2, alpha=0)
        again = rollout(model, query, state, path='event_lead2', n_steps=50, start=k, end=k+2)
        first = rollout(model, query, state, path='event_lead2', n_steps=50, start=k, end=k+1)
        changed = replace(first.state, x=centered(first.state.x+.001*d, query.atom_mask))
        continued = rollout(model, query, changed, path='event_lead2', n_steps=50, start=k+1, end=k+2)
        row['alpha_zero_bitwise'] = bool(torch.equal(zero.state.b, again.state.b) and torch.equal(zero.state.x, again.state.x))
        row['same_step_event_unchanged'] = bool(torch.equal(first.state.b, changed.b))
        row['next_step_event_delta_norm'] = float(torch.linalg.vector_norm(continued.state.b-zero.state.b))
        row['next_step_geometry_delta_norm_angstrom'] = float(torch.linalg.vector_norm(continued.state.x-zero.state.x))
        for name, item in [('baseline', zero.state), ('perturbed', continued.state)]:
            decoded, _ = decode_be(item.b[0], query.element_index[0], 16)
            row[name+'_decoded'] = None if decoded is None else decoded.tolist()
        report['response'].append(row)
    report.update(research_commit=commit, checkpoint_sha256=expected['sha256'], torch=torch.__version__,
                  dtype='float64 after float32 initial noise', new_generation_candidates=0,
                  diagnostic_reconstruction_forwards=0 if args.resume_states is not None else 50,
                  forbidden_imports=sorted(set(sys.modules)&{'xtbflow.v1.data', 'xtbflow.m0.t1x_data'}))
    if report['forbidden_imports']:
        raise ValueError('forbidden label loader imported')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('qualified-rows', 'research-source', 'model-root', 'packet', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--resume-states', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    config = root/'configs/research/narrowed_scope_20261111.json'
    cfg = json.loads(config.read_text())
    cap = cfg['caps']['S1_S2_frozen_diagnostic_total']
    if args.resume_states is not None:
        cap = json.loads((root/'configs/research/numerical_recovery_20261111.json').read_text())['caps']
    resource.setrlimit(resource.RLIMIT_CPU, (cap['cpu_seconds'],)*2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (cap['max_output_bytes'],)*2)
    start, cpu = time.monotonic(), time.process_time()
    report = dict(schema='xtbflow-narrowed-numerical/v1', host=socket.gethostname(), python=platform.python_version(),
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root).decode().strip(),
                  config_sha256=file_hash(config), driver_sha256=file_hash(Path(__file__)), caps=cap,
                  status='executing', new_calculator_calls=0, training_steps=0, gpu_seconds=0)
    try:
        manifest = json.loads((root/'configs/research/narrowed_assets_20261111.json').read_text())
        expected = next(a['sha256'] for a in manifest['assets'] if a['path'].endswith('qualified_baseline_confirmation_20261109.json'))
        if file_hash(args.qualified_rows) != expected:
            raise ValueError('retained row hash mismatch')
        report['retained_rows_sha256'] = expected
        report['step_audit'] = retained_steps(json.loads(args.qualified_rows.read_text())['rows'])
        report['analytic'] = analytic_fixtures()
        frozen_model(args, cfg, report, start, cpu)
        report['status'] = 'completed'
    except Exception as exc:
        report.update(status='diagnostic_failed', failure_type=type(exc).__name__, failure=str(exc))
    report.update(wall_seconds=time.monotonic()-start, cpu_seconds=time.process_time()-cpu,
                  raw_artifacts=[dict(name=p.name, sha256=file_hash(p), bytes=p.stat().st_size) for p in sorted(args.out.iterdir()) if p.is_file()],
                  claim_limits=['Calibration/numerical response only; no physical direction, chemistry or efficacy result.',
                                'Saved states reconstructed under frozen rules, not recovered archived tensors.',
                                'No automatic J-transpose saddle correction or retuning on nine exposed parents.'])
    save(args.out/'numerical_diagnostics.json', report)
    print(report['status'])
    return int(report['status'] != 'completed')


if __name__ == '__main__':
    raise SystemExit(main())
