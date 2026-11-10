#!/usr/bin/env python3
"""S0 read-only n2 audit; hashes/metadata only, no forward or calculator work."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import resource
import socket
import subprocess
import time


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--generation-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.out.exists():
        raise FileExistsError('audit output exists')
    config_path = root / 'configs/research/narrowed_scope_20261111.json'
    cfg = json.loads(config_path.read_text())
    cap = cfg['caps']['S0']
    resource.setrlimit(resource.RLIMIT_CPU, (cap['cpu_seconds'],) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (cap['max_output_bytes'],) * 2)
    start, cpu = time.monotonic(), time.process_time()
    result = dict(schema='xtbflow-narrowed-availability/v1', host=socket.gethostname(),
                  python=platform.python_version(), config_sha256=digest(config_path),
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip(),
                  driver_sha256=digest(Path(__file__)), caps=cap, assets=[], models={},
                  new_model_forwards=0, new_training_steps=0, new_calculator_calls=0,
                  status='executing', claim_limits=['Availability and input identity, not scientific qualification.'])
    used_bytes = 0

    def checked(path: Path, name: str, expected: str | None = None) -> None:
        nonlocal used_bytes
        used_bytes += path.stat().st_size
        if used_bytes > cap['max_input_bytes'] or time.monotonic() - start > cap['wall_seconds']:
            raise RuntimeError('read budget exhausted')
        value = digest(path)
        result['assets'].append(dict(asset=name, bytes=path.stat().st_size, sha256=value,
                                     matches_expected=None if expected is None else value == expected))
        if expected is not None and value != expected:
            raise ValueError(f'identity mismatch: {name}')

    try:
        import torch
        result['torch'] = torch.__version__
        for role, expected in cfg['trained']['checkpoints'].items():
            path = args.model_root / 'generators' / f'{role}_s0' / 'ckpt_60000.pt'
            checked(path, f'{role}_s0/ckpt_60000.pt', expected['sha256'])
            checked(path.parent / 'status.json', f'{role}_s0/status.json', expected['training_status_sha256'])
            obj = torch.load(path, map_location='cpu', weights_only=True)
            meta = obj['meta']
            if obj['step'] != 60000 or meta['role'] != role or meta['seed'] != 0:
                raise ValueError('checkpoint metadata mismatch')
            result['models'][role] = {key: meta[key] for key in
                ('role', 'seed', 'model_cfg', 'dual_time', 'sigma_b', 'sigma_x', 'source_commit', 'split_hash')}
            result['models'][role]['training_parent_ids_sha256'] = hashlib.sha256(
                json.dumps(meta['training_parent_ids'], sort_keys=True).encode()).hexdigest()
        for rel, expected in [('raw/reactants.json', cfg['diagnostic']['reactant_packet_sha256']),
                              ('raw/worker-packet.json', cfg['diagnostic']['worker_packet_sha256']),
                              ('raw/generation.json', None), ('raw/calls.json', None),
                              ('raw/certification-plan.json', None), ('report.json', None), ('ledger.json', None)]:
            checked(args.generation_root / rel, rel, expected)
        query = json.loads((args.generation_root / 'raw/reactants.json').read_text())['queries'][0]
        if query['parent_id'] != cfg['diagnostic']['parent_id'] or len(query['atomic_numbers']) != 16:
            raise ValueError('frozen parent mismatch')
        if (query['charge'], query['multiplicity']) != (0, 1) or len(set(query['atom_ids'])) != 16:
            raise ValueError('frozen electronic state or mapping mismatch')
        result['input'] = {k: query[k] for k in ('query_id', 'parent_id', 'charge', 'multiplicity', 'environment', 'provenance', 'atom_ids')}
        result['input']['automorphism_count'] = len(query['reactant_automorphisms'])
        packet = json.loads((args.generation_root / 'raw/worker-packet.json').read_text())
        if any(set(q) != {'query_id', 'atomic_numbers', 'x_r', 'b_r', 'charge', 'multiplicity'} for q in packet['queries']):
            raise ValueError('inference field whitelist mismatch')
        result['status'] = 'verified'
    except Exception as exc:
        result.update(status='audit_failed', failure_type=type(exc).__name__)
    result.update(bytes_read=used_bytes, wall_seconds=time.monotonic()-start,
                  cpu_seconds=time.process_time()-cpu, gpu_seconds=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(result['status'])
    return 0 if result['status'] == 'verified' else 1


if __name__ == '__main__':
    raise SystemExit(main())
