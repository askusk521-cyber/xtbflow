#!/usr/bin/env python3
"""Read retained artifacts and project stage evidence; cannot run a backend."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import resource
import socket
import subprocess
import time

import numpy as np

from xtbflow.governance.generation_bridge import certification_candidate, legacy_query, validate_journal
from xtbflow.validation.certification import CertificationCandidate, feedback, identity


HISTORICAL_REPORT_COMMIT = '5f5304bcded99f2711ce21a090392cc2d6db8973'
HISTORICAL_REPORT_SHA256 = 'f3cc2274aa5b3e1d47a8bb0beb4c32f04dbf99948f5b4f4cb6455b773aba8ff0'
HISTORICAL_REPORT_PATH = 'docs/evidence/narrowed_scope_20261111/anchors/live_feedback_generation_reference_terminal_20261108.json'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generation_protocol(root: Path) -> dict:
    """Use the archived handoff protocol, which is not the worker config."""
    plan = root/'raw/certification-plan.json'
    report = json.loads((root/'report.json').read_text())
    if digest(plan) != report['certification_plan_sha256']:
        raise ValueError('generation plan hash mismatch')
    return json.loads(plan.read_text())['generation_protocol']


def historical_anchors(root: Path) -> dict[tuple[str, str], str]:
    """Anchor to the immutable published report, not a result's own assertions."""
    path = root / HISTORICAL_REPORT_PATH
    if digest(path) != HISTORICAL_REPORT_SHA256:
        raise ValueError('published terminal report byte hash mismatch')
    stages = json.loads(path.read_text())['stages']
    pairs = [('trained_initial', row) for row in stages['4101_original_reference']['candidate_outcomes']]
    pairs.append(('trained_feedback', stages['4106_new_candidate_reference']['candidate_outcome']))
    return {(label, row['candidate_id']): row['result_content_sha256'] for label, row in pairs}


def verify_result_identity(record: dict, label: str, anchors: dict) -> dict:
    actual = identity({k: v for k, v in record.items() if k != 'content_sha256'})
    if record.get('content_sha256') != actual:
        raise ValueError('result canonical content identity mismatch')
    expected = anchors.get((label, record['candidate_id']))
    if label in {'trained_initial', 'trained_feedback'} and (expected is None or expected != actual):
        raise ValueError('result differs from published historical content identity')
    return dict(result_content_sha256=actual, published_result_content_sha256=expected,
                published_anchor_verified=None if expected is None else True)


def geometry_summary(raw: dict) -> dict:
    ts, ends, irc = raw.get('ts', {}), raw.get('endpoints', []), raw.get('irc', {})
    frequencies = ts.get('frequencies_cm1', [])
    result = dict(ts_status=ts.get('status'), ts_frequencies_cm1=frequencies,
                  endpoint_statuses=[e.get('status') for e in ends],
                  irc_status=irc.get('status'), irc_frames=irc.get('n_frames'),
                  endpoint_change_interpretation='Distances/mode participation only; no torsion or chemical mechanism inferred.')
    mode = np.asarray(ts.get('reaction_mode', []))
    if mode.size:
        amplitude = np.linalg.norm(mode, axis=1)
        result['mode_atom_amplitude'] = amplitude.tolist()
        result['mode_convention'] = ts.get('reaction_mode_convention')
    if len(ends) == 2 and all('x' in e for e in ends):
        left, right = (np.asarray(e['x']) for e in ends)
        delta = np.linalg.norm(left[:, None]-left[None], axis=-1)-np.linalg.norm(right[:, None]-right[None], axis=-1)
        result['endpoint_pair_distance_change_rms_angstrom'] = float(np.sqrt(np.mean(delta**2)))
        result['endpoint_pair_distance_change_max_angstrom'] = float(np.max(np.abs(delta)))
        a, b = left-left.mean(0), right-right.mean(0)
        u, _, vh = np.linalg.svd(a.T@b)
        rotation = u@np.diag([1, 1, np.linalg.det(u@vh)])@vh
        result['proper_aligned_endpoint_rmsd_angstrom'] = float(np.sqrt(np.mean(np.sum((a@rotation-b)**2, axis=1))))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--service', nargs=2, action='append', metavar=('LABEL','DIRECTORY'), required=True)
    parser.add_argument('--generation-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('output exists')
    root = Path(__file__).resolve().parents[1]
    resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16*1024**2,)*2)
    start, cpu, bytes_read = time.monotonic(), time.process_time(), 0
    report = dict(schema='xtbflow-narrowed-reference-audit/v1', host=socket.gethostname(),
                  python=platform.python_version(), numpy=np.__version__,
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root).decode().strip(),
                  driver_sha256=digest(Path(__file__)), records=[], assets=[], new_calculator_calls=0,
                  new_model_forwards=0, training_steps=0, gpu_seconds=0, status='executing',
                  caps=dict(wall_seconds=600,cpu_seconds=600,input_bytes=256*1024**2,output_bytes=16*1024**2,retries=0,concurrency=1))
    def checked(path, name, expected=None):
        nonlocal bytes_read
        if path.is_symlink():
            raise ValueError('symlink outside audit contract')
        bytes_read += path.stat().st_size
        if bytes_read > 256*1024**2 or time.monotonic()-start > 600:
            raise RuntimeError('audit budget exhausted')
        value = digest(path)
        if expected is not None and value != expected:
            raise ValueError('raw artifact hash mismatch: '+name)
        report['assets'].append(dict(asset=name, bytes=path.stat().st_size, sha256=value))
        return value
    try:
        if sorted(label for label, _ in args.service) != ['live_rule', 'trained_feedback', 'trained_initial']:
            raise ValueError('exactly the three frozen service groups are required')
        anchors = historical_anchors(root)
        checked(root/HISTORICAL_REPORT_PATH, 'published_terminal_anchor', HISTORICAL_REPORT_SHA256)
        report['historical_anchor'] = dict(commit=HISTORICAL_REPORT_COMMIT, sha256=HISTORICAL_REPORT_SHA256,
                                          original_path='docs/evidence/live_feedback_generation_reference_terminal_20261108.json')
        seen = set()
        requests = {}
        for label, folder in args.service:
            paths = sorted(Path(folder).glob('*/attempt-*/result.json'))
            if not paths:
                raise ValueError('missing service records: '+label)
            for path in paths:
                record_hash = checked(path, f'{label}/{path.parent.parent.name}/{path.parent.name}/result.json')
                record = json.loads(path.read_text())
                identity_check = verify_result_identity(record, label, anchors)
                key = (label, record['candidate_id'])
                if key in seen:
                    raise ValueError('duplicate frozen candidate result')
                seen.add(key)
                raw = record['evidence']
                request_path = path.parent/'request.json'
                checked(request_path, f'{label}/{record["candidate_id"]}/request.json', record['request_sha256'])
                req = json.loads(request_path.read_text())
                requests[record['candidate_id']] = req['candidate']
                for item in record['artifacts']:
                    relative = Path(item['path'])
                    if relative.is_absolute() or '..' in relative.parts:
                        raise ValueError('invalid artifact path')
                    checked(path.parent/relative, f'{label}/{record["candidate_id"]}/{relative}', item['sha256'])
                current = feedback(raw)
                if current['evidence_status'] != record['feedback']['evidence_status']:
                    raise ValueError('projection unexpectedly changed historical verdict')
                report['records'].append(dict(label=label,candidate_id=record['candidate_id'],
                    original_result_sha256=record_hash, original_evidence_status=record['feedback']['evidence_status'],
                    execution_status=record['execution_status'], historical_charged_calls=record['charged_calls'],
                    original_reason=record['feedback'].get('reason'),
                    projection={k:current[k] for k in ('projection_schema','terminal_stage','terminal_stage_status','stage_reason','raw_evidence_sha256','stages')},
                    geometry=geometry_summary(raw), historical_resources=record['resources'],
                    unchanged_during_audit=digest(path)==record_hash, **identity_check))
        if not set(anchors).issubset(seen):
            raise ValueError('missing published trained result')
        gen = args.generation_root
        for rel in ('raw/reactants.json','raw/generation.json','raw/worker-packet.json','raw/calls.json',
                    'raw/certification-plan.json','report.json'):
            checked(gen/rel, 'trained_generation/'+rel)
        query = json.loads((gen/'raw/reactants.json').read_text())['queries'][0]
        outputs = json.loads((gen/'raw/generation.json').read_text())
        packet = json.loads((gen/'raw/worker-packet.json').read_text())
        if legacy_query(query) != packet['queries'][0]:
            raise ValueError('inference-visible packet differs from mapped query')
        bridge = []
        protocol = generation_protocol(gen)
        for row in outputs['candidates']:
            candidate = certification_candidate(query,row,generation_protocol=protocol)
            original = CertificationCandidate(**requests[candidate.candidate_id])
            if identity(candidate.content()) != identity(original.content()):
                raise ValueError('cached trained handoff differs from archived reference input')
            bridge.append(dict(candidate_id=candidate.candidate_id, method=row['method'],
                               candidate_content_sha256=identity(candidate.content()),
                               geometry_sha256=identity(candidate.coordinates_angstrom),
                               decoded_BE_sha256=identity(candidate.planned_product_be),
                               atoms=len(candidate.atom_ids), atom_ids_sha256=identity(candidate.atom_ids),
                               exact_cached_reference_input=True, historical_forwards=row['forward_evaluations']))
        report['trained_bridge'] = dict(candidates=bridge, models=outputs['models'],
            forward_journal=validate_journal(json.loads((gen/'raw/calls.json').read_text()),cap=150),
            scope='Cached trained generation to archived reference input; no new live trained-physics loop or efficacy success.',
            permitted_fields=list(legacy_query(query)), reference_only_labels_visible=False)
        report['status'] = 'verified'
    except Exception as exc:
        report.update(status='audit_failed',failure_type=type(exc).__name__,failure=str(exc))
    report.update(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,bytes_read=bytes_read,
                  claim_limits=['Read-only projection and known exposed calibration replay, not new certification.',
                                'Original result/hash/cost/verdict is retained; stage reasons are v2 derived fields.',
                                'No method advantage from primitive-count differences or one known positive control.'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')
    print(report['status'])
    return int(report['status']!='verified')


if __name__ == '__main__':
    raise SystemExit(main())
