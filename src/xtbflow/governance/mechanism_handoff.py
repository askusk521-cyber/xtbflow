"""Hash-pinned new intervened candidate handoff; cannot read legacy certification caches."""
from pathlib import Path
import json

from xtbflow.validation.certification import CertificationCandidate, file_hash, identity


def load_mechanism_candidates(folder: Path, config_path: Path, expected_report_sha256: str):
    if file_hash(folder/'report.json')!=expected_report_sha256:
        raise ValueError('generation_report_identity_mismatch')
    cfg=json.loads(config_path.read_text());report=json.loads((folder/'report.json').read_text())
    if (report['status']!='generation_completed' or report['reference_gate']!='eligible'
            or report['settlement']!='settled' or report['termination']!='confirmed'
            or report['work']['unresolved_started']):
        raise ValueError('generation_or_accounting_gate_closed')
    if report['config_sha256']!=file_hash(config_path):raise ValueError('generation_configuration_mismatch')
    if [r['arm'] for r in report['candidates']]!=cfg['arms']:raise ValueError('arm_denominator_mismatch')
    if file_hash(folder/'plan.json')!=report['candidate_plan_sha256']:raise ValueError('candidate_plan_identity_mismatch')
    plan=json.loads((folder/'plan.json').read_text())
    if plan['source_commit']!=report['source_commit'] or plan['config_sha256']!=report['config_sha256']:
        raise ValueError('candidate_source_mismatch')
    expected=[r for r in report['candidates'] if r['status']=='generated']
    handoff=report['new_candidate_handoff']
    if len(plan['candidates'])!=len(expected) or len(handoff)!=len(expected):raise ValueError('candidate_denominator_mismatch')
    rows=[]
    for item,summary,check in zip(plan['candidates'],expected,handoff):
        candidate=CertificationCandidate(**item['candidate'])
        sha=identity(candidate.content())
        if (item['arm']!=summary['arm'] or check['arm']!=summary['arm']
                or item['candidate_sha256']!=sha or summary['candidate_sha256']!=sha or check['candidate_sha256']!=sha
                or not check['cache_hit'] or check['new_replay_calls']!=0
                or candidate.provenance['generation_protocol_sha256']!=identity(plan['protocol'])):
            raise ValueError('new_intervention_candidate_handoff_mismatch')
        rows.append((item['arm'],candidate))
    return report,rows
