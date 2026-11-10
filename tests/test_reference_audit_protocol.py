import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('reference_audit', Path(__file__).parents[1]/'scripts/audit_narrowed_reference.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_handoff_protocol_is_read_from_hash_verified_plan(tmp_path):
    (tmp_path/'raw').mkdir()
    path = tmp_path/'raw/certification-plan.json'
    path.write_text(json.dumps({'generation_protocol': {'config_sha256': 'a'*64, 'models': {}}}))
    (tmp_path/'report.json').write_text(json.dumps({'certification_plan_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))
    assert audit.generation_protocol(tmp_path)['config_sha256'] == 'a'*64
    path.write_text('{}')
    with pytest.raises(ValueError, match='plan hash'):
        audit.generation_protocol(tmp_path)


@pytest.mark.parametrize('field', ['charged_calls', 'evidence'])
def test_self_consistent_same_verdict_replacement_is_rejected_by_published_hash(field):
    record = dict(candidate_id='trained-fixture', charged_calls=468,
                  feedback={'evidence_status':'attempted_unresolved'}, evidence={'ts':{'status':'PASS'}})
    record['content_sha256'] = audit.identity(record)
    anchors = {('trained_initial', record['candidate_id']): record['content_sha256']}
    assert audit.verify_result_identity(record, 'trained_initial', anchors)['published_anchor_verified']
    changed = deepcopy(record)
    if field == 'charged_calls':
        changed[field] += 1
    else:
        changed[field]['ts']['extra'] = 'different run, same verdict'
    changed['content_sha256'] = audit.identity({k:v for k,v in changed.items() if k!='content_sha256'})
    assert changed['feedback'] == record['feedback']
    with pytest.raises(ValueError, match='published historical'):
        audit.verify_result_identity(changed, 'trained_initial', anchors)


def test_published_anchor_has_exact_three_original_content_hashes():
    anchors = audit.historical_anchors(Path(__file__).parents[1])
    assert len(anchors) == 3
    assert anchors[('trained_initial','dft-da:R0:joint_generation:p0')] == 'ccf7cbd7892aaf5c37474b80ec63618acd79b9ba4c33fd89bdd49c09c34d8dd1'
    assert anchors[('trained_initial','dft-da:R0:independent_event_then_geometry:p0')] == 'fc22caeb4421943d1d85f0c4c73efd3451ac8e93a231a881e15f112600a8bac1'
