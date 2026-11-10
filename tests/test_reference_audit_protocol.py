import hashlib
import importlib.util
import json
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
