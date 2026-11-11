"""Stage-local failure projection changes no raw evidence or old artifact."""
from copy import deepcopy

import pytest

from xtbflow.validation.certification import feedback, stage_diagnostics


@pytest.mark.parametrize('status,stage', [
    ('TS_OPT_LIMIT', 'ts_optimization'), ('TS_GRADIENT_LIMIT', 'ts_gradient'),
    ('MULTIPLE_NEGATIVE_MODES', 'ts_modes'), ('FREQUENCY_GRAY_AT_CAP', 'ts_modes')])
def test_ts_failures_survive_feedback(status, stage):
    raw = dict(ts=dict(status=status), strict_status='PROTOCOL_FAILURE')
    before = deepcopy(raw)
    out = feedback(raw)
    assert out['evidence_status'] == 'attempted_unresolved'
    assert out['reason'] == status
    assert out['terminal_stage'] == stage
    assert raw == before


@pytest.mark.parametrize('tail,stage,reason', [
    ({'irc': {'status': 'IRC_LIMIT'}}, 'irc', 'IRC_LIMIT'),
    ({'irc': {'status': 'IRC_COMPLETE'}, 'endpoints': [{'status': 'MIN_OPT_LIMIT'}]}, 'endpoint_0', 'MIN_OPT_LIMIT'),
    ({'irc': {'status': 'IRC_COMPLETE'}, 'endpoints': [{'status': 'MINIMUM_HAS_NEGATIVE_MODE'}]}, 'endpoint_0', 'MINIMUM_HAS_NEGATIVE_MODE'),
    ({'identity': {'status': 'STRICT_EVENT_MISMATCH', 'reason': 'both_ends_reactant'}}, 'graph_identity', 'both_ends_reactant'),
    ({'failure': 'wall_timeout'}, 'execution', 'wall_timeout')])
def test_later_stage_reasons_and_execution_failure_are_retained(tail, stage, reason):
    raw = dict(ts={'status': 'TS_OPTFREQ_PASS'}, **tail)
    result = stage_diagnostics(raw)
    assert result['terminal_stage'] == stage
    assert result['stage_reason'] == reason
    assert len(result['raw_evidence_sha256']) == 64


def test_absent_evidence_is_unknown_not_a_chemical_negative():
    result = feedback({})
    assert result['terminal_stage'] == 'unknown'
    assert result['reason'] is None
    assert result['evidence_status'] == 'attempted_unresolved'
