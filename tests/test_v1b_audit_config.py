"""Structural checks of configs/v1b/audit.json (v2.2 amendment draft)."""
import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / 'configs/v1b/audit.json'


def load():
    return json.loads(CONFIG.read_text(encoding='utf-8'))


def test_v22_draft_is_not_frozen_and_keeps_v21_decision():
    c = load()
    assert c['schema_version'] == 'xtbflow_v1b_protocol_v2_2' and c['frozen'] is False
    assert c['required_v1a_decision'] == 'GO_V1b'
    a = c['analysis']
    assert a['decision_estimand'] == 'delta_cert_B1_minus_A2_budget16_training_seed0'
    assert a['decision'] == {'V1A_CONFIRMED': 'lower_one95 > 0', 'V1A_OVERTURNED': 'upper_one95 <= 0',
                             'V1B_INCONCLUSIVE': 'otherwise',
                             'INCOMPLETE_BOUNDS': 'unresolved labels affect decision'}
    assert a['parent_interval'] == 'stratified_HT_normal_one_sided' and a['alpha_one_sided'] == 0.05
    assert c['source']['budget_equiv_B0_candidates'] == 16 and c['source']['training_seed'] == 0


def test_stage_a_unchanged_and_stage_b_certifies_four_arms():
    s = load()['sampling']
    assert s['A']['seed'] == 'v1b_candidate_A_v2_1' and s['A']['route'] == 'direct_irc'
    assert s['A']['proxy_classes'][0] == 'MATCH_BEST' and len(s['A']['proxy_classes']) == 6
    assert s['B']['arms'] == ['A1', 'A2', 'B0', 'B1']
    assert s['B']['strata'] == ['B1_only', 'A2_only', 'both', 'neither_with_best_event',
                                'primary_empty_secondary_nonempty', 'all_empty']
    assert s['B']['all_empty_policy'] == 'census_without_qc'
    assert s['B']['primary_empty_secondary_nonempty_policy'] == 'stratified_SRSWOR'
    assert 'neither_empty' not in s['B']['strata'] and s['B']['seed'] != s['A']['seed']


def test_secondary_estimands_have_no_gate_and_cut_order_keeps_s3():
    c = load()
    sec = c['analysis']['secondary_estimands']
    assert {k for k in sec if k.startswith('S')} == {'S1', 'S2', 'S3a', 'S3b'}
    assert all(sec[k]['gate'] is False for k in ('S1', 'S2', 'S3a', 'S3b'))
    assert c['budget']['cost_overrun_cut_order'] == ['S1', 'S2']
    assert set(c['budget']['never_cut']) == {'primary', 'S3a', 'S3b'}
    t = c['source']['truncated_sets']
    assert t['S3a']['truncate'] == ['B0'] and t['S3b']['truncate'] == ['A1', 'B1']
    assert t['export_at_D0_with_sha256'] is True
