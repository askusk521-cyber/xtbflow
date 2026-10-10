import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from review_open_world_dispatch_v3 import classify, preqc_failure, quarantine  # noqa: E402


def _failure(work, stages):
    work.mkdir(parents=True)
    (work / 'unexpected_failure.json').write_text(json.dumps(
        {'type': 'CalledProcessError', 'message': 'nvidia-smi 18', 'stage_timings': stages}))


def test_failure_before_any_qc_stage_is_infrastructure(tmp_path):
    _failure(tmp_path / 'w', [])
    assert classify({'JobState': 'FAILED', 'ExitCode': '1:0'}, tmp_path / 'w')[0::2] == ('INFRASTRUCTURE', True)


def test_failure_after_a_qc_stage_is_a_chain_exception(tmp_path):
    _failure(tmp_path / 'w', [{'stage': 'ts_optts'}])
    assert classify({'JobState': 'FAILED', 'ExitCode': '1:0'}, tmp_path / 'w')[0::2] == ('CHAIN_EXCEPTION', False)


def test_quarantine_moves_only_preqc_failures(tmp_path):
    run = tmp_path / 'o2-run'
    _failure(run / 'candidate_aaaa', [])
    _failure(run / 'reference_bbbb', [{'stage': 'irc'}])
    (run / 'anchor_cccc').mkdir()
    (run / 'anchor_cccc' / 'verdict.json').write_text('{}')
    assert preqc_failure(run / 'candidate_aaaa') and not preqc_failure(run / 'reference_bbbb')
    moved = quarantine(tmp_path, [[{'item_id': 'aaaa', 'job_id': '2825'}]])
    assert [m['item_id'] for m in moved] == ['aaaa']
    assert not (run / 'candidate_aaaa').exists() and (run / 'reference_bbbb').exists()
    assert (tmp_path / 'o2-run-preqc-failed' / 'candidate_aaaa.job2825' / 'unexpected_failure.json').exists()
    assert json.loads((tmp_path / 'o2-run-preqc-failed' / 'quarantine.json').read_text())[0]['job_id'] == '2825'
