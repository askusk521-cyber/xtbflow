import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from review_open_world_dispatch_v2 import classify, seconds  # noqa: E402


def test_seconds_parses_slurm_runtimes():
    assert seconds('00:12:07') == 727
    assert seconds('1-00:00:01') == 86401


def test_completed_job_reports_the_runner_verdict(tmp_path):
    (tmp_path / 'verdict.json').write_text(json.dumps({'strict_status': 'STRICT_JOINT_GRAPH_VALID'}))
    (tmp_path / 'environment.json').write_text('{}')
    assert classify({'JobState': 'COMPLETED', 'ExitCode': '0:0'}, tmp_path) == (
        'VERDICT', 'STRICT_JOINT_GRAPH_VALID', False)


def test_completed_job_without_records_stops(tmp_path):
    assert classify({'JobState': 'COMPLETED', 'ExitCode': '0:0'}, tmp_path)[2] is True


def test_recorded_chain_exception_continues(tmp_path):
    (tmp_path / 'unexpected_failure.json').write_text(json.dumps({'type': 'EngineError', 'message': 'SCF_LIMIT'}))
    assert classify({'JobState': 'FAILED', 'ExitCode': '1:0'}, tmp_path) == (
        'CHAIN_EXCEPTION', 'EngineError:SCF_LIMIT', False)


def test_unrecorded_failure_and_other_states_stop(tmp_path):
    assert classify({'JobState': 'FAILED', 'ExitCode': '1:0'}, tmp_path)[2] is True
    assert classify({'JobState': 'OUT_OF_MEMORY', 'ExitCode': '0:125'}, tmp_path)[2] is True
    assert classify({'JobState': 'TIMEOUT', 'ExitCode': '0:0'}, tmp_path) == ('JOB_TIMEOUT', None, False)
