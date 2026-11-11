"""Run-layer regressions for #193. Recording/failing backends never do QC."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from xtbflow.calculators import CalculationResult, CalculatorProtocol, MolecularSystem
from xtbflow.sampling.parallel_guidance import GenerationSnapshot
from xtbflow.sampling import live_physical


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / 'scripts/run_real_feedback_loop.py'
    spec = importlib.util.spec_from_file_location('live_arm_recovery', path)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    certifications = []
    def certification(**kwargs):
        certifications.append(kwargs['final_snapshot'].snapshot_id)
        return {'status': 'recording_fake', 'reason_code': 'FAKE', 'backend_result': None}
    monkeypatch.setattr(runner, '_run_certification', certification)
    protocol = CalculatorProtocol('fake', 'xtb_oracle', 'GFN2-xTB')
    system = MolecularSystem(('O','H'), ((0.,0.,0.), (1.,0.,0.)), 0, 1, {'periodic':False})
    snapshot = GenerationSnapshot('p','a',0,system.coordinates,(0.,0.,0.,0.),(True,)*4,0)
    class Backend:
        calls = 0
        def evaluate(self, current, **kwargs):
            self.calls += 1
            return CalculationResult(calculator='xtb_oracle', protocol_id='fake', input_hash=current.input_hash,
                charge=current.charge, multiplicity=current.multiplicity, operation='energy_forces',
                energy=-1., forces=((.2,0.,0.), (0.,.1,0.)), calculator_build_hash='fake')
    backend = Backend()
    config = dict(budget=dict(max_model_forwards=4, max_calculator_calls=2, max_wall_seconds=10.,
        max_cpu_core_hours=.1, max_wall_minutes_per_job=1), guidance={'estimated_wall_seconds':.25},
        merge=dict(geometry_trust_radius=.1, collision_distance=.2, event_freeze_step=10,
                   constraint_matrix=[[1.,1.,1.,1.]]), certification={'enabled':True})
    kwargs = dict(config=config, system=system, snapshot=snapshot, backend=backend,
                  protocol=protocol, blocked_reason=None, output_dir=tmp_path)
    return runner, kwargs, backend, certifications


def hashes(path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()}


def test_completed_arm_replays_candidate_and_cumulative_cost_without_recertifying(fixture):
    runner, kwargs, backend, certifications = fixture
    first = runner._run_arm('physical_feedback', **kwargs)
    before = hashes(kwargs['output_dir'])
    second = runner._run_arm('physical_feedback', **kwargs)
    assert backend.calls == 1 and len(certifications) == 1
    assert first['feedback_used'] is second['feedback_used'] is True
    assert first['geometry_changed'] is second['geometry_changed'] is True
    assert first['snapshot_after'] == second['snapshot_after']
    assert first['feedback'] == second['feedback']
    assert first['runtime_ledger'] == second['runtime_ledger']
    assert second['runtime_ledger']['totals']['calculator_calls'] == 1
    assert second['new_guidance_cost']['calculator_calls'] == 0
    assert second['new_certification_calls'] == 0
    assert hashes(kwargs['output_dir']) == before


def test_wall_overrun_blocks_certification_despite_returned_feedback(fixture, monkeypatch):
    runner, kwargs, backend, certifications = fixture
    kwargs['config']['budget']['max_wall_seconds'] = 1.
    times = iter([0.,2.,2.])
    monkeypatch.setattr(live_physical, 'perf_counter', lambda: next(times))
    row = runner._run_arm('physical_feedback', **kwargs)
    assert backend.calls == 1 and row['feedback'] is not None
    assert row['physical_provider_status'] == 'budget_exhausted'
    assert row['guidance_budget_halted']
    assert row['certification']['reason_code'] == 'GUIDANCE_BUDGET_EXHAUSTED'
    assert row['guidance_cumulative_cost']['wall_seconds'] == 2.
    assert row['runtime_ledger']['totals']['calculator_calls'] == 1
    assert certifications == []


def test_unknown_started_journal_stops_recovery_without_any_write(fixture):
    runner, kwargs, backend, certifications = fixture
    path = kwargs['output_dir']/'provider_attempts_physical_feedback.json'
    path.write_text(json.dumps({'attempts': {'request': {'status':'started',
        'backend_termination':'unknown', 'measured_cost':None,
        'charged_cost':{'calculator_calls':1,'wall_seconds':2.}}}}))
    before = hashes(kwargs['output_dir'])
    row = runner._run_arm('physical_feedback', **kwargs)
    assert backend.calls == 0 and certifications == []
    assert row['certification']['reason_code'] == 'BACKEND_TERMINATION_UNKNOWN'
    assert row['guidance_cumulative_cost']['calculator_calls'] == 1
    assert row['runtime_ledger'] is None  # no invented zero-work public ledger
    assert hashes(kwargs['output_dir']) == before


def test_unsettled_cost_blocks_certifier_and_retains_partial_event(fixture, monkeypatch):
    runner, kwargs, backend, certifications = fixture
    def fail(*args, **kw):
        raise RuntimeError('injected settlement failure')
    monkeypatch.setattr(runner.RunLedger, 'append', fail)
    row = runner._run_arm('physical_feedback', **kwargs)
    assert backend.calls == 1 and certifications == []
    assert row['certification']['reason_code'] == 'COST_SETTLEMENT_INCOMPLETE'
    assert row['unsettled_event']['calculator_calls'] == 1
    assert row['runtime_ledger'] is None
    assert (kwargs['output_dir']/'settlement_failed_physical_feedback.json').exists()


def test_interrupted_certification_preserves_candidate_and_never_restarts(fixture, monkeypatch):
    runner, kwargs, backend, certifications = fixture
    def fail(**values):
        certifications.append(values['final_snapshot'].snapshot_id)
        raise RuntimeError('interrupted after certification entry')
    monkeypatch.setattr(runner, '_run_certification', fail)
    with pytest.raises(RuntimeError, match='interrupted'):
        runner._run_arm('physical_feedback', **kwargs)
    before = hashes(kwargs['output_dir'])
    row = runner._run_arm('physical_feedback', **kwargs)
    assert backend.calls == 1 and len(certifications) == 1
    assert row['certification']['reason_code'] == 'ARM_RECOVERY_INCOMPLETE'
    assert row['preserved_candidate']['row']['feedback_used']
    assert row['preserved_candidate']['row']['geometry_changed']
    assert row['runtime_ledger']['totals']['calculator_calls'] == 1
    assert hashes(kwargs['output_dir']) == before


@pytest.mark.parametrize('change', ['config', 'snapshot', 'ledger'])
def test_mismatched_resume_refuses_work_and_preserves_files(fixture, change):
    runner, kwargs, backend, certifications = fixture
    runner._run_arm('physical_feedback', **kwargs)
    if change == 'config':
        kwargs['config']['guidance']['geometry_scale'] = .5
    elif change == 'snapshot':
        kwargs['snapshot'] = replace(kwargs['snapshot'], generation_step=1, snapshot_id='')
    else:
        with (kwargs['output_dir']/'ledger_physical_feedback.json').open('a') as stream:
            stream.write('\n')
    before = hashes(kwargs['output_dir'])
    row = runner._run_arm('physical_feedback', **kwargs)
    assert row['certification']['reason_code'] == 'ARM_RECOVERY_IDENTITY_MISMATCH'
    assert backend.calls == 1 and len(certifications) == 1
    assert hashes(kwargs['output_dir']) == before
