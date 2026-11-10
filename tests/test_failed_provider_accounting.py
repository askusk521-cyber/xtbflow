"""Started failure, budget overrun and durable recovery; fake work only."""
from dataclasses import replace
import json

import pytest

from xtbflow.sampling.adapters import AdapterRunner, AdapterStatus, GuidanceBudget, MockPhysicalProvider, ProviderRequest
from xtbflow.sampling.parallel_guidance import GenerationSnapshot, GuidanceCost, GuidanceLane, GuidanceOutput


def setup(provider, limit=4):
    snapshot = GenerationSnapshot('p', 'a', 0, ((0.,0.,0.), (1.,0.,0.)), (0.,0.), (True,True), 0)
    req = ProviderRequest(snapshot, GuidanceLane.GEOMETRY, provider.provider_revision,
                          provider.config_fingerprint, provider.protocol_fingerprint)
    budget = GuidanceBudget(GuidanceCost(model_forwards=4, calculator_calls=limit, wall_seconds=10, cpu_seconds=10))
    return req, budget


class PartialFailure(MockPhysicalProvider):
    work_terminated = True
    def __init__(self, calls=1, timeout=False, malformed=False):
        super().__init__()
        self.calls, self.timeout, self.malformed, self.started = calls, timeout, malformed, 0
    def invoke(self, request):
        self.started += self.calls
        if self.malformed:
            return {'not': 'a GuidanceOutput'}
        raise TimeoutError('after start') if self.timeout else RuntimeError('after start')
    def partial_cost(self, request):
        return GuidanceCost(calculator_calls=self.calls, wall_seconds=.02, cpu_seconds=.01)


@pytest.mark.parametrize('calls,timeout,malformed', [(1,False,False), (1,True,False), (3,False,False), (1,False,True)])
def test_failed_work_survives_and_duplicate_does_not_launch(tmp_path, calls, timeout, malformed):
    provider = PartialFailure(calls, timeout, malformed)
    request, budget = setup(provider)
    path = tmp_path/'attempts.json'
    runner = AdapterRunner(budget, journal_path=path)
    run = runner.run(request, provider)
    assert run.charged_cost.calculator_calls == provider.started == calls
    assert run.measured_cost.cpu_seconds == .01
    assert budget.reserved == GuidanceCost.zero()
    assert runner.run(request, provider).status is AdapterStatus.DUPLICATE
    _, recovered_budget = setup(provider)
    recovered = AdapterRunner(recovered_budget, journal_path=path)
    assert recovered.run(request, provider).status is AdapterStatus.DUPLICATE
    assert recovered_budget.charged.calculator_calls == calls
    assert provider.started == calls


def test_prelaunch_refusal_performs_zero_work(tmp_path):
    provider = PartialFailure()
    request, budget = setup(provider, 0)
    run = AdapterRunner(budget, journal_path=tmp_path/'a.json').run(request, provider)
    assert provider.started == 0
    assert run.charged_cost == GuidanceCost.zero()
    assert run.failure_stage == 'budget_reservation'


def test_overrun_charges_all_work_and_blocks_next_call(tmp_path):
    provider = PartialFailure(calls=3)
    request, budget = setup(provider, 2)
    runner = AdapterRunner(budget, journal_path=tmp_path/'a.json')
    run = runner.run(request, provider)
    assert run.status is AdapterStatus.BUDGET_EXHAUSTED
    assert budget.charged.calculator_calls == 3
    assert budget.reserved == GuidanceCost.zero()
    next_request = replace(request, snapshot=replace(request.snapshot, generation_step=1, snapshot_id=''), request_fingerprint='')
    assert runner.run(next_request, provider).failure_stage == 'budget_reservation'
    assert provider.started == 3


def test_unknown_timeout_is_not_measured_zero_and_halts_shared_budget(tmp_path):
    provider = MockPhysicalProvider()
    provider.invoke = lambda request: (_ for _ in ()).throw(TimeoutError('termination unconfirmed'))
    request, budget = setup(provider)
    run = AdapterRunner(budget, journal_path=tmp_path/'a.json').run(request, provider)
    assert run.measured_cost is None
    assert run.charged_cost.calculator_calls == 1
    assert run.ledger_entry.backend_termination == 'unknown'
    assert budget.halted
    from xtbflow.sampling.parallel_guidance import merge_guidance
    from xtbflow.sampling.physical_adapter import PhysicalAblationArtifact
    from xtbflow.sampling.policy_adapter import PolicyAblationArtifact
    from xtbflow.sampling.adapters import AdapterLedger
    merged = merge_guidance(request.snapshot, ())
    for cls in (PhysicalAblationArtifact, PolicyAblationArtifact):
        artifact = cls(True, 'pool', request.snapshot, merged.snapshot, run, merged,
                       AdapterLedger((run.ledger_entry,)), ('software only',))
        assert artifact.to_mapping()['adapter_run']['measured_cost'] is None
        assert artifact.to_mapping()['adapter_run']['backend_termination'] == 'unknown'


def test_started_journal_exists_before_call_and_recovers_as_unknown(tmp_path):
    provider = MockPhysicalProvider()
    request, budget = setup(provider)
    path = tmp_path/'attempts.json'
    def crash(req):
        recorded = json.loads(path.read_text())['attempts'][req.request_fingerprint]
        assert recorded['status'] == 'started'
        assert recorded['measured_cost'] is None
        raise KeyboardInterrupt('process interruption before settlement')
    provider.invoke = crash
    result = AdapterRunner(budget, journal_path=path).run(request, provider)
    assert result.status is AdapterStatus.CANCELLED
    assert budget.reserved == GuidanceCost.zero()
    _, fresh = setup(provider)
    recovered = AdapterRunner(fresh, journal_path=path)
    assert fresh.charged.calculator_calls == 1 and fresh.halted
    assert recovered.run(request, provider).status is AdapterStatus.DUPLICATE


def test_live_failure_common_ledger_reconciles_same_call_identity(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    import sys
    from xtbflow.calculators import CalculatorProtocol, MolecularSystem
    path = Path(__file__).parents[1]/'scripts/run_real_feedback_loop.py'
    spec = importlib.util.spec_from_file_location('live_failure_runner', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, '_run_certification', lambda **kwargs: {'status': 'blocked'})
    class Backend:
        started = 0
        def evaluate(self, system, **kwargs):
            self.started += 1
            raise RuntimeError('failed synchronous backend')
    backend = Backend()
    system = MolecularSystem(('H','H'), ((0.,0.,0.), (1.,0.,0.)), 0, 1, {'periodic':False})
    snapshot = GenerationSnapshot('p','a',0,system.coordinates,(0.,0.),(True,True),0)
    config = dict(budget=dict(max_model_forwards=4,max_calculator_calls=2,max_wall_seconds=10,
                             max_cpu_core_hours=.1,max_wall_minutes_per_job=1), guidance={})
    row = module._run_arm('no_feedback', config=config, system=system, snapshot=snapshot,
                          backend=backend, protocol=CalculatorProtocol('fake','xtb_oracle','GFN2-xTB'),
                          blocked_reason=None, output_dir=tmp_path)
    assert backend.started == 1
    assert row['feedback'] is None
    assert row['runtime_ledger']['totals']['calculator_calls'] == 1
    assert row['adapter_ledger']['entries'][0]['charged_cost']['calculator_calls'] == 1
    assert row['physical_cost_basis'] == 'measured_partial'
