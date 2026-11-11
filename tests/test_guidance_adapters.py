from __future__ import annotations

import pytest

from xtbflow.sampling import (
    AdapterRunner,
    AdapterStatus,
    GuidanceBudget,
    GuidanceCost,
    GuidanceLane,
    GuidanceOutput,
    GenerationSnapshot,
    MockPhysicalProvider,
    MockPolicyProvider,
    ProviderRequest,
    fingerprint_text,
    merge_guidance,
    validate_sha256,
)


def state() -> GenerationSnapshot:
    return GenerationSnapshot(
        parent_id="parent-adapter",
        attempt_id="attempt-adapter",
        generation_step=0,
        geometry=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        event=(0.0, 0.0, 0.0, 0.0),
        event_mask=(True, True, False, True),
        random_seed=1,
        metadata={"temperature": 298.15},
        provenance={"fixture": "adapter"},
    )


def request(snapshot: GenerationSnapshot, lane: GuidanceLane, provider) -> ProviderRequest:
    return ProviderRequest(
        snapshot=snapshot,
        lane=lane,
        provider_revision=provider.provider_revision,
        config_fingerprint=provider.config_fingerprint,
        protocol_fingerprint=provider.protocol_fingerprint,
        feedback={"force_norm": 0.2},
        action_space={"max_delta": 0.1},
    )


def test_fingerprints_are_strict_lowercase_sha256():
    value = fingerprint_text("adapter-v1")
    assert len(value) == 64
    assert validate_sha256(value, "value") == value
    with pytest.raises(ValueError, match="lowercase hexadecimal SHA-256"):
        validate_sha256("A" * 64, "value")
    with pytest.raises(ValueError, match="lowercase hexadecimal SHA-256"):
        ProviderRequest(
            snapshot=state(), lane=GuidanceLane.EVENT,
            provider_revision="bad", config_fingerprint=value, protocol_fingerprint=value,
        )


def test_physical_adapter_reserves_before_invoke_and_returns_only_guidance_output():
    snapshot = state()
    provider = MockPhysicalProvider()
    budget = GuidanceBudget(GuidanceCost(model_forwards=1, calculator_calls=1, wall_seconds=1.0))
    run = AdapterRunner(budget).run(request(snapshot, GuidanceLane.GEOMETRY, provider), provider)
    assert run.status is AdapterStatus.ACCEPTED
    assert run.output is not None
    assert run.output.lane is GuidanceLane.GEOMETRY
    assert run.output.input_fingerprint == snapshot.snapshot_id
    assert run.charged_cost.calculator_calls == 1
    assert budget.reserved == GuidanceCost.zero()
    assert budget.charged.calculator_calls == 1
    assert snapshot.geometry == ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert run.output.to_mapping()["provenance"]["adapter"]["request_fingerprint"] == run.request_fingerprint


def test_policy_adapter_output_reaches_merge_gate_without_mutating_snapshot():
    snapshot = state()
    provider = MockPolicyProvider()
    budget = GuidanceBudget(GuidanceCost(model_forwards=1, calculator_calls=0, wall_seconds=1.0))
    run = AdapterRunner(budget).run(request(snapshot, GuidanceLane.EVENT, provider), provider)
    assert run.status is AdapterStatus.ACCEPTED
    merged = merge_guidance(
        snapshot,
        (run.output,),
        constraint_matrix=((1.0, 1.0, 1.0, 1.0),),
    )
    assert merged.snapshot.event != snapshot.event
    assert snapshot.event == (0.0, 0.0, 0.0, 0.0)


def test_budget_reservation_happens_before_provider_invocation():
    snapshot = state()
    provider = MockPhysicalProvider()
    budget = GuidanceBudget(GuidanceCost(model_forwards=1, calculator_calls=1, wall_seconds=1.0))
    observed = []
    original = provider.invoke

    def invoke(request):
        observed.append(budget.reserved)
        return original(request)

    provider.invoke = invoke
    AdapterRunner(budget).run(request(snapshot, GuidanceLane.GEOMETRY, provider), provider)
    assert observed and observed[0].model_forwards == 1
    assert observed[0].calculator_calls == 1


def test_budget_exhaustion_prevents_provider_invocation_and_is_ledgered():
    snapshot = state()
    provider = MockPhysicalProvider()
    budget = GuidanceBudget(GuidanceCost(model_forwards=0, calculator_calls=0, wall_seconds=0.0))
    runner = AdapterRunner(budget)
    run = runner.run(request(snapshot, GuidanceLane.GEOMETRY, provider), provider)
    assert run.status is AdapterStatus.BUDGET_EXHAUSTED
    assert run.failure_stage == "budget_reservation"
    assert provider.invocations == 0
    assert runner.ledger.entries[0].charged_cost == GuidanceCost.zero()


def test_timeout_is_mapped_to_explicit_failure_stage_and_reconciled():
    snapshot = state()
    provider = MockPhysicalProvider()
    provider.invoke = lambda request: (_ for _ in ()).throw(TimeoutError("provider timeout"))
    budget = GuidanceBudget(GuidanceCost(model_forwards=1, calculator_calls=1, wall_seconds=1.0))
    run = AdapterRunner(budget).run(request(snapshot, GuidanceLane.GEOMETRY, provider), provider)
    assert run.status is AdapterStatus.TIMEOUT
    assert run.failure_stage == "provider_timeout"
    assert run.charged_cost.calculator_calls == 1
    assert run.measured_cost is None
    assert budget.halted
    assert budget.reserved == GuidanceCost.zero()


def test_duplicate_request_is_not_invoked_or_charged_twice():
    snapshot = state()
    provider = MockPolicyProvider()
    budget = GuidanceBudget(GuidanceCost(model_forwards=2, wall_seconds=1.0))
    runner = AdapterRunner(budget)
    first = runner.run(request(snapshot, GuidanceLane.EVENT, provider), provider)
    second = runner.run(request(snapshot, GuidanceLane.EVENT, provider), provider)
    assert first.status is AdapterStatus.ACCEPTED
    assert second.status is AdapterStatus.DUPLICATE
    assert provider.invocations == 1
    assert second.charged_cost == GuidanceCost.zero()
    assert budget.charged.model_forwards == 1


def test_request_firewall_rejects_nested_target_fields():
    provider = MockPolicyProvider()
    with pytest.raises(ValueError, match="target-derived"):
        ProviderRequest(
            snapshot=state(), lane=GuidanceLane.EVENT,
            provider_revision=provider.provider_revision,
            config_fingerprint=provider.config_fingerprint,
            protocol_fingerprint=provider.protocol_fingerprint,
            feedback={"nested": [{"referenceMode": "hidden"}]},
        )
