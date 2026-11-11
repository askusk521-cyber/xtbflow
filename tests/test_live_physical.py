from __future__ import annotations

from xtbflow.calculators import CalculationResult, CalculatorBackend, CalculatorProtocol, MolecularSystem
from xtbflow.sampling.adapters import AdapterRunner, GuidanceBudget, ProviderRequest
from xtbflow.sampling.live_physical import LiveXTBEventProvider, LiveXTBGuidanceConfig, LiveXTBPhysicalProvider
from xtbflow.sampling.parallel_guidance import GenerationSnapshot, GuidanceCost, GuidanceLane, MergeConfig, merge_guidance


class FakeXTB(CalculatorBackend):
    def __init__(self, protocol: CalculatorProtocol) -> None:
        self.protocol = protocol
        self.calls = 0

    @property
    def capabilities(self):
        raise NotImplementedError

    def evaluate(self, system, *, operation="energy_forces", budget_token=None):
        self.calls += 1
        return CalculationResult(
            calculator=self.protocol.calculator,
            protocol_id=self.protocol.protocol_id,
            input_hash=system.input_hash,
            charge=system.charge,
            multiplicity=system.multiplicity,
            operation=operation,
            energy=-1.0,
            forces=((0.2, 0.0, 0.0), (0.0, 0.1, 0.0)),
            calculator_build_hash="fake-build",
        )


def _fixture():
    protocol = CalculatorProtocol(
        protocol_id="test-live-xtb-v1",
        calculator="xtb_oracle",
        method="GFN2-xTB",
        parameters={"accuracy": 1.0, "max_iterations": 10, "electronic_temperature": 300, "solvent": None},
    )
    system = MolecularSystem(("O", "H"), ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), 0, 1, {"periodic": False}, "parent")
    snapshot = GenerationSnapshot("parent", "attempt", 0, system.coordinates, (0.0, 0.0, 0.0, 0.0), (True, True, True, True), 7)
    return protocol, system, snapshot


def test_live_provider_uses_current_snapshot_and_merge_updates_geometry_event():
    protocol, system, snapshot = _fixture()
    backend = FakeXTB(protocol)
    config = LiveXTBGuidanceConfig(geometry_scale=0.2, geometry_trust_radius=0.1, event_scale=0.03, event_trust_radius=0.1)
    physical = LiveXTBPhysicalProvider(backend, system, protocol, config=config)
    budget = GuidanceBudget(GuidanceCost(calculator_calls=1, wall_seconds=10.0, cpu_seconds=10.0))
    runner = AdapterRunner(budget)
    request = ProviderRequest(snapshot, GuidanceLane.GEOMETRY, physical.provider_revision, physical.config_fingerprint, physical.protocol_fingerprint)
    run = runner.run(request, physical)
    assert run.status.value == "accepted"
    assert backend.calls == 1
    assert physical.last_feedback is not None
    event_provider = LiveXTBEventProvider(physical.last_feedback, 4, config=config)
    event_request = ProviderRequest(snapshot, GuidanceLane.EVENT, event_provider.provider_revision, event_provider.config_fingerprint, event_provider.protocol_fingerprint)
    event_run = runner.run(event_request, event_provider)
    merged = merge_guidance(
        snapshot,
        tuple(item for item in (run.output, event_run.output) if item is not None),
        config=MergeConfig(geometry_trust_radius=0.1, collision_distance=0.2),
        constraint_matrix=((1.0, 1.0, 1.0, 1.0),),
        event_velocity_fn=lambda _geometry, event, _snapshot: event,
    )
    assert merged.snapshot.geometry != snapshot.geometry
    assert merged.snapshot.event != snapshot.event
    assert sum(merged.decision.conservation_residual) == 0.0
    assert len(merged.ledger.entries) == 2


def test_live_provider_is_snapshot_bound():
    protocol, system, snapshot = _fixture()
    backend = FakeXTB(protocol)
    provider = LiveXTBPhysicalProvider(backend, system, protocol)
    other = GenerationSnapshot("parent", "other", 0, ((0.0, 0.0, 0.0),), snapshot.event, snapshot.event_mask, 7)
    request = ProviderRequest(other, GuidanceLane.GEOMETRY, provider.provider_revision, provider.config_fingerprint, provider.protocol_fingerprint)
    try:
        provider.estimate(request)
    except ValueError:
        pass
    else:
        raise AssertionError("a live provider must reject a mismatched snapshot geometry context")
