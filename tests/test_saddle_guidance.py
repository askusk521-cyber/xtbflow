from __future__ import annotations

import pytest
import torch

from xtbflow.calculators import CalculatorProtocol, MolecularSystem, XTBloomAdapter
from xtbflow.physics import (
    FailurePolicy,
    GuidanceConfig,
    GuidanceMode,
    PhysicalPostProcessor,
    apply_guidance,
    finite_difference_hvp,
    saddle_guidance,
)
from xtbflow.sampling import GuidanceSchedule, schedule_strength


def protocol():
    return CalculatorProtocol("toy-gfn2", "xtbloom", "GFN2-xTB", parameters={"toy": True})


def toy(item: MolecularSystem, operation: str):
    energy = 0.5 * sum(value * value for row in item.coordinates for value in row)
    forces = [[-value for value in row] for row in item.coordinates]
    output = {"charge": item.charge, "multiplicity": item.multiplicity, "converged": True}
    if operation in {"energy", "energy_forces"}:
        output["energy"] = energy
    if operation in {"forces", "energy_forces"}:
        output["forces"] = forces
    return output


def system(*, close: bool = False) -> MolecularSystem:
    coordinates = ((0.0, 0.0, 0.0), (0.3, 0.0, 0.0)) if close else ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    return MolecularSystem(("H", "H"), coordinates, 0, 1)


def test_saddle_reflection_uses_force_and_applies_strength_once():
    forces = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    direction = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert torch.equal(saddle_guidance(forces, direction, strength=0.5), torch.tensor([[-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]))


def test_zero_guidance_is_exact_zero_call_baseline():
    backend = XTBloomAdapter(protocol=protocol(), evaluator=toy)
    result = apply_guidance(backend, system(), GuidanceConfig(mode=GuidanceMode.NONE))
    assert result.status == "unchanged"
    assert result.calculator_calls == 0
    assert result.forces is None
    assert result.coordinates == system().coordinates


def test_energy_descent_and_saddle_use_same_bounded_coordinate_interface():
    backend = XTBloomAdapter(protocol=protocol(), evaluator=toy)
    descent = apply_guidance(backend, system(), GuidanceConfig(mode=GuidanceMode.ENERGY_DESCENT, strength=1.0, max_step=0.2, trust_radius=0.2))
    assert descent.status == "updated"
    assert descent.calculator_calls == 1
    assert torch.tensor(descent.guidance).norm().item() == pytest.approx(0.2, abs=1e-7)
    saddle = apply_guidance(backend, system(), GuidanceConfig(mode=GuidanceMode.SADDLE, strength=1.0, max_step=0.2, trust_radius=0.2), reaction_direction=((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    assert saddle.status == "updated"
    assert saddle.calculator_calls == 1


def test_collision_is_rejected_before_calculator_call():
    backend = XTBloomAdapter(protocol=protocol(), evaluator=toy)
    result = apply_guidance(backend, system(close=True), GuidanceConfig(mode=GuidanceMode.ENERGY_DESCENT, strength=1.0))
    assert result.status == "collision_rejected"
    assert result.collision_rejected is True
    assert result.calculator_calls == 0


def test_failure_policy_is_explicit_and_postprocessor_is_bounded():
    def failing(item, operation):
        raise RuntimeError("SCC failure")

    backend = XTBloomAdapter(protocol=protocol(), evaluator=failing)
    result = apply_guidance(backend, system(), GuidanceConfig(mode=GuidanceMode.ENERGY_DESCENT, strength=1.0, failure_policy=FailurePolicy.SKIP))
    assert result.status == "skipped"
    assert result.calculator_calls == 1
    processor = PhysicalPostProcessor(backend, GuidanceConfig(mode=GuidanceMode.ENERGY_DESCENT, strength=1.0), max_calls=2)
    assert len(processor.run(system())) == 1


def test_finite_difference_hvp_is_counted_and_matches_toy_curvature():
    backend = XTBloomAdapter(protocol=protocol(), evaluator=toy)
    result = finite_difference_hvp(backend, system(), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), step=1e-5)
    assert result.status == "success"
    assert result.calculator_calls == 2
    assert torch.tensor(result.hessian_vector_product).norm().item() == pytest.approx(1.0, rel=1e-5)


def test_schedule_and_strength_validation():
    schedule = GuidanceSchedule(start=0.0, end=2.0, warmup_fraction=0.5)
    assert schedule_strength(schedule, 0.25) == 0.0
    assert schedule_strength(schedule, 1.0) == 2.0
    with pytest.raises(ValueError):
        schedule_strength(schedule, 1.1)
