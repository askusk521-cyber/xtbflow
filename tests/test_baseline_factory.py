from __future__ import annotations

import pytest
import torch

from xtbflow.models import (
    BASELINE_ARM_IDS,
    BaselineFactory,
    BondEditHint,
    build_control,
    pack_be,
    total_electron_projector,
)


SYMBOLS = ("C", "O", "H")


def factory():
    return BaselineFactory(
        total_electron_projector(SYMBOLS, dtype=torch.float64),
        node_feature_dim=4,
        condition_feature_dim=2,
        hidden_dim=8,
        radial_features=4,
        default_dt=0.2,
    )


def inputs():
    event = pack_be(torch.tensor([[[4.0, 0.2, 0.1], [0.2, 6.0, 0.3], [0.1, 0.3, 1.0]]], dtype=torch.float64))
    coordinates = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float64)
    node = torch.randn(1, 3, 4, dtype=torch.float64)
    mask = torch.ones(1, 3, dtype=torch.bool)
    condition = torch.tensor([[0.1, -0.2]], dtype=torch.float64)
    return event, coordinates, node, mask, condition


def test_factory_exposes_all_eight_arms_and_fresh_parameters():
    controls = factory().build_all(seed=11)
    assert tuple(controls) == BASELINE_ARM_IDS
    assert controls["strong_rule"].parameter_count == 0
    learned = [controls[name] for name in BASELINE_ARM_IDS if name != "strong_rule"]
    assert all(control.parameter_count > 0 for control in learned)
    parameter_ids = [id(parameter) for control in learned for parameter in control.parameters()]
    assert len(parameter_ids) == len(set(parameter_ids))


def test_true_independent_and_full_two_stage_do_not_share_parameter_objects():
    independent = factory().build("separate_independent", seed=11)
    full = factory().build("full_two_stage", seed=11)
    assert independent.module.event_branch is not independent.module.geometry_branch
    assert full.module.event_stage is not full.module.geometry_stage
    assert set(id(parameter) for parameter in independent.module.event_branch.parameters()).isdisjoint(
        id(parameter) for parameter in independent.module.geometry_branch.parameters()
    )
    assert set(id(parameter) for parameter in full.module.event_stage.parameters()).isdisjoint(
        id(parameter) for parameter in full.module.geometry_stage.parameters()
    )


def test_forward_contract_returns_paired_output_for_learned_arms():
    event, coordinates, node, mask, condition = inputs()
    for arm_id in BASELINE_ARM_IDS[1:]:
        control = factory().build(arm_id, seed=11)
        output = control(event, coordinates, node, mask, tau=0.3, condition_features=condition)
        assert output.event_velocity.shape == event.shape
        assert output.geometry_velocity.shape == coordinates.shape
        assert torch.isfinite(output.event_velocity).all()
        assert torch.isfinite(output.geometry_velocity).all()


def test_no_projection_is_explicit_mechanism_intervention():
    event, coordinates, node, mask, condition = inputs()
    control = factory().build("current_two_way_no_projection", seed=11)
    projected = factory().build("current_two_way", seed=11)
    no_projection = control(event, coordinates, node, mask, tau=0.3, condition_features=condition)
    with_projection = projected(event, coordinates, node, mask, tau=0.3, condition_features=condition)
    assert not torch.allclose(no_projection.event_velocity, with_projection.event_velocity)
    assert control.contract()["experiment_type"] == "mechanism_intervention"


def test_strong_rule_initial_guess_is_reactant_only_and_bounded():
    event, coordinates, _, mask, _ = inputs()
    strong = factory().build("strong_rule")
    candidates = strong.sample(coordinates, mask, bond_edits=(BondEditHint(0, 1, 1),))
    assert candidates[0]["calculator_calls"] == 0
    assert candidates[0]["status"] == "proposed"
    assert (candidates[0]["geometry"] - coordinates).norm(dim=-1).max().item() <= 0.30 + 1e-8
    unsupported = strong.sample(coordinates, mask)
    assert unsupported[0]["status"] == "unsupported"


def test_manifest_reports_capacity_and_separates_intervention():
    manifest = factory().manifest(seed=11)
    assert manifest["schema"] == "xtbflow-baseline-controls/v1"
    assert manifest["arm_ids"] == list(BASELINE_ARM_IDS)
    records = {record["arm_id"]: record for record in manifest["controls"]}
    assert records["current_two_way_no_projection"]["experiment_type"] == "mechanism_intervention"
    assert "capacity_report" in records["separate_independent"]
    assert manifest["resource_contract"]["candidate_refill"] is False


def test_build_alias_rejects_unknown_arm():
    with pytest.raises(ValueError, match="unknown baseline arm"):
        build_control(factory(), "unknown")
