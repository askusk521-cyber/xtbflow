from __future__ import annotations

import torch
from torch import nn

from xtbflow.models.joint_flow import JointFlowOutput
from xtbflow.training import apply_gradient_clip, gradient_diagnostics, parameter_groups, train_joint_step


class _ToyJoint(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(2, 2)
        self.event_base = nn.Linear(2, 1)
        self.geometry = nn.Linear(2, 1)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(value)
        return self.event_base(shared), self.geometry(shared)


def _losses(model: _ToyJoint) -> tuple[torch.Tensor, torch.Tensor]:
    value = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
    event, geometry = model(value)
    return (event - 3.0).square().mean(), (geometry + 2.0).square().mean()


def test_gradient_diagnostics_separates_objectives_and_shared_parameters():
    torch.manual_seed(4)
    model = _ToyJoint().double()
    event_loss, geometry_loss = _losses(model)
    report = gradient_diagnostics(model, event_loss, geometry_loss, max_norm=1e-12)
    assert report["schema"] == "xtbflow-joint-gradient-diagnostics/v1"
    assert report["objectives"]["event"]["l2_norm"] > 0
    assert report["objectives"]["geometry"]["l2_norm"] > 0
    assert report["parameter_group_counts"]["event_only"] > 0
    assert report["parameter_group_counts"]["geometry_only"] > 0
    assert report["parameter_group_counts"]["shared"] > 0
    assert report["global_clip"]["triggered"] is True
    assert report["objective_group_norms"]["event"]["geometry_only"]["l2_norm"] == 0.0


def test_global_and_branchwise_clipping_are_explicit_controls():
    torch.manual_seed(8)
    model = _ToyJoint().double()
    event_loss, geometry_loss = _losses(model)
    (event_loss + geometry_loss).backward()
    global_report = apply_gradient_clip(model, max_norm=1e-9, mode="global")
    assert global_report["mode"] == "global"
    assert global_report["global"]["triggered"] is True

    torch.manual_seed(8)
    branch_model = _ToyJoint().double()
    branch_event, branch_geometry = _losses(branch_model)
    (branch_event + branch_geometry).backward()
    branch_report = apply_gradient_clip(branch_model, max_norm=1e-9, mode="branchwise")
    assert branch_report["mode"] == "branchwise"
    assert set(branch_report["groups"]) == {"event_only", "geometry_only", "shared"}
    assert all(item["triggered"] for item in branch_report["groups"].values())


def test_parameter_groups_keep_unclassified_parameters_in_shared_group():
    model = _ToyJoint()
    groups = parameter_groups(model)
    assert len(groups["shared"]) == 2


def test_train_joint_step_persists_preclip_and_applied_clip_records():
    torch.manual_seed(9)
    model = _ToyJoint().double()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    value = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
    event, geometry = model(value)
    output = JointFlowOutput(
        event_velocity=event,
        geometry_velocity=geometry,
        event_message=torch.zeros_like(event),
        geometry_message=torch.zeros_like(geometry),
    )
    result = train_joint_step(
        model,
        optimizer,
        output,
        target_event_velocity=torch.full_like(event, 3.0),
        target_geometry_velocity=torch.full_like(geometry, -2.0),
        clip_mode="branchwise",
        max_norm=1e-9,
    )
    assert result["schema"] == "xtbflow-joint-training-step/v1"
    assert result["gradient_diagnostics"]["global_clip"]["triggered"] is True
    assert result["gradient_clip"]["mode"] == "branchwise"
    assert result["losses"]["total"] > 0
