"""True-independent and complete serial baseline wrappers."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .joint_flow import JointEventGeometryFlow, JointFlowOutput


class SeparateIndependentFlow(nn.Module):
    """Two parameter spaces that exchange neither generated state.

    The two branches receive the same declared reactant-side tensors and
    condition features.  Their parameters and module buffers are distinct;
    the output messages are explicitly zero so a caller cannot mistake this
    control for a shared-weight intervention.
    """

    def __init__(self, event_branch: JointEventGeometryFlow, geometry_branch: JointEventGeometryFlow) -> None:
        super().__init__()
        if event_branch is geometry_branch:
            raise ValueError("independent branches must be distinct module objects")
        self.event_branch = event_branch
        self.geometry_branch = geometry_branch

    def forward(
        self,
        event_state: Tensor,
        coordinates: Tensor,
        node_features: Tensor,
        atom_mask: Tensor,
        *,
        tau: float | Tensor,
        condition_features: Tensor | None = None,
        conservation_projection: bool = True,
    ) -> JointFlowOutput:
        event_output = self.event_branch.forward_control(
            "both_off",
            event_state,
            coordinates,
            node_features,
            atom_mask,
            tau=tau,
            condition_features=condition_features,
            conservation_projection=conservation_projection,
        )
        geometry_output = self.geometry_branch.forward_control(
            "both_off",
            event_state,
            coordinates,
            node_features,
            atom_mask,
            tau=tau,
            condition_features=condition_features,
            conservation_projection=conservation_projection,
        )
        return JointFlowOutput(
            event_output.event_velocity,
            geometry_output.geometry_velocity,
            torch.zeros_like(event_output.event_velocity),
            torch.zeros_like(geometry_output.geometry_velocity),
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def state_stage_contract(self) -> dict[str, object]:
        return {
            "event_state_dependency": "reactant_event_only",
            "geometry_state_dependency": "reactant_geometry_only",
            "cross_branch_messages": False,
            "parameter_spaces": "independent",
        }


class FullTwoStageFlow(nn.Module):
    """Complete event stage followed by a frozen-event geometry stage."""

    def __init__(self, event_stage: JointEventGeometryFlow, geometry_stage: JointEventGeometryFlow) -> None:
        super().__init__()
        if event_stage is geometry_stage:
            raise ValueError("full two-stage baseline requires distinct stage modules")
        self.event_stage = event_stage
        self.geometry_stage = geometry_stage

    def forward(
        self,
        event_state: Tensor,
        coordinates: Tensor,
        node_features: Tensor,
        atom_mask: Tensor,
        *,
        tau: float | Tensor,
        dt: float,
        coupling_strength: float = 1.0,
        condition_features: Tensor | None = None,
        conservation_projection: bool = True,
    ) -> JointFlowOutput:
        if isinstance(dt, bool) or not isinstance(dt, (int, float)) or float(dt) <= 0:
            raise ValueError("dt must be a finite positive scalar")
        event_output = self.event_stage.forward_control(
            "both_off",
            event_state,
            coordinates,
            node_features,
            atom_mask,
            tau=tau,
            condition_features=condition_features,
            conservation_projection=conservation_projection,
        )
        frozen_event = event_state + float(dt) * event_output.event_velocity
        geometry_output = self.geometry_stage.forward(
            frozen_event,
            coordinates,
            node_features,
            atom_mask,
            tau=tau,
            condition_features=condition_features,
            geometry_to_event_strength=0.0,
            event_to_geometry_strength=coupling_strength,
            conservation_projection=conservation_projection,
        )
        return JointFlowOutput(
            event_output.event_velocity,
            geometry_output.geometry_velocity,
            torch.zeros_like(event_output.event_velocity),
            geometry_output.geometry_message,
        )

    def state_stage_contract(self) -> dict[str, object]:
        return {
            "event_stage": "complete_then_frozen",
            "geometry_stage": "conditioned_on_predicted_event",
            "geometry_to_event_feedback": False,
            "oracle_event": False,
        }
