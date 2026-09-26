"""Small constructive event-flow module with a shared conservation interface."""
from __future__ import annotations

import torch
from torch import nn

from .conserved_be import ConservationProjector


class ConservedEventFlow(nn.Module):
    """Project event velocities and initial noise in a declared null space.

    This module does not invent a chemical event score or a training dataset;
    it only supplies the conservation-preserving state transition used by a
    future learned velocity field.
    """

    def __init__(self, projector: ConservationProjector) -> None:
        super().__init__()
        self.register_buffer("constraint_matrix", projector.constraint_matrix.detach().clone())
        self.tolerance = projector.tolerance

    @property
    def projector(self) -> ConservationProjector:
        return ConservationProjector(self.constraint_matrix, self.tolerance)

    def project_velocity(self, velocity: torch.Tensor) -> torch.Tensor:
        return self.projector.project(velocity)

    def initial_state(self, reactant_state: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if reactant_state.shape != noise.shape:
            raise ValueError("reactant_state and noise must have the same shape")
        return reactant_state + self.project_velocity(noise)

    def step(self, state: torch.Tensor, velocity: torch.Tensor, dt: float) -> torch.Tensor:
        if not isinstance(dt, (int, float)) or dt < 0:
            raise ValueError("dt must be nonnegative")
        if state.shape != velocity.shape:
            raise ValueError("state and velocity must have the same shape")
        return state + float(dt) * self.project_velocity(velocity)

    def forward(self, state: torch.Tensor, velocity: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        return self.step(state, velocity, dt)
