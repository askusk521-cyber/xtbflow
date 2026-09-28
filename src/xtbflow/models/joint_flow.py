"""Joint and serial event/geometry flow wiring for controlled ablations.

The joint module exchanges the current electronic state and geometric summary
at every declared velocity evaluation. ``coupling_strength=0`` removes both
messages exactly, providing a strict no-coupling control with shared weights.
It does not interpret the flow coordinate as physical time.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .conserved_be import ConservationProjector
from .event_flow import ConservedEventFlow
from .geometry_flow import EquivariantGeometryFlow


def _normalize_tau(tau: float | Tensor, batch: int, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Normalize scalar or per-sample flow time to a finite ``[B, 1]`` tensor."""

    if isinstance(tau, Tensor):
        if tau.device != device or tau.dtype != dtype or tau.ndim not in {0, 1, 2}:
            raise ValueError("tau tensor must match the state device/dtype and be scalar, [B], or [B,1]")
        if tau.ndim == 0:
            value = tau.reshape(1, 1).expand(batch, 1)
        elif tau.ndim == 1 and tau.shape == (batch,):
            value = tau[:, None]
        elif tau.ndim == 2 and tau.shape == (batch, 1):
            value = tau
        else:
            raise ValueError("tau tensor must be scalar, [B], or [B,1]")
    elif isinstance(tau, (int, float)) and not isinstance(tau, bool):
        value = torch.full((batch, 1), float(tau), dtype=dtype, device=device)
    else:
        raise ValueError("tau must be a finite scalar or per-sample tensor")
    if not bool(torch.isfinite(value).all()) or not bool(((value >= 0) & (value <= 1)).all()):
        raise ValueError("tau must be finite and in [0,1]")
    return value



def _validate_state(event_state: Tensor, coordinates: Tensor, node_features: Tensor, atom_mask: Tensor, event_dim: int, node_dim: int) -> tuple[int, int]:
    if event_state.ndim != 2 or event_state.shape[1] != event_dim or not event_state.is_floating_point():
        raise ValueError("event_state must be floating [B,F]")
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3 or not coordinates.is_floating_point():
        raise ValueError("coordinates must be floating [B,N,3]")
    batch, atoms, _ = coordinates.shape
    if event_state.shape[0] != batch:
        raise ValueError("event_state and coordinates batch dimensions must match")
    if node_features.shape != (batch, atoms, node_dim) or node_features.dtype != coordinates.dtype or node_features.device != coordinates.device:
        raise ValueError("node_features must match coordinates and declared width")
    if event_state.dtype != coordinates.dtype or event_state.device != coordinates.device:
        raise ValueError("event_state must match coordinates dtype and device")
    if atom_mask.shape != (batch, atoms) or atom_mask.dtype is not torch.bool or atom_mask.device != coordinates.device:
        raise ValueError("atom_mask must be boolean [B,N] on the coordinate device")
    if not bool(torch.isfinite(event_state).all()) or not bool(torch.isfinite(coordinates).all()) or not bool(torch.isfinite(node_features).all()):
        raise ValueError("flow inputs must be finite")
    return batch, atoms


class JointFlowOutput:
    """Velocity fields and their exchanged intermediate summaries."""

    def __init__(self, event_velocity: Tensor, geometry_velocity: Tensor, event_message: Tensor, geometry_message: Tensor):
        self.event_velocity = event_velocity
        self.geometry_velocity = geometry_velocity
        self.event_message = event_message
        self.geometry_message = geometry_message


class JointEventGeometryFlow(nn.Module):
    """Bidirectional event/geometry velocity field with explicit coupling."""

    def __init__(self, event_projector: ConservationProjector, node_feature_dim: int, *, hidden_dim: int = 64, radial_features: int = 16):
        super().__init__()
        if type(node_feature_dim) is not int or node_feature_dim < 1:
            raise ValueError("node_feature_dim must be positive")
        self.register_buffer("constraint_matrix", event_projector.constraint_matrix.detach().clone())
        self.tolerance = event_projector.tolerance
        self.event_dim = event_projector.n_features
        self.node_feature_dim = node_feature_dim
        self.event_to_geometry = nn.Sequential(nn.Linear(self.event_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, node_feature_dim))
        self.geometry_to_event = nn.Sequential(nn.Linear(node_feature_dim + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, self.event_dim))
        self.event_head = nn.Sequential(nn.Linear(self.event_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, self.event_dim))
        self.geometry = EquivariantGeometryFlow(node_feature_dim, hidden_dim=hidden_dim, radial_features=radial_features)

    @property
    def event_flow(self) -> ConservedEventFlow:
        return ConservedEventFlow(ConservationProjector(self.constraint_matrix, self.tolerance))

    def forward(
        self,
        event_state: Tensor,
        coordinates: Tensor,
        node_features: Tensor,
        atom_mask: Tensor,
        *,
        coupling_strength: float = 1.0,
        tau: float | Tensor = 0.0,
    ) -> JointFlowOutput:
        _validate_state(event_state, coordinates, node_features, atom_mask, self.event_dim, self.node_feature_dim)
        tau_value = _normalize_tau(tau, event_state.shape[0], dtype=event_state.dtype, device=event_state.device)
        if not isinstance(coupling_strength, (int, float)) or coupling_strength < 0 or not torch.isfinite(torch.tensor(float(coupling_strength))):
            raise ValueError("coupling_strength must be finite and nonnegative")
        strength = float(coupling_strength)
        # The parameter-free time condition preserves old checkpoint shapes while
        # making the declared normalized generation coordinate observable.
        conditioned_event = event_state + tau_value
        geometry_message = self.event_to_geometry(conditioned_event)
        conditioned_nodes = node_features + strength * geometry_message[:, None, :]
        geometry_velocity = self.geometry(coordinates, conditioned_nodes, atom_mask)
        masked_nodes = torch.where(atom_mask[..., None], node_features, torch.zeros_like(node_features))
        node_mean = masked_nodes.sum(dim=1) / atom_mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=node_features.dtype)
        displacement = coordinates[:, :, None, :] - coordinates[:, None, :, :]
        squared_distances = displacement.square().sum(dim=-1)
        # ``sqrt(0)`` has an undefined derivative.  Clamp before the square
        # root so padded/self pairs and coincident atoms produce a finite,
        # explicitly regularized summary and therefore a usable coordinate
        # gradient.  Self pairs are excluded from the mean below, so the
        # regularizer cannot change the value for any valid pair.
        distances = squared_distances.clamp_min(torch.finfo(coordinates.dtype).eps).sqrt()
        off_diagonal = ~torch.eye(coordinates.shape[1], dtype=torch.bool, device=coordinates.device)[None, :, :]
        pair_mask = atom_mask[:, :, None] & atom_mask[:, None, :] & off_diagonal
        distance_mean = torch.where(pair_mask, distances, torch.zeros_like(distances)).sum(dim=(1, 2)) / pair_mask.sum(dim=(1, 2)).clamp_min(1).to(dtype=distances.dtype)
        geometry_summary = torch.cat((node_mean, distance_mean[:, None]), dim=-1)
        event_message = self.geometry_to_event(geometry_summary)
        raw_event_velocity = self.event_head(conditioned_event) + strength * event_message
        event_velocity = self.event_flow.project_velocity(raw_event_velocity)
        return JointFlowOutput(event_velocity, geometry_velocity, event_message, geometry_message)


class SerialEventGeometryFlow(nn.Module):
    """Registered event-then-geometry baseline using the same component APIs."""

    def __init__(self, event_projector: ConservationProjector, node_feature_dim: int, *, hidden_dim: int = 64, radial_features: int = 16):
        super().__init__()
        self.joint = JointEventGeometryFlow(event_projector, node_feature_dim, hidden_dim=hidden_dim, radial_features=radial_features)
        self.serial_event_to_geometry = nn.Linear(self.joint.event_dim, node_feature_dim)

    def forward(
        self,
        event_state: Tensor,
        coordinates: Tensor,
        node_features: Tensor,
        atom_mask: Tensor,
        *,
        tau: float | Tensor = 0.0,
        dt: float = 1.0,
    ) -> JointFlowOutput:
        if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not torch.isfinite(torch.tensor(float(dt))) or float(dt) <= 0:
            raise ValueError("dt must be a finite positive scalar")
        output = self.joint(event_state, coordinates, node_features, atom_mask, coupling_strength=0.0, tau=tau)
        event_state_next = event_state + float(dt) * output.event_velocity
        geometry_nodes = node_features + self.serial_event_to_geometry(event_state_next)[:, None, :]
        geometry_velocity = self.joint.geometry(coordinates, geometry_nodes, atom_mask)
        return JointFlowOutput(output.event_velocity, geometry_velocity, output.event_message * 0.0, self.serial_event_to_geometry(event_state_next))
