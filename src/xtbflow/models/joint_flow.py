"""Permutation-equivariant joint and serial event/geometry flow controls.

The canonical event state is the packed upper triangle of a symmetric
bond/electron matrix.  Pairwise shared networks make event velocities
permutation equivariant, while coordinate updates remain SE(3)-equivariant.
The four learned controls use the same parameters:

``both_off``
    Event and geometry fields see the same reactant-side conditions but do not
    exchange their generated states.
``serial_independent``
    The independently predicted event is advanced by the declared integrator
    step and then conditions geometry; geometry cannot update the event.
``joint_bidirectional``
    Current event and geometry states influence one another at every velocity
    evaluation.
``joint_unidirectional``
    The same current-state Euler evaluation as the joint control, with only
    geometry-to-event feedback disabled.  This is the causal ablation for
    attributing gains to bidirectional coupling; it is not the serial
    post-update baseline.

The normalized flow coordinate is a generation parameter, not physical time.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .conserved_be import (
    ConservationProjector,
    pack_be,
    packed_size,
    unpack_be,
    upper_triangle_indices,
)
from .event_flow import ConservedEventFlow
from .geometry_flow import EquivariantGeometryFlow


JOINT_UNIDIRECTIONAL_MODE = "joint_unidirectional"
CONTROL_MODES = ("both_off", "serial_independent", JOINT_UNIDIRECTIONAL_MODE, "joint_bidirectional")


def _normalize_tau(
    tau: float | Tensor,
    batch: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Normalize scalar or per-sample flow time to finite ``[B, 1]``."""

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


def _atoms_from_packed_size(features: int) -> int:
    if type(features) is not int or features < 1:
        raise ValueError("event feature count must be positive")
    atoms = int((math.isqrt(8 * features + 1) - 1) // 2)
    if packed_size(atoms) != features:
        raise ValueError("event projector width must be a packed symmetric atom-pair size")
    return atoms


def _finite_strength(value: float, name: str, *, allow_zero: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite scalar")
    normalized = float(value)
    if normalized < 0 or (not allow_zero and normalized <= 0):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return normalized


class JointFlowOutput:
    """Velocity fields and the messages actually applied between branches."""

    def __init__(
        self,
        event_velocity: Tensor,
        geometry_velocity: Tensor,
        event_message: Tensor,
        geometry_message: Tensor,
    ) -> None:
        self.event_velocity = event_velocity
        self.geometry_velocity = geometry_velocity
        self.event_message = event_message
        self.geometry_message = geometry_message


class JointEventGeometryFlow(nn.Module):
    """Packed pair-event and equivariant geometry field with shared controls."""

    def __init__(
        self,
        event_projector: ConservationProjector,
        node_feature_dim: int,
        *,
        condition_feature_dim: int = 0,
        hidden_dim: int = 64,
        radial_features: int = 16,
    ) -> None:
        super().__init__()
        if any(type(value) is not int or value < 1 for value in (node_feature_dim, hidden_dim, radial_features)):
            raise ValueError("node, hidden, and radial feature widths must be positive integers")
        if type(condition_feature_dim) is not int or condition_feature_dim < 0:
            raise ValueError("condition_feature_dim must be a nonnegative integer")
        self.register_buffer("constraint_matrix", event_projector.constraint_matrix.detach().clone())
        self.tolerance = event_projector.tolerance
        self.event_dim = event_projector.n_features
        self.n_atoms = _atoms_from_packed_size(self.event_dim)
        self.node_feature_dim = node_feature_dim
        self.condition_feature_dim = condition_feature_dim
        self.hidden_dim = hidden_dim
        self.radial_features = radial_features

        self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)
        self.time_to_hidden = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.time_to_node = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, node_feature_dim))
        if condition_feature_dim:
            self.condition_to_hidden: nn.Module | None = nn.Sequential(
                nn.Linear(condition_feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
            )
            self.condition_to_node: nn.Module | None = nn.Sequential(
                nn.Linear(condition_feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, node_feature_dim)
            )
        else:
            self.condition_to_hidden = None
            self.condition_to_node = None

        # Static pair features are symmetric under i<->j: h_i+h_j,
        # |h_i-h_j|, the current BE value, and a diagonal indicator.
        self.event_base = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.geometry_to_event = nn.Sequential(
            nn.Linear(radial_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.event_to_node = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_feature_dim),
        )
        self.geometry = EquivariantGeometryFlow(
            node_feature_dim,
            hidden_dim=hidden_dim,
            radial_features=radial_features,
        )

    @property
    def event_flow(self) -> ConservedEventFlow:
        return ConservedEventFlow(ConservationProjector(self.constraint_matrix, self.tolerance))

    @property
    def event_projector(self) -> ConservationProjector:
        return ConservationProjector(self.constraint_matrix, self.tolerance)

    def _validate_inputs(
        self,
        event_state: Tensor,
        coordinates: Tensor,
        node_features: Tensor,
        atom_mask: Tensor,
        condition_features: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if event_state.ndim != 2 or event_state.shape[1] != self.event_dim or not event_state.is_floating_point():
            raise ValueError("event_state must be floating [B,F]")
        if coordinates.ndim != 3 or coordinates.shape[1:] != (self.n_atoms, 3) or not coordinates.is_floating_point():
            raise ValueError("coordinates must be floating [B,N,3] matching the packed event state")
        batch = coordinates.shape[0]
        if event_state.shape[0] != batch:
            raise ValueError("event_state and coordinates batch dimensions must match")
        if node_features.shape != (batch, self.n_atoms, self.node_feature_dim):
            raise ValueError("node_features must match coordinates and declared width")
        if node_features.dtype != coordinates.dtype or node_features.device != coordinates.device:
            raise ValueError("node_features must match coordinates dtype and device")
        if event_state.dtype != coordinates.dtype or event_state.device != coordinates.device:
            raise ValueError("event_state must match coordinates dtype and device")
        if atom_mask.shape != (batch, self.n_atoms) or atom_mask.dtype is not torch.bool or atom_mask.device != coordinates.device:
            raise ValueError("atom_mask must be boolean [B,N] on the coordinate device")
        if not bool(atom_mask.any(dim=1).all()):
            raise ValueError("each sample must contain at least one active atom")
        if not bool(torch.isfinite(event_state).all()) or not bool(torch.isfinite(coordinates).all()) or not bool(torch.isfinite(node_features).all()):
            raise ValueError("flow inputs must be finite")

        if self.condition_feature_dim:
            if condition_features is None or condition_features.shape != (batch, self.condition_feature_dim):
                raise ValueError("condition_features must be [B,C] for the configured condition width")
            if condition_features.dtype != coordinates.dtype or condition_features.device != coordinates.device:
                raise ValueError("condition_features must match coordinates dtype and device")
            if not bool(torch.isfinite(condition_features).all()):
                raise ValueError("condition_features must be finite")
        elif condition_features is not None:
            raise ValueError("condition_features were supplied to a model configured without them")

        rows, cols = upper_triangle_indices(self.n_atoms, device=coordinates.device)
        active_event = atom_mask[:, rows] & atom_mask[:, cols]
        if bool((event_state.masked_select(~active_event).abs() > self.tolerance).any()):
            raise ValueError("padded event-state entries must be zero")
        return active_event, condition_features if condition_features is not None else event_state.new_zeros((batch, 0))

    def _conditioned_nodes(
        self,
        node_features: Tensor,
        tau_value: Tensor,
        condition_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        hidden = self.node_encoder(node_features) + self.time_to_hidden(tau_value)[:, None, :]
        nodes = node_features + self.time_to_node(tau_value)[:, None, :]
        if self.condition_to_hidden is not None and self.condition_to_node is not None:
            hidden = hidden + self.condition_to_hidden(condition_features)[:, None, :]
            nodes = nodes + self.condition_to_node(condition_features)[:, None, :]
        return hidden, nodes

    def _pair_masks(self, atom_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        valid = atom_mask[:, :, None] & atom_mask[:, None, :]
        diagonal = torch.eye(self.n_atoms, dtype=torch.bool, device=atom_mask.device)[None]
        return valid, diagonal, valid & ~diagonal

    def _event_velocity(
        self,
        event_state: Tensor,
        coordinates: Tensor,
        hidden_nodes: Tensor,
        atom_mask: Tensor,
        active_event: Tensor,
        *,
        geometry_to_event_strength: float,
    ) -> tuple[Tensor, Tensor]:
        event_matrix = unpack_be(event_state, self.n_atoms)
        sender = hidden_nodes[:, :, None, :].expand(-1, -1, self.n_atoms, -1)
        receiver = hidden_nodes[:, None, :, :].expand(-1, self.n_atoms, -1, -1)
        valid_pairs, diagonal, off_diagonal = self._pair_masks(atom_mask)
        pair_input = torch.cat(
            (
                sender + receiver,
                (sender - receiver).abs(),
                event_matrix[..., None],
                diagonal.expand_as(event_matrix)[..., None].to(dtype=event_state.dtype),
            ),
            dim=-1,
        )
        base_pair = self.event_base(pair_input).squeeze(-1)
        base_pair = torch.where(valid_pairs, base_pair, torch.zeros_like(base_pair))

        displacement = coordinates[:, :, None, :] - coordinates[:, None, :, :]
        distance = torch.sqrt(displacement.square().sum(dim=-1) + 1e-12)
        centers = self.geometry.radial_centers.to(dtype=coordinates.dtype, device=coordinates.device)
        width = self.geometry.radial_width.to(dtype=coordinates.dtype, device=coordinates.device)
        radial = torch.exp(-((distance[..., None] - centers) / width).square())
        geometry_pair = self.geometry_to_event(radial).squeeze(-1)
        geometry_pair = torch.where(off_diagonal, geometry_pair, torch.zeros_like(geometry_pair))

        base_packed = pack_be(base_pair)
        geometry_packed = pack_be(geometry_pair) * geometry_to_event_strength
        projector = self.event_projector
        event_message = projector.project_masked(geometry_packed, active_event)
        event_velocity = projector.project_masked(base_packed + geometry_packed, active_event)
        return event_velocity, event_message

    def _geometry_velocity(
        self,
        event_state: Tensor,
        coordinates: Tensor,
        conditioned_nodes: Tensor,
        atom_mask: Tensor,
        *,
        event_to_geometry_strength: float,
    ) -> tuple[Tensor, Tensor]:
        event_matrix = unpack_be(event_state, self.n_atoms)
        valid_pairs, diagonal, _ = self._pair_masks(atom_mask)
        event_pair_input = torch.stack(
            (event_matrix, diagonal.expand_as(event_matrix).to(dtype=event_matrix.dtype)),
            dim=-1,
        )
        pair_context = self.event_to_node(event_pair_input)
        pair_context = torch.where(valid_pairs[..., None], pair_context, torch.zeros_like(pair_context))
        counts = valid_pairs.sum(dim=2, keepdim=True).clamp_min(1).to(dtype=event_matrix.dtype)
        event_context = pair_context.sum(dim=2) / counts
        applied_context = event_context * event_to_geometry_strength
        geometry_nodes = conditioned_nodes + applied_context
        pair_features = torch.where(valid_pairs, event_matrix, torch.zeros_like(event_matrix))
        pair_features = pair_features * event_to_geometry_strength
        geometry_velocity = self.geometry(
            coordinates,
            geometry_nodes,
            atom_mask,
            pair_features=pair_features,
        )
        return geometry_velocity, applied_context

    def forward(
        self,
        event_state: Tensor,
        coordinates: Tensor,
        node_features: Tensor,
        atom_mask: Tensor,
        *,
        coupling_strength: float = 1.0,
        tau: float | Tensor = 0.0,
        condition_features: Tensor | None = None,
        geometry_to_event_strength: float | None = None,
        event_to_geometry_strength: float | None = None,
    ) -> JointFlowOutput:
        """Evaluate the bidirectional control with a shared coupling strength."""

        strength = _finite_strength(coupling_strength, "coupling_strength")
        geometry_strength = strength if geometry_to_event_strength is None else _finite_strength(
            geometry_to_event_strength, "geometry_to_event_strength"
        )
        event_strength = strength if event_to_geometry_strength is None else _finite_strength(
            event_to_geometry_strength, "event_to_geometry_strength"
        )
        active_event, conditions = self._validate_inputs(
            event_state, coordinates, node_features, atom_mask, condition_features
        )
        tau_value = _normalize_tau(
            tau,
            event_state.shape[0],
            dtype=event_state.dtype,
            device=event_state.device,
        )
        hidden_nodes, conditioned_nodes = self._conditioned_nodes(node_features, tau_value, conditions)
        event_velocity, event_message = self._event_velocity(
            event_state,
            coordinates,
            hidden_nodes,
            atom_mask,
            active_event,
            geometry_to_event_strength=geometry_strength,
        )
        geometry_velocity, geometry_message = self._geometry_velocity(
            event_state,
            coordinates,
            conditioned_nodes,
            atom_mask,
            event_to_geometry_strength=event_strength,
        )
        return JointFlowOutput(event_velocity, geometry_velocity, event_message, geometry_message)

    def forward_control(
        self,
        mode: str,
        event_state: Tensor,
        coordinates: Tensor,
        node_features: Tensor,
        atom_mask: Tensor,
        *,
        tau: float | Tensor,
        dt: float | None = None,
        coupling_strength: float = 1.0,
        condition_features: Tensor | None = None,
    ) -> JointFlowOutput:
        """Evaluate one registered learned control using this exact parameter set."""

        if mode not in CONTROL_MODES:
            raise ValueError(f"unsupported control mode: {mode}")
        if mode == "joint_bidirectional":
            return self(
                event_state,
                coordinates,
                node_features,
                atom_mask,
                coupling_strength=coupling_strength,
                tau=tau,
                condition_features=condition_features,
            )
        if mode == JOINT_UNIDIRECTIONAL_MODE:
            # Keep the same state time point and integrator contract as the
            # bidirectional field.  Only one message channel is disabled.
            if dt is None:
                raise ValueError("joint_unidirectional requires an explicit positive dt")
            _finite_strength(dt, "dt", allow_zero=False)
            return self(
                event_state,
                coordinates,
                node_features,
                atom_mask,
                coupling_strength=coupling_strength,
                tau=tau,
                condition_features=condition_features,
                geometry_to_event_strength=0.0,
                event_to_geometry_strength=coupling_strength,
            )
        if mode == "both_off":
            return self(
                event_state,
                coordinates,
                node_features,
                atom_mask,
                coupling_strength=0.0,
                tau=tau,
                condition_features=condition_features,
            )

        step = _finite_strength(dt, "dt", allow_zero=False) if dt is not None else None
        if step is None:
            raise ValueError("serial_independent requires an explicit positive dt")
        strength = _finite_strength(coupling_strength, "coupling_strength")
        active_event, conditions = self._validate_inputs(
            event_state, coordinates, node_features, atom_mask, condition_features
        )
        tau_value = _normalize_tau(
            tau,
            event_state.shape[0],
            dtype=event_state.dtype,
            device=event_state.device,
        )
        hidden_nodes, conditioned_nodes = self._conditioned_nodes(node_features, tau_value, conditions)
        event_velocity, _ = self._event_velocity(
            event_state,
            coordinates,
            hidden_nodes,
            atom_mask,
            active_event,
            geometry_to_event_strength=0.0,
        )
        event_state_next = event_state + step * event_velocity
        geometry_velocity, geometry_message = self._geometry_velocity(
            event_state_next,
            coordinates,
            conditioned_nodes,
            atom_mask,
            event_to_geometry_strength=strength,
        )
        return JointFlowOutput(
            event_velocity,
            geometry_velocity,
            torch.zeros_like(event_velocity),
            geometry_message,
        )


class SerialEventGeometryFlow(nn.Module):
    """One-way event-to-geometry view over a shared joint parameter set."""

    def __init__(
        self,
        event_projector: ConservationProjector | None = None,
        node_feature_dim: int | None = None,
        *,
        condition_feature_dim: int = 0,
        hidden_dim: int = 64,
        radial_features: int = 16,
        joint_model: JointEventGeometryFlow | None = None,
    ) -> None:
        super().__init__()
        if joint_model is not None:
            if event_projector is not None or node_feature_dim is not None:
                raise ValueError("supply either joint_model or constructor dimensions, not both")
            self.joint = joint_model
        else:
            if event_projector is None or node_feature_dim is None:
                raise ValueError("event_projector and node_feature_dim are required without joint_model")
            self.joint = JointEventGeometryFlow(
                event_projector,
                node_feature_dim,
                condition_feature_dim=condition_feature_dim,
                hidden_dim=hidden_dim,
                radial_features=radial_features,
            )

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
    ) -> JointFlowOutput:
        return self.joint.forward_control(
            "serial_independent",
            event_state,
            coordinates,
            node_features,
            atom_mask,
            tau=tau,
            dt=dt,
            coupling_strength=coupling_strength,
            condition_features=condition_features,
        )
