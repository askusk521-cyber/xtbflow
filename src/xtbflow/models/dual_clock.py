"""Independent generation clocks and state-derived local bidirectional messages.

This class is a new trainable architecture. Archived research checkpoints use
their pinned model and are not silently interpreted as these weights.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .conserved_be import unpack_be
from .joint_flow import JointEventGeometryFlow, JointFlowOutput, _finite_strength, _normalize_tau


class DualClockJointFlow(JointEventGeometryFlow):
    """Two-time extension with exact synchronous/global/shared-weight control.

    Each branch sees its own clock and the other clock's offset. Offset maps
    have no bias, so synchronous clocks exactly recover the inherited field.
    Local gates multiply cross-branch messages before conservation projection;
    they never multiply the final projected event velocity.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.clock_offset_to_event = nn.Linear(1, self.hidden_dim, bias=False)
        self.clock_offset_to_geometry = nn.Linear(1, self.node_feature_dim, bias=False)

    def local_gate(
        self, event_state: Tensor, reactant_event: Tensor, coordinates: Tensor,
        atom_mask: Tensor, *, radius_angstrom: float, event_scale: float,
    ) -> Tensor:
        """Symmetric contact-or-event-change gate from current and reactant state.

        Contact uses a Gaussian of current interatomic distance in angstrom.
        Change uses the absolute current-minus-reactant BE value (electrons).
        Both are invariants; no target, mode, endpoint, or chosen atom IDs enter.
        """
        radius = _finite_strength(radius_angstrom, "radius_angstrom", allow_zero=False)
        scale = _finite_strength(event_scale, "event_scale", allow_zero=False)
        if (reactant_event.shape != event_state.shape or reactant_event.dtype != event_state.dtype
                or reactant_event.device != event_state.device or not bool(torch.isfinite(reactant_event).all())):
            raise ValueError("reactant_event must match the finite current event state")
        valid, _, _ = self._pair_masks(atom_mask)
        reference = unpack_be(reactant_event, self.n_atoms)
        if bool((reference.masked_select(~valid).abs() > self.tolerance).any()):
            raise ValueError("padded reactant event entries must be zero")
        change = (unpack_be(event_state, self.n_atoms) - reference).abs()
        distance2 = (coordinates[:, :, None] - coordinates[:, None, :]).square().sum(-1)
        contact = torch.exp(-distance2 / (2 * radius * radius))
        changed = -torch.expm1(-change / scale)
        return torch.where(valid, 1 - (1 - contact) * (1 - changed), torch.zeros_like(change))

    def forward_dual(
        self, event_state: Tensor, coordinates: Tensor, node_features: Tensor, atom_mask: Tensor,
        *, event_tau: float | Tensor, geometry_tau: float | Tensor,
        condition_features: Tensor | None = None, reactant_event: Tensor | None = None,
        gate: str = "global", radius_angstrom: float = 3.0, event_scale: float = 1.0,
        geometry_to_event_strength: float = 1.0, event_to_geometry_strength: float = 1.0,
    ) -> JointFlowOutput:
        active, conditions = self._validate_inputs(event_state, coordinates, node_features, atom_mask, condition_features)
        tb = _normalize_tau(event_tau, len(event_state), dtype=event_state.dtype, device=event_state.device)
        tx = _normalize_tau(geometry_tau, len(event_state), dtype=event_state.dtype, device=event_state.device)
        g2e = _finite_strength(geometry_to_event_strength, "geometry_to_event_strength")
        e2g = _finite_strength(event_to_geometry_strength, "event_to_geometry_strength")
        hidden, _ = self._conditioned_nodes(node_features, tb, conditions)
        _, nodes = self._conditioned_nodes(node_features, tx, conditions)
        hidden = hidden + self.clock_offset_to_event(tx - tb)[:, None]
        nodes = nodes + self.clock_offset_to_geometry(tb - tx)[:, None]
        if gate == "global":
            pair_gate = None
        elif gate == "local" and reactant_event is not None:
            pair_gate = self.local_gate(event_state, reactant_event, coordinates, atom_mask,
                                        radius_angstrom=radius_angstrom, event_scale=event_scale)
        else:
            raise ValueError("gate must be global or local with an explicit reactant event")
        vb, mb = self._event_velocity(event_state, coordinates, hidden, atom_mask, active,
            geometry_to_event_strength=g2e, conservation_projection=True, pair_gate=pair_gate)
        vx, mx = self._geometry_velocity(event_state, coordinates, nodes, atom_mask,
            event_to_geometry_strength=e2g, pair_gate=pair_gate)
        result = JointFlowOutput(vb, vx, mb, mx)
        result.pair_gate = pair_gate
        result.event_tau, result.geometry_tau = tb, tx
        return result
