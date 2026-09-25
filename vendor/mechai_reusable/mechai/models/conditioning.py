"""Zero-initialized scalar-condition residuals for coordinate-noise predictors.

This module neither creates atoms nor labels reactive solvent. Its only geometry
input is the *current noisy* coordinate tensor. All additional features must be
rotation-invariant scalars with an explicit provenance. The pairwise construction
is O(3)-equivariant (hence SE(3)-equivariant); it is not a new chirality mechanism.
The surrounding baseline retains its own graph, stereochemical and diffusion
conditions. No energy, force, transition-state or attention labels are inferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor, nn


class ConditionSource(str, Enum):
    """Provenance classes, not claims that a condition is experimentally known.

    INPUT_GRAPH reuses the graph supplied to the baseline. In UniTS-Lib that graph
    can itself derive from a reference TS; record this upstream origin in the run.
    INDEPENDENT_METADATA requires a separately audited observation/descriptor.
    REFERENCE_TS_ORACLE denotes *additional* target-derived information and is
    rejected by default, so an oracle cannot silently become a deployment input.
    """

    INPUT_GRAPH = "provided_input_graph"
    INDEPENDENT_METADATA = "independent_metadata"
    REFERENCE_TS_ORACLE = "reference_ts_oracle"


@dataclass(frozen=True)
class EnvironmentCondition:
    """Per-atom scalar conditions, in exactly the baseline's atom ordering.

    ``values`` has shape [batch, atoms, condition_dim]; it contains scalar graph
    embeddings or explicitly encoded descriptors, never Cartesian coordinates.
    ``observed_mask`` has shape [batch, atoms] or [batch, atoms, 1]. False denotes
    missing conditioning information, *not* a deleted/inactive solvent molecule.
    All atoms still receive baseline predictions and remain in the output.
    Metadata alone cannot prove provenance: the data pipeline must audit it.
    """

    values: Tensor
    observed_mask: Tensor
    source: ConditionSource


def _mask(mask: Tensor, shape: tuple[int, int], device: torch.device, name: str) -> Tensor:
    """Accept official float masks but reject soft weights or mismatched padding."""
    if mask.shape == (*shape, 1):
        mask = mask.squeeze(-1)
    if mask.shape != shape or mask.device != device:
        raise ValueError(f"{name} must have shape {shape} on device {device}")
    if mask.dtype != torch.bool and not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError(f"{name} must be boolean or contain only zero and one")
    return mask.bool()


class EquivariantEnvironmentResidual(nn.Module):
    """Predict an additive, zero-center coordinate-noise correction.

    For receiver i and observed context atom j, a scalar MLP uses node features,
    context features, diffusion time and current pair distance to weight
    (x_i-x_j)/(1+distance). This is permutation-equivariant and independent of
    the overall coordinate origin. All real atoms remain receivers, including
    atoms without observed context. A per-sample mean removal preserves UniTS's
    zero-center noise subspace. Pair coefficients have no solvent-activity meaning.

    The final scalar readout is zero-initialized: enabling this module initially
    adds exactly zero. Its readout receives gradients on the first training step;
    earlier layers receive nonzero gradients once the readout has moved from zero.
    No sampler, noise schedule, training target or baseline weights are changed.

    Args:
        node_dim: Width of baseline invariant node features (e.g. xh[..., 3:]).
        condition_dim: Width of additional/provided invariant scalar conditions.
        hidden_dim: Internal scalar channel width.
        radial_features: Number of Gaussian distance basis functions.
        radial_scale: Positive distance scale in the baseline's normalized units.
            This is an RBF scale, not an edge cutoff; distant atoms are retained.
        query_chunk_size: Bounds the number of receiver atoms materialized at once.
        allow_reference_ts_oracle: Explicit opt-in for separately reported oracle
            diagnostics, never a claim that the inputs are independently available.
    """

    def __init__(
        self,
        node_dim: int,
        condition_dim: int,
        hidden_dim: int = 32,
        radial_features: int = 16,
        radial_scale: float = 8.0,
        query_chunk_size: int = 64,
        allow_reference_ts_oracle: bool = False,
    ) -> None:
        super().__init__()
        if min(node_dim, condition_dim, hidden_dim, radial_features, query_chunk_size) < 1:
            raise ValueError("feature widths and query_chunk_size must be positive")
        if not 0 < radial_scale < float("inf"):
            raise ValueError("radial_scale must be positive and finite")
        self.node_dim = node_dim
        self.condition_dim = condition_dim
        self.query_chunk_size = query_chunk_size
        self.allow_reference_ts_oracle = allow_reference_ts_oracle
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.condition_encoder = nn.Linear(condition_dim, hidden_dim)
        self.time_encoder = nn.Linear(1, hidden_dim)
        self.pair_encoder = nn.Sequential(
            nn.Linear(3 * hidden_dim + radial_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.readout = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.zeros_(self.readout.weight)
        self.register_buffer("radial_centers", torch.linspace(0, radial_scale, radial_features))
        self.register_buffer("radial_width", torch.tensor(radial_scale / radial_features))

    def forward(
        self,
        noisy_coordinates: Tensor,
        node_features: Tensor,
        timestep: Tensor,
        node_mask: Tensor,
        condition: EnvironmentCondition | None = None,
    ) -> Tensor:
        """Return [B,N,3] residual; None/missing conditions return exact zeros.

        Coordinates and all feature tensors must be floating point with the same
        dtype/device; they are not silently rescaled or reordered. ``timestep``
        is the baseline's scalar or [B]/[B,1] diffusion time. Padded coordinates
        and missing features may be arbitrary, including NaN: they are replaced
        before arithmetic. Non-finite *observed* values fail explicitly.
        """
        x = noisy_coordinates
        if x.ndim != 3 or x.shape[-1] != 3 or not x.is_floating_point():
            raise ValueError("noisy_coordinates must be floating point [B,N,3]")
        batch_size, n_atoms, _ = x.shape
        valid = _mask(node_mask, (batch_size, n_atoms), x.device, "node_mask")
        if not bool(valid.any(dim=1).all()):
            raise ValueError("each sample must contain at least one real atom")
        if condition is None:
            return torch.zeros_like(x)
        if not isinstance(condition.source, ConditionSource):
            raise ValueError("condition.source must explicitly use ConditionSource")
        if condition.source == ConditionSource.REFERENCE_TS_ORACLE and not self.allow_reference_ts_oracle:
            raise ValueError("reference-TS oracle conditions require explicit opt-in")
        observed = _mask(condition.observed_mask, (batch_size, n_atoms), x.device, "observed_mask")
        if bool((observed & ~valid).any()):
            raise ValueError("observed condition cannot refer to a padded atom")
        for name, value, width in (
            ("node_features", node_features, self.node_dim),
            ("condition.values", condition.values, self.condition_dim),
        ):
            if value.shape != (batch_size, n_atoms, width):
                raise ValueError(f"{name} must have shape {(batch_size, n_atoms, width)}")
            if value.device != x.device or value.dtype != x.dtype:
                raise ValueError(f"{name} must match coordinate dtype and device")
        if not bool(observed.any()):
            return torch.zeros_like(x)
        for name, value in (
            ("real coordinates", x[valid]),
            ("real node features", node_features[valid]),
            ("observed condition features", condition.values[observed]),
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")
        if timestep.device != x.device or timestep.dtype != x.dtype:
            raise ValueError("timestep must match coordinate dtype and device")
        if timestep.numel() == 1:
            time = timestep.reshape(1, 1).expand(batch_size, 1)
        elif timestep.shape in ((batch_size,), (batch_size, 1)):
            time = timestep.reshape(batch_size, 1)
        else:
            raise ValueError("timestep must be scalar, [B] or [B,1]")
        if not bool(torch.isfinite(time).all()):
            raise ValueError("timestep must be finite")

        # Sanitize before distances/MLPs: multiplication by zero does not remove NaN.
        x = torch.where(valid[..., None], x, 0)
        h = torch.where(valid[..., None], node_features, 0)
        c = torch.where(observed[..., None], condition.values, 0)
        h = torch.nn.functional.silu(self.node_encoder(h) + self.time_encoder(time)[:, None, :])
        c = torch.nn.functional.silu(self.condition_encoder(c))
        sender_ids = torch.arange(n_atoms, device=x.device)
        pieces = []
        for start in range(0, n_atoms, self.query_chunk_size):
            stop = min(start + self.query_chunk_size, n_atoms)
            count = stop - start
            rel = x[:, start:stop, None, :] - x[:, None, :, :]
            distance = torch.linalg.vector_norm(rel, dim=-1, keepdim=True)
            radial = torch.exp(-((distance - self.radial_centers) / self.radial_width).square())
            pair_features = torch.cat(
                (
                    h[:, start:stop, None, :].expand(-1, -1, n_atoms, -1),
                    h[:, None, :, :].expand(-1, count, -1, -1),
                    c[:, None, :, :].expand(-1, count, -1, -1),
                    radial,
                ),
                dim=-1,
            )
            pair_mask = valid[:, start:stop, None] & observed[:, None, :]
            pair_mask &= sender_ids[None, :] != sender_ids[start:stop, None]
            weights = self.readout(self.pair_encoder(pair_features)).squeeze(-1)
            weights = torch.where(pair_mask, weights, 0)
            # Divide by available context count, not padded length or batch size.
            normalizer = pair_mask.sum(dim=-1, keepdim=True).clamp_min(1)
            pieces.append((weights[..., None] * rel / (1 + distance)).sum(dim=2) / normalizer)
        residual = torch.cat(pieces, dim=1)
        residual = torch.where(valid[..., None], residual, 0)
        mean = residual.sum(dim=1, keepdim=True) / valid.sum(dim=1)[:, None, None]
        return torch.where(valid[..., None], residual - mean, 0)


class UniTSConditionedDynamics(nn.Module):
    """Wrap the official ``dynamics._forward`` contract without importing UniTS.

    The existing padded ``context`` is used as per-atom scalar conditioning.
    Thus this ready-to-connect mode changes *how existing input graph information
    is used*, not which independent environmental measurements are available.
    An independently audited condition can also be used through the residual's
    public interface in a later data adapter; it is not silently manufactured here.

    Load the official checkpoint into the unwrapped model first, then replace
    ``diffusion.dynamics`` with this module. Save adapted checkpoints separately:
    parameter names gain ``base_dynamics.`` and ``residual.`` prefixes. The wrapper
    does not freeze any parameters; the training harness must explicitly specify
    whether the backbone, graph encoders and/or adapter are optimized.
    """

    def __init__(
        self,
        base_dynamics: nn.Module,
        residual: EquivariantEnvironmentResidual,
        enabled: bool = True,
        condition_mode: str = "node",
    ) -> None:
        super().__init__()
        if condition_mode not in ("node", "pooled"):
            raise ValueError("condition_mode must be node or pooled")
        self.base_dynamics = base_dynamics
        self.residual = residual
        self.enabled = enabled
        self.condition_mode = condition_mode

    def _forward(
        self,
        xh: Tensor,
        t: Tensor,
        node_mask: Tensor,
        edge_mask: Tensor,
        context: Tensor | None,
        batch: Tensor,
    ) -> Tensor:
        """Preserve the official arguments, original output and all atom indices."""
        prediction = self.base_dynamics._forward(xh, t, node_mask, edge_mask, context, batch)
        if not self.enabled or context is None:
            return prediction
        if prediction.shape != xh.shape[:2] + (3,):
            raise ValueError("the wrapped model must predict coordinate noise [B,N,3]")
        values = context
        if self.condition_mode == "pooled":
            # A parameter-matched control removes atom-specific context from
            # the residual only. The frozen backbone still receives its exact
            # original context; all atoms and the same 6,208 parameters remain.
            valid = _mask(node_mask, context.shape[:2], context.device, "node_mask")
            if not bool(valid.any(dim=1).all()):
                raise ValueError("each sample must contain a real atom")
            clean = torch.where(valid[..., None], context, 0)
            mean = clean.sum(dim=1, keepdim=True) / valid.sum(dim=1)[:, None, None]
            values = mean.expand_as(context)
        condition = EnvironmentCondition(values, node_mask, ConditionSource.INPUT_GRAPH)
        delta = self.residual(xh[..., :3], xh[..., 3:], t, node_mask, condition)
        return prediction + delta

    forward = _forward
