"""One small event–geometry compatibility mechanism, with no physical energy claim.

A shared invariant scalar C(event, candidate coordinates) changes an external
prior through softmax(log prior + strength * C). Its coordinate derivative can
be supplied to a future sampler. The derivative is model-score feedback, not a
force, physical trajectory, conservative electronic energy, or TS guarantee.

No graph/geometry baseline is wrapped or modified here. Reference coordinates,
products and correspondence evidence are never manufactured as input features.
See ``python -m mechai.models.event_geometry`` for a CPU-only software example;
that example is neither training nor evidence of chemical/model effectiveness.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .conditioning import ConditionSource


class CandidateGeometrySource(str, Enum):
    """Declared origin of evaluated coordinates; source assertions need auditing.

    ``DECLARED_ENDPOINT`` is reserved for endpoint-conditioned diagnostics such
    as Transition1x.  Those coordinates are supplied R/P conditions, not
    generated TS candidates; the endpoint adapter keeps that task distinction
    in its surrounding record and never treats the reference TS as input.
    """

    GENERATED_CANDIDATE = "generated_candidate"
    INDEPENDENT_INITIAL_GUESS = "independent_initial_guess"
    DECLARED_ENDPOINT = "declared_endpoint"
    REFERENCE_TS_ORACLE = "reference_ts_oracle"


@dataclass(frozen=True)
class EventGeometryCondition:
    """Input-side candidate events; no coordinates or reference labels occur here.

    ``event_features`` is [B,E,F] of invariant scalar descriptors supplied from
    the audited current graph, proposed event, and explicitly given intent.
    It must not contain atom IDs/map numbers or target-TS descriptors.
    ``pair_edits`` is [B,E,N,N], symmetric, with zero diagonal: signed proposed
    bond-order edits in the shared atom ordering. Atom-local electron/charge
    changes require explicit scalar descriptors, not a diagonal bond edit.
    ``event_valid_mask`` [B,E] encodes structural eligibility, not whether a
    reference confirms the event. Unknown correspondence is retained separately
    in CompatibilityEvidence. Proposal generation and conservation checks are
    external; this tensor interface does not establish chemical feasibility.
    """

    event_features: Tensor
    pair_edits: Tensor
    event_valid_mask: Tensor
    source: ConditionSource


@dataclass(frozen=True)
class CandidateGeometry:
    """Coordinates evaluated against the event pool, distinct from conditioning.

    ``coordinates`` is float32/float64 [B,G,N,3] in Angstrom with a common atom order
    across G geometries and E events. ``atom_mask`` is boolean [B,N]; it denotes
    padding only, never deletion of a real solvent/spectator. A common inventory
    per batch item is required. ``geometry_valid_mask`` [B,G] is computational
    validity, not evidence of a TS. All axes have at least one padded slot;
    batches with no eligible atoms/events/geometries safely return zero outputs.
    """

    coordinates: Tensor
    atom_mask: Tensor
    geometry_valid_mask: Tensor
    source: CandidateGeometrySource


@dataclass(frozen=True)
class CompatibilityScores:
    """Invariant logits [B,E,G]; invalid slots are zero and explicitly masked.

    Positive/negative scores have no standalone physical interpretation. The
    distance-only representation is O(3) invariant, including reflections, so
    external stereochemical and endpoint validation remain necessary.
    """

    values: Tensor
    valid_mask: Tensor
    is_oracle: bool


@dataclass(frozen=True)
class CompatibilityEvidence:
    """Optional reference correspondence labels, never an inference condition.

    ``targets`` and boolean ``observed_mask`` are [B,E,G]. Observed targets are
    0 or 1 under a declared reference correspondence protocol. A zero means
    mismatch to that reference, not a physically impossible channel. Unobserved
    targets may be NaN; optional positive ``weights`` of the same shape can
    express externally fixed group weights. Evidence does not define candidate
    validity and is never consumed by forward/couple.
    """

    targets: Tensor
    observed_mask: Tensor
    weights: Tensor | None = None


@dataclass(frozen=True)
class CoupledProposal:
    """An event posterior per geometry and score feedback for an external sampler.

    ``posterior`` and ``valid_mask`` have shape [B,E,G]. ``coordinate_feedback``
    is [B,G,N,3], the derivative of logsumexp_e(log prior + strength*C); strength
    is already included and must not be multiplied in a second time. No input
    coordinates are mutated. At strength zero, scores is None because the
    compatibility network is bypassed, the posterior equals the masked prior,
    and coordinate feedback is exactly zero. A zero posterior denotes no valid
    prior-supported candidates, never a normalized chemical distribution.
    """

    posterior: Tensor
    coordinate_feedback: Tensor
    valid_mask: Tensor
    scores: CompatibilityScores | None


def _boolean_mask(value, shape, device, name):
    if value.shape != shape or value.dtype != torch.bool or value.device != device:
        raise ValueError(f"{name} must be boolean {shape} on the coordinate device")


def _masked_distribution(logits: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Softmax/log-partition over E; all-invalid columns are differentiable zero."""
    any_valid = valid.any(dim=1, keepdim=True)
    masked = torch.where(valid, logits, -torch.inf)
    # Softmax([-inf,...]) would be NaN; empty columns get a harmless intermediate
    # distribution which is masked back to zero together with its derivatives.
    safe = torch.where(any_valid, masked, 0)
    posterior = torch.where(valid, torch.softmax(safe, dim=1), 0)
    log_partition = torch.where(any_valid.squeeze(1), torch.logsumexp(safe, dim=1), 0)
    return posterior, log_partition


class EventGeometryCompatibility(nn.Module):
    """Single scalar MLP over edit-conditioned distance summaries.

    Signed/magnitude edit-weighted RBF distances encode proposed changed pairs;
    distances from event atoms to all real atoms retain a minimal environment
    view. For a no-edit candidate the context view uses all distinct atom pairs.
    These three pooled RBF vectors plus event features and two log-counts feed
    one shared MLP. No separate event head, geometry head, force head, energy
    potential or flow is introduced. No bond thresholds or preferred TS bond
    lengths are built in.

    Args:
        event_feature_dim: F, width of independently supplied scalar descriptors.
        hidden_dim: H of the sole scalar MLP; default 32.
        radial_features: R Gaussian distance channels; default 12.
        radial_scale: RBF-center extent in Angstrom, not a truncation cutoff.
        distance_epsilon: Positive smoothing length for finite collision
            derivatives; default 1e-6 Angstrom. Does not repair collisions.
        allow_reference_ts_oracle: Explicit opt-in for separately identified
            oracle diagnostics. It cannot certify truth of declared provenance.

    This prototype supports float32/float64 only. Half and bfloat16 are
    rejected: their distance smoothing and derivative precision have not been
    validated. Mixed-precision integration is outside this prototype.

    Parameter count is (F + 3*R + 2)*H + 2*H + 1: 1,409 for F=4,H=32,R=12.
    Pair processing uses O(B*G*N^2*R + B*E*N^2) intermediates, suitable for a
    bounded prototype, not unbounded solvent boxes. Standard forward is fully
    differentiable; ``couple`` optionally retains derivative graphs for tests
    or future differentiation through feedback.
    """

    def __init__(self, event_feature_dim: int, hidden_dim: int = 32,
                 radial_features: int = 12, radial_scale: float = 8.0,
                 distance_epsilon: float = 1e-6,
                 allow_reference_ts_oracle: bool = False):
        super().__init__()
        if any(type(v) is not int or v < 1 for v in (event_feature_dim, hidden_dim, radial_features)):
            raise ValueError("Feature and hidden widths must be positive integers")
        if any(not math.isfinite(v) or v <= 0 for v in (radial_scale, distance_epsilon)):
            raise ValueError("Distance scales must be finite and positive")
        self.event_feature_dim = event_feature_dim
        self.distance_epsilon = distance_epsilon
        self.allow_reference_ts_oracle = allow_reference_ts_oracle
        self.readout = nn.Sequential(nn.Linear(event_feature_dim + 3 * radial_features + 2, hidden_dim),
                                     nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.register_buffer("radial_centers", torch.linspace(0, radial_scale, radial_features))
        self.register_buffer("radial_width", torch.tensor(radial_scale / max(radial_features - 1, 1)))

    def _validate(self, condition, geometry):
        x = geometry.coordinates
        if x.ndim != 4 or x.shape[-1] != 3 or not x.is_floating_point() or min(x.shape[:-1]) < 1:
            raise ValueError("Coordinates must be nonempty floating [B,G,N,3]")
        if x.dtype not in (torch.float32, torch.float64):
            raise ValueError("Compatibility coordinates support only float32/float64; half/bfloat16 are not supported")
        # Even a supported dtype can underflow a user-supplied smoothing
        # length. Require a normal finite squared length before sqrt/autograd.
        limits = torch.finfo(x.dtype)
        if not math.sqrt(limits.tiny) <= self.distance_epsilon <= math.sqrt(limits.max):
            raise ValueError("distance_epsilon squared must be finite and normal in the coordinate dtype")
        b, g, n, _ = x.shape
        features = condition.event_features
        if features.ndim != 3 or features.shape[0] != b or features.shape[1] < 1 or features.shape[2] != self.event_feature_dim:
            raise ValueError("event_features must be [B,E,event_feature_dim] with nonempty E")
        e = features.shape[1]
        _boolean_mask(geometry.atom_mask, (b, n), x.device, "atom_mask")
        _boolean_mask(geometry.geometry_valid_mask, (b, g), x.device, "geometry_valid_mask")
        _boolean_mask(condition.event_valid_mask, (b, e), x.device, "event_valid_mask")
        if condition.pair_edits.shape != (b, e, n, n):
            raise ValueError("pair_edits must be [B,E,N,N] in the coordinate atom order")
        for name, value in (("event_features", features), ("pair_edits", condition.pair_edits)):
            if value.dtype != x.dtype or value.device != x.device:
                raise ValueError(f"{name} must match coordinate dtype/device")
        if self.radial_centers.dtype != x.dtype or self.radial_centers.device != x.device:
            raise ValueError("Model dtype/device must match coordinates")
        if not isinstance(condition.source, ConditionSource) or not isinstance(geometry.source, CandidateGeometrySource):
            raise ValueError("Explicit ConditionSource and CandidateGeometrySource are required")
        oracle = (condition.source == ConditionSource.REFERENCE_TS_ORACLE or
                  geometry.source == CandidateGeometrySource.REFERENCE_TS_ORACLE)
        if oracle and not self.allow_reference_ts_oracle:
            raise ValueError("Reference-TS oracle provenance requires explicit opt-in")
        atoms = geometry.atom_mask
        geometric = geometry.geometry_valid_mask & atoms.any(dim=-1)[:, None]
        valid = condition.event_valid_mask[:, :, None] & geometric[:, None, :]
        observed_x = geometric[..., None] & atoms[:, None, :]
        if not bool(torch.isfinite(x[observed_x]).all()):
            raise ValueError("Valid candidate coordinates must be finite")
        if not bool(torch.isfinite(features[condition.event_valid_mask]).all()):
            raise ValueError("Valid event features must be finite")
        real_pairs = atoms[:, None, :, None] & atoms[:, None, None, :]
        edit_valid = condition.event_valid_mask[:, :, None, None] & real_pairs
        edits = torch.where(edit_valid, condition.pair_edits, 0)
        if not bool(torch.isfinite(edits).all()):
            raise ValueError("Valid pair edits must be finite")
        if not torch.allclose(edits, edits.transpose(-1, -2), atol=1e-7, rtol=1e-7):
            raise ValueError("Pair edits must be symmetric")
        if bool((edits.diagonal(dim1=-2, dim2=-1) != 0).any()):
            raise ValueError("Pair edits must have zero diagonal; atom-local changes need explicit features")
        return valid, geometric, observed_x, edits, oracle

    def forward(self, condition: EventGeometryCondition, geometry: CandidateGeometry) -> CompatibilityScores:
        """Return a nonphysical invariant score for every event/geometry pair.

        Padded/invalid coordinates, edits and features may contain NaN. They
        are replaced before arithmetic; nonfinite valid entries fail explicitly.
        Missing correspondence evidence does not remove inference candidates.
        """
        valid, _, observed_x, edits, oracle = self._validate(condition, geometry)
        x = torch.where(observed_x[..., None], geometry.coordinates, 0)
        features = torch.where(condition.event_valid_mask[..., None], condition.event_features, 0)
        b, g, n, _ = x.shape
        displacement = x[:, :, :, None, :] - x[:, :, None, :, :]
        distance = torch.sqrt(displacement.square().sum(dim=-1) + self.distance_epsilon ** 2)
        radial = torch.exp(-((distance[..., None] - self.radial_centers) / self.radial_width).square())
        pair_mask = geometry.atom_mask[:, :, None] & geometry.atom_mask[:, None, :]
        pair_mask = pair_mask & ~torch.eye(n, dtype=torch.bool, device=x.device)[None]
        radial = torch.where(pair_mask[:, None, :, :, None], radial, 0)
        magnitude = edits.abs()
        edit_norm = magnitude.sum(dim=(-1, -2)).clamp_min(1)
        signed = torch.einsum("beij,bgijr->begr", edits, radial) / edit_norm[..., None, None]
        absolute = torch.einsum("beij,bgijr->begr", magnitude, radial) / edit_norm[..., None, None]
        event_atoms = magnitude.sum(dim=-1) > 0
        has_edits = event_atoms.any(dim=-1)
        context_atoms = torch.where(has_edits[..., None], event_atoms, geometry.atom_mask[:, None, :])
        context_pairs = context_atoms[..., :, None] & pair_mask[:, None, :, :]
        context = torch.einsum("beij,bgijr->begr", context_pairs.to(x.dtype), radial)
        context = context / context_pairs.sum(dim=(-1, -2)).clamp_min(1)[..., None, None]
        counts = torch.stack((magnitude.sum(dim=(-1, -2)) / 2,
                              event_atoms.sum(dim=-1).to(x.dtype)), dim=-1).log1p()
        pooled = torch.cat((features[:, :, None, :].expand(-1, -1, g, -1),
                            signed, absolute, context, counts[:, :, None, :].expand(-1, -1, g, -1)), dim=-1)
        values = self.readout(pooled).squeeze(-1)
        return CompatibilityScores(torch.where(valid, values, 0), valid, oracle)

    def couple(self, condition: EventGeometryCondition, geometry: CandidateGeometry,
               prior_logits: Tensor, *, coupling_strength: float = 0.0,
               create_graph: bool = False) -> CoupledProposal:
        """Combine an external prior with C and return its coordinate feedback.

        prior_logits [B,E] are independent of these candidate coordinates.
        Finite logits or -inf (zero prior support) are accepted at valid events;
        invalid slots may contain NaN. The prior is normalized over supported
        events before use. All-invalid masks or all-zero prior support yield
        zero posterior and feedback, with an explicit false valid_mask.

        strength=0 bypasses forward/autograd and gives exactly the masked prior
        on each valid geometry. For positive strength, this method enables
        gradients even inside torch.no_grad(); torch.inference_mode() is not
        supported because coordinate derivatives are part of the requested API.
        ``create_graph=False`` returns detached feedback while retaining the
        posterior's graph for optional downstream losses. No update step size,
        clipping, physical time, or sampling schedule is chosen here.
        """
        if not isinstance(coupling_strength, (int, float)) or not math.isfinite(coupling_strength) or coupling_strength < 0:
            raise ValueError("coupling_strength must be finite and nonnegative")
        valid, _, _, _, _ = self._validate(condition, geometry)
        x = geometry.coordinates
        if prior_logits.shape != condition.event_valid_mask.shape or prior_logits.dtype != x.dtype or prior_logits.device != x.device:
            raise ValueError("prior_logits must be [B,E] with coordinate dtype/device")
        observed = prior_logits[condition.event_valid_mask]
        if not bool((torch.isfinite(observed) | torch.isneginf(observed)).all()):
            raise ValueError("Valid prior logits must be finite or negative infinity")
        support = condition.event_valid_mask & ~torch.isneginf(prior_logits)
        clean_prior = torch.where(support, prior_logits, 0)
        prior_probability, prior_partition = _masked_distribution(clean_prior[..., None], support[..., None])
        log_prior = clean_prior - prior_partition
        valid = valid & support[..., None]
        if coupling_strength == 0:
            posterior = torch.where(valid, prior_probability.expand_as(valid), 0)
            return CoupledProposal(posterior, torch.zeros_like(x), valid, None)
        if torch.is_inference_mode_enabled():
            raise ValueError("couple requires autograd; use no_grad rather than inference_mode")
        with torch.enable_grad():
            coordinates = x if x.requires_grad else x.detach().requires_grad_(True)
            scored = self(condition, replace(geometry, coordinates=coordinates))
            logits = log_prior[..., None] + coupling_strength * scored.values
            posterior, partition = _masked_distribution(logits, valid)
            feedback = torch.autograd.grad(partition.sum(), coordinates, create_graph=create_graph,
                                           retain_graph=True)[0]
        if not create_graph:
            feedback = feedback.detach()
        return CoupledProposal(posterior, feedback, valid, scored)


def compatibility_correspondence_loss(scores: CompatibilityScores,
                                     evidence: CompatibilityEvidence) -> Tensor:
    """Weighted BCE only on evidenced reference correspondence, using the same C.

    This optional loss neither supplies labels nor declares a scientific training
    protocol. All-unobserved evidence produces a differentiable zero. Evidence
    asserted on an invalid candidate is an interface error rather than silently
    discarded supervision. Unknown targets and weights may be NaN.
    """
    values, targets, observed = scores.values, evidence.targets, evidence.observed_mask
    _boolean_mask(observed, tuple(values.shape), values.device, "observed_mask")
    if targets.shape != values.shape or targets.dtype != values.dtype or targets.device != values.device:
        raise ValueError("Evidence targets must match score shape/dtype/device")
    if bool((observed & ~scores.valid_mask).any()):
        raise ValueError("Observed evidence cannot refer to an invalid candidate")
    if not bool(((targets[observed] == 0) | (targets[observed] == 1)).all()):
        raise ValueError("Observed correspondence targets must be zero or one")
    weights = torch.ones_like(values) if evidence.weights is None else evidence.weights
    if weights.shape != values.shape or weights.dtype != values.dtype or weights.device != values.device:
        raise ValueError("Evidence weights must match score shape/dtype/device")
    if not bool((torch.isfinite(weights[observed]) & (weights[observed] > 0)).all()):
        raise ValueError("Observed evidence weights must be positive and finite")
    if not bool(observed.any()):
        return values.sum() * 0
    losses = F.binary_cross_entropy_with_logits(values[observed], targets[observed], reduction="none")
    return (losses * weights[observed]).sum() / weights[observed].sum()


def _software_example():
    """A deterministic CPU interface exercise with invented, nonchemical data."""
    torch.manual_seed(7)
    model = EventGeometryCompatibility(event_feature_dim=4).double()
    edits = torch.zeros(1, 2, 3, 3, dtype=torch.float64)
    edits[0, 0, 0, 1] = edits[0, 0, 1, 0] = -1
    edits[0, 1, 1, 2] = edits[0, 1, 2, 1] = 1
    condition = EventGeometryCondition(torch.randn(1, 2, 4, dtype=torch.float64), edits,
        torch.ones(1, 2, dtype=torch.bool), ConditionSource.INDEPENDENT_METADATA)
    geometry = CandidateGeometry(torch.randn(1, 2, 3, 3, dtype=torch.float64),
        torch.ones(1, 3, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool),
        CandidateGeometrySource.GENERATED_CANDIDATE)
    coupled = model.couple(condition, geometry, torch.zeros(1, 2, dtype=torch.float64), coupling_strength=.2)
    print({"scope": "software example only; untrained and nonchemical", "parameters": sum(p.numel() for p in model.parameters()),
           "posterior_shape": list(coupled.posterior.shape), "feedback_finite": bool(torch.isfinite(coupled.coordinate_feedback).all())})


if __name__ == "__main__":
    _software_example()
