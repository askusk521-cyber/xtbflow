"""Label-free terminal decoding of a bounded event–geometry score grid.

The internal E×G grid is not a prediction set: each supplied geometry can
contribute at most one event–geometry pair to the final K-slot output. Scores
are model compatibility values, not physical probabilities or energies.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from enum import Enum
from typing import Literal, Mapping

import torch
from torch import Tensor

from mechai.models.event_geometry import CompatibilityScores


class ControlMode(str, Enum):
    """Inference wiring arms registered by the event--geometry protocol.

    These flags describe which *calls* an arm is allowed to make.  They do not
    select a model checkpoint, event prior, sampler, or chemical interpretation.
    In particular, ``shared_bidirectional`` uses the same scalar compatibility
    function for both the event posterior and the marginal coordinate feedback;
    the latter must be the derivative of the same log-sum-exp objective and must
    already include ``lambda`` exactly once in ``EventGeometryCompatibility``.
    """

    STRONG_RULES = "strong_rules"
    ONE_WAY_FIXED_EVENT = "one_way_fixed_event"
    FINAL_RERANKING_ONLY = "final_reranking_only"
    EVENT_UPDATE_ONLY = "event_update_only"
    BOTH_DIRECTIONS_OFF = "both_directions_off"
    SHARED_BIDIRECTIONAL = "shared_bidirectional"


@dataclass(frozen=True)
class ControlWiring:
    """Explicit feature switches for one registered inference control.

    ``ControlWiring`` is deliberately separate from a sampler implementation:
    an experiment runner can validate the intended switches before dispatching
    work, and then validate the recorded call counts afterwards.  The switches
    are an accounting contract, not evidence that a control is chemically fair.
    """

    mode: ControlMode

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ControlMode):
            raise ValueError("mode must be a registered ControlMode")

    @property
    def compatibility_event_updates(self) -> bool:
        return self.mode in (ControlMode.EVENT_UPDATE_ONLY, ControlMode.SHARED_BIDIRECTIONAL)

    @property
    def compatibility_gradient_feedback(self) -> bool:
        return self.mode in (ControlMode.ONE_WAY_FIXED_EVENT, ControlMode.SHARED_BIDIRECTIONAL)

    @property
    def final_compatibility_ranking(self) -> bool:
        return self.mode in (
            ControlMode.ONE_WAY_FIXED_EVENT,
            ControlMode.FINAL_RERANKING_ONLY,
            ControlMode.EVENT_UPDATE_ONLY,
            ControlMode.SHARED_BIDIRECTIONAL,
        )

    def validate_ledger(self, ledger: "InferenceCostLedger") -> None:
        """Reject call traces that perform a disabled intervention.

        Forward compatibility calls are not rejected here because the same
        score may be evaluated for an enabled gradient or final ranking.  The
        intervention-specific counters are what distinguish causal wiring from
        harmless diagnostics.
        """
        if not isinstance(ledger, InferenceCostLedger):
            raise TypeError("ledger must be an InferenceCostLedger")
        if not self.compatibility_event_updates and ledger.event_update_calls:
            raise ValueError(f"{self.mode.value} disabled compatibility event updates")
        if not self.compatibility_gradient_feedback and ledger.gradient_feedback_calls:
            raise ValueError(f"{self.mode.value} disabled compatibility gradient feedback")
        if not self.final_compatibility_ranking and ledger.final_ranking_calls:
            raise ValueError(f"{self.mode.value} disabled final compatibility ranking")


@dataclass(frozen=True)
class InferenceCostLedger:
    """Immutable per-input accounting for a candidate-generation arm.

    Counts use event--particle pairs rather than batched tensor calls so a
    larger batch cannot hide extra work.  ``event_proposals_attempted`` keeps
    pre-cap and unsupported proposals visible; ``event_slots_supplied`` is the
    actual E axis passed to the compatibility model.  Every trajectory attempt,
    including failures and retries, is retained.  This is a software ledger;
    it does not estimate FLOPs, wall-clock equivalence, or scientific value.

    The ledger is intended to be recorded once per input/control arm and then
    validated against ``InferenceCostBudget``.  Use :meth:`add` for phase-local
    updates and :meth:`merge` only after each input has passed its own budget
    check.  ``wall_time_seconds`` is optional elapsed time on the declared
    hardware and is never used to infer chemical efficiency.
    """

    event_slots_supplied: int = 0
    event_proposals_attempted: int = 0
    event_proposals_truncated: int = 0
    geometry_trajectory_attempts: int = 0
    geometry_trajectories_completed: int = 0
    geometry_trajectories_failed: int = 0
    geometry_retries: int = 0
    backbone_evaluations: int = 0
    compatibility_forward_calls: int = 0
    compatibility_forward_pairs: int = 0
    compatibility_gradient_calls: int = 0
    compatibility_gradient_pairs: int = 0
    conditioning_calls: int = 0
    screening_candidates: int = 0
    ranking_pairs: int = 0
    final_pairs_emitted: int = 0
    event_update_calls: int = 0
    gradient_feedback_calls: int = 0
    final_ranking_calls: int = 0
    training_updates: int = 0
    training_examples: int = 0
    trainable_parameters: int = 0
    wall_time_seconds: float = 0.0

    def __post_init__(self) -> None:
        integer_fields = (
            "event_slots_supplied", "event_proposals_attempted", "event_proposals_truncated",
            "geometry_trajectory_attempts", "geometry_trajectories_completed",
            "geometry_trajectories_failed", "geometry_retries", "backbone_evaluations",
            "compatibility_forward_calls", "compatibility_forward_pairs",
            "compatibility_gradient_calls", "compatibility_gradient_pairs",
            "conditioning_calls", "screening_candidates", "ranking_pairs",
            "final_pairs_emitted", "event_update_calls", "gradient_feedback_calls",
            "final_ranking_calls", "training_updates", "training_examples",
            "trainable_parameters",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if isinstance(self.wall_time_seconds, bool) or not isinstance(self.wall_time_seconds, (int, float)):
            raise ValueError("wall_time_seconds must be a nonnegative finite number")
        if not math.isfinite(float(self.wall_time_seconds)) or self.wall_time_seconds < 0:
            raise ValueError("wall_time_seconds must be a nonnegative finite number")
        if self.event_proposals_truncated > self.event_proposals_attempted:
            raise ValueError("truncated proposals cannot exceed attempted proposals")
        if self.event_slots_supplied > self.event_proposals_attempted:
            raise ValueError("supplied event slots cannot exceed attempted proposals")
        if self.geometry_trajectories_completed + self.geometry_trajectories_failed > self.geometry_trajectory_attempts:
            raise ValueError("completed and failed trajectories exceed attempts")
        if self.geometry_retries > self.geometry_trajectory_attempts:
            raise ValueError("retries cannot exceed trajectory attempts")
        if self.final_pairs_emitted > self.geometry_trajectories_completed:
            raise ValueError("emitted pairs cannot exceed completed trajectories")

    def add(self, **increments: int | float) -> "InferenceCostLedger":
        """Return a new ledger after adding nonnegative phase-local counters."""
        known = set(asdict(self))
        unknown = set(increments) - known
        if unknown:
            raise ValueError(f"unknown ledger fields: {sorted(unknown)}")
        for name, value in increments.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{name} increment must be nonnegative")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} increment must be finite")
            if name != "wall_time_seconds" and type(value) is not int:
                raise ValueError(f"{name} increment must be an integer")
        values = asdict(self)
        for name, value in increments.items():
            values[name] += value
        return replace(self, **values)

    def merge(self, *others: "InferenceCostLedger") -> "InferenceCostLedger":
        """Sum independent ledgers after their per-input checks have passed."""
        values = asdict(self)
        for other in others:
            if not isinstance(other, InferenceCostLedger):
                raise TypeError("all merged values must be InferenceCostLedger")
            for name, value in asdict(other).items():
                values[name] += value
        return replace(self, **values)


@dataclass(frozen=True)
class InferenceCostBudget:
    """Candidate and compatibility limits from the draft protocol.

    Limits are per input/control arm.  The pre-cap proposal count is optionally
    bounded separately from the supplied event axis so that an experiment can
    report a complete enumeration without silently treating discarded proposals
    as free.  ``total_compute_cap_seconds`` is a declared wall-time cap only;
    it should be frozen by an independent backend pilot before real comparison.
    """

    max_events_per_input: int = 32
    max_geometry_trajectory_attempts: int = 8
    max_final_pairs: int = 8
    max_compatibility_forward_pairs: int = 768
    max_compatibility_gradient_pairs: int = 512
    max_event_proposals_attempted: int | None = None
    total_compute_cap_seconds: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_events_per_input", "max_geometry_trajectory_attempts", "max_final_pairs",
            "max_compatibility_forward_pairs", "max_compatibility_gradient_pairs",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_event_proposals_attempted is not None and (
            type(self.max_event_proposals_attempted) is not int or self.max_event_proposals_attempted < 1
        ):
            raise ValueError("max_event_proposals_attempted must be None or a positive integer")
        if self.total_compute_cap_seconds is not None and (
            isinstance(self.total_compute_cap_seconds, bool)
            or not isinstance(self.total_compute_cap_seconds, (int, float))
            or not math.isfinite(float(self.total_compute_cap_seconds))
            or self.total_compute_cap_seconds <= 0
        ):
            raise ValueError("total_compute_cap_seconds must be None or a positive finite number")

    def validate(self, ledger: InferenceCostLedger) -> None:
        """Raise when a recorded arm exceeds a declared candidate/cost limit."""
        if not isinstance(ledger, InferenceCostLedger):
            raise TypeError("ledger must be an InferenceCostLedger")
        limits = (
            ("event_slots_supplied", self.max_events_per_input),
            ("geometry_trajectory_attempts", self.max_geometry_trajectory_attempts),
            ("final_pairs_emitted", self.max_final_pairs),
            ("compatibility_forward_pairs", self.max_compatibility_forward_pairs),
            ("compatibility_gradient_pairs", self.max_compatibility_gradient_pairs),
        )
        for field, limit in limits:
            value = getattr(ledger, field)
            if value > limit:
                raise ValueError(f"{field}={value} exceeds budget {limit}")
        if self.max_event_proposals_attempted is not None and (
            ledger.event_proposals_attempted > self.max_event_proposals_attempted
        ):
            raise ValueError(
                f"event_proposals_attempted={ledger.event_proposals_attempted} exceeds budget "
                f"{self.max_event_proposals_attempted}"
            )
        if self.total_compute_cap_seconds is not None and ledger.wall_time_seconds > self.total_compute_cap_seconds:
            raise ValueError(
                f"wall_time_seconds={ledger.wall_time_seconds} exceeds budget {self.total_compute_cap_seconds}"
            )


def validate_shared_control_budget(
    ledgers: Mapping[ControlMode, InferenceCostLedger],
    budget: InferenceCostBudget,
) -> None:
    """Validate common candidate/backend accounting across control arms.

    Every arm must use the same supplied event pool, pre-cap proposal count,
    trajectory-attempt budget, and backbone evaluation count.  Intervention
    counters and final emitted pairs may differ by design.  This helper checks
    accounting only; equal counts do not prove equal wall time or scientific
    fairness, which still require the protocol's hardware and sampler records.
    """
    if not isinstance(ledgers, Mapping) or not ledgers:
        raise ValueError("at least one control ledger is required")
    if not isinstance(budget, InferenceCostBudget):
        raise TypeError("budget must be an InferenceCostBudget")
    common_fields = (
        "event_slots_supplied", "event_proposals_attempted",
        "geometry_trajectory_attempts", "backbone_evaluations",
    )
    expected = None
    for mode, ledger in ledgers.items():
        wiring = ControlWiring(mode)
        budget.validate(ledger)
        wiring.validate_ledger(ledger)
        observed = tuple(getattr(ledger, field) for field in common_fields)
        if expected is None:
            expected = observed
        elif observed != expected:
            raise ValueError(
                f"control arms do not share candidate/backend counts: {common_fields}"
            )


@dataclass(frozen=True)
class DecodedEventGeometryPair:
    """One emitted pair; IDs are zero-based slots in the supplied E/G axes.

    ``terminal_score`` is normalized log prior + strength × compatibility.
    It is not normalized over events separately for each geometry. The caller
    must retain the external IDs and provenance associated with these slots.
    """

    event_id: int
    geometry_id: int
    terminal_score: float


@dataclass(frozen=True)
class DecodedEventGeometry:
    """Detached terminal output and unfilled-budget accounting for one input.

    Supplied counts include invalid/padded slots. ``eligible_geometry_count``
    counts geometries with at least one valid, prior-supported event before
    top-K truncation. ``unused_slots`` is requested_k minus the emitted count;
    it is retained even when fewer geometries were supplied. ``status`` is
    ``complete`` or ``partial`` for nonempty output, ``empty_no_valid_pairs``
    when no input pair is valid, and ``empty_no_prior_support`` when valid
    pairs exist but all corresponding event priors are zero (-inf logits).

    The oracle flag is propagated even at zero coupling and on empty output.
    None of these fields establishes chemical success or provenance correctness.
    """

    pairs: tuple[DecodedEventGeometryPair, ...]
    requested_k: int
    supplied_event_count: int
    supplied_geometry_count: int
    eligible_geometry_count: int
    unused_slots: int
    status: Literal["complete", "partial", "empty_no_valid_pairs", "empty_no_prior_support"]
    is_oracle: bool


def decode_event_geometry(
    scores: CompatibilityScores,
    prior_logits: Tensor,
    *,
    coupling_strength: float = 0.0,
    k: int = 8,
    max_geometry_slots: int = 8,
    max_events: int = 32,
) -> tuple[DecodedEventGeometry, ...]:
    """Select at most K explicit pairs per batch item without reference labels.

    The prior [B,E] is normalized once over events having a valid pair and a
    finite prior logit. Valid events accept finite logits or -inf (zero
    support); entirely invalid events may have NaN padding. For each geometry,
    retain the supported event maximizing ``log_prior + coupling_strength*C``.
    Then rank these pairs by that same score. Equal scores use event index,
    then geometry index, so each geometry appears at most once. An
    event-normalized posterior per geometry must not replace the terminal
    score: its geometry-dependent denominator would change the ranking.

    Scores/masks are [B,E,G], and prior logits share score dtype and device.
    Masks are boolean. For positive strength, compatibility must be finite
    only at valid, prior-supported pairs. At strength zero compatibility is
    bypassed, including NaN values, rather than evaluating 0*NaN. Outputs are
    immutable Python records; tensors are detached and evaluated on CPU in
    float64, and no gradient graph, sampling, retry, or coordinate edit occurs.
    Nonfinite normalized priors or combined scores fail explicitly.

    B must be nonempty; E or G may be zero to record missing supplied slots.
    Supplied E/G dimensions are checked against the limits before masking,
    including invalid padding. The defaults enforce the current 32-event,
    8-geometry terminal contract; changing those limits needs its own external
    protocol. K is positive and at most max_geometry_slots. Missing candidates
    leave unused output slots; they never cause additional generation.

    This helper validates only supplied slots and final outputs, not sampling
    failures, retries, or the full compute ledger. Caller-side accounting must
    retain all attempted proposals/trajectories, including those not supplied
    here. The decoder never interprets an absent pair as a chemical negative.
    """
    for name, value in (("k", k), ("max_geometry_slots", max_geometry_slots),
                        ("max_events", max_events)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if k > max_geometry_slots:
        raise ValueError("k cannot exceed max_geometry_slots")
    if (isinstance(coupling_strength, bool)
            or not isinstance(coupling_strength, (int, float))
            or not math.isfinite(coupling_strength) or coupling_strength < 0):
        raise ValueError("coupling_strength must be finite and nonnegative")
    if not isinstance(scores, CompatibilityScores):
        raise TypeError("scores must be CompatibilityScores")
    values = scores.values
    if not isinstance(values, Tensor) or values.ndim != 3 or not values.is_floating_point():
        raise ValueError("scores.values must be floating [B,E,G]")
    batch_size, event_count, geometry_count = values.shape
    if batch_size < 1:
        raise ValueError("scores.values must have a nonempty batch")
    # Check supplied dimensions, not valid counts: masking extra trajectories
    # cannot turn a candidate-budget overrun into a compliant terminal output.
    if event_count > max_events:
        raise ValueError("Supplied event slots exceed max_events, including invalid slots")
    if geometry_count > max_geometry_slots:
        raise ValueError("Supplied geometry slots exceed max_geometry_slots, including invalid slots")
    valid = scores.valid_mask
    if (not isinstance(valid, Tensor) or valid.shape != values.shape
            or valid.dtype != torch.bool or valid.device != values.device):
        raise ValueError("scores.valid_mask must be boolean [B,E,G] on the score device")
    if (not isinstance(prior_logits, Tensor) or prior_logits.shape != (batch_size, event_count)
            or prior_logits.dtype != values.dtype or prior_logits.device != values.device):
        raise ValueError("prior_logits must be [B,E] with score dtype/device")
    if type(scores.is_oracle) is not bool:
        raise ValueError("scores.is_oracle must be an explicit boolean")

    cpu_valid = valid.detach().to(device="cpu")
    cpu_prior = prior_logits.detach().to(device="cpu", dtype=torch.float64)
    # Avoid even reading compatibility values when the intervention is off.
    cpu_scores = (values.detach().to(device="cpu", dtype=torch.float64)
                  if coupling_strength > 0 else None)
    outputs = []
    for batch_index in range(batch_size):
        valid_here = cpu_valid[batch_index]
        prior_here = cpu_prior[batch_index]
        event_valid = valid_here.any(dim=1)
        observed_priors = prior_here[event_valid]
        if not bool((torch.isfinite(observed_priors) | torch.isneginf(observed_priors)).all()):
            raise ValueError("Valid prior logits must be finite or negative infinity")
        supported = event_valid & torch.isfinite(prior_here)
        supported_ids = torch.nonzero(supported, as_tuple=False).flatten().tolist()
        log_prior = {}
        if supported_ids:
            # Center before computing the log partition. Directly subtracting
            # logsumexp([1e308, 1e308]) would lose the log(2) normalization.
            offset = max(float(prior_here[event]) for event in supported_ids)
            shifted = {event: float(prior_here[event]) - offset for event in supported_ids}
            partition = math.log(math.fsum(math.exp(value) for value in shifted.values()))
            log_prior = {event: value - partition for event, value in shifted.items()}
            if not all(math.isfinite(value) for value in log_prior.values()):
                raise ValueError("Normalized prior logits must be finite on supported events")

        candidates = []
        for geometry in range(geometry_count):
            best = None
            # Ascending event IDs implement the declared per-geometry tie rule;
            # equality does not replace the smaller-ID current best.
            for event in supported_ids:
                if not bool(valid_here[event, geometry]):
                    continue
                terminal_score = log_prior[event]
                if cpu_scores is not None:
                    compatibility = float(cpu_scores[batch_index, event, geometry])
                    if not math.isfinite(compatibility):
                        raise ValueError("Valid prior-supported compatibility scores must be finite")
                    terminal_score += coupling_strength * compatibility
                if not math.isfinite(terminal_score):
                    raise ValueError("Combined terminal scores must be finite")
                if best is None or terminal_score > best.terminal_score:
                    best = DecodedEventGeometryPair(event, geometry, terminal_score)
            if best is not None:
                candidates.append(best)

        candidates.sort(key=lambda pair: (-pair.terminal_score, pair.event_id, pair.geometry_id))
        selected = tuple(candidates[:k])
        if selected:
            status = "complete" if len(selected) == k else "partial"
        else:
            status = "empty_no_valid_pairs" if not bool(valid_here.any()) else "empty_no_prior_support"
        outputs.append(DecodedEventGeometry(
            pairs=selected,
            requested_k=k,
            supplied_event_count=event_count,
            supplied_geometry_count=geometry_count,
            eligible_geometry_count=len(candidates),
            unused_slots=k - len(selected),
            status=status,
            is_oracle=scores.is_oracle,
        ))
    return tuple(outputs)
