"""Evaluation helpers for bounded candidate outputs."""

from .generation import (
    GeneratedCandidate,
    GenerationBatch,
    ReactantInput,
    ReactantView,
    ReferenceLabels,
    candidate_fingerprint,
    candidate_fingerprints,
    centered_reactant_coordinates,
    checkpoint_atom_count,
    generate_candidate,
    generate_candidates,
    label_replacement_invariant,
    load_generation_model,
    load_reactant_inputs,
    reactant_view_from_sample,
    reactant_packed_state,
    reference_labels_from_sample,
    score_candidates,
)
from .grouped import paired_group_comparison, summarize_by_group
from .terminal import TerminalCandidate, deduplicate_terminal_candidates

__all__ = [
    "GeneratedCandidate",
    "GenerationBatch",
    "ReactantInput",
    "ReactantView",
    "ReferenceLabels",
    "TerminalCandidate",
    "candidate_fingerprint",
    "candidate_fingerprints",
    "checkpoint_atom_count",
    "deduplicate_terminal_candidates",
    "generate_candidate",
    "generate_candidates",
    "label_replacement_invariant",
    "load_generation_model",
    "load_reactant_inputs",
    "paired_group_comparison",
    "reactant_view_from_sample",
    "reference_labels_from_sample",
    "score_candidates",
    "summarize_by_group",
]
