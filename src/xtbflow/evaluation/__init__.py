"""Evaluation helpers for bounded candidate outputs.

The lightweight terminal contracts stay importable without scientific
dependencies. Reactant generation and grouped statistics are loaded lazily
when a caller requests them, which keeps audit and static-planning commands
usable on a machine without NumPy or PyTorch.
"""
from __future__ import annotations

from importlib import import_module

from .terminal import (
    GENERATION_FAILURE_STAGES,
    GENERATION_STATUSES,
    INITIAL_STATE_POLICIES,
    GenerationAttemptRecord,
    GenerationProtocol,
    TerminalCandidate,
    deduplicate_terminal_candidates,
)

_GENERATION_EXPORTS = {
    "GeneratedCandidate", "GenerationBatch", "RepeatedGenerationResult", "ReactantInput", "ReactantView",
    "ReferenceLabels", "candidate_fingerprint", "candidate_fingerprints",
    "centered_reactant_coordinates", "checkpoint_atom_count",
    "deduplicate_generated_candidates", "generated_candidate_key", "generate_candidate", "generate_candidates",
    "generate_repeated_candidates", "generate_repeated_candidates_with_accounting", "label_replacement_invariant", "load_generation_model",
    "load_reactant_inputs", "reactant_packed_state", "reactant_view_from_sample",
    "reference_labels_from_sample", "score_candidates",
}
_GROUPED_EXPORTS = {"paired_group_comparison", "summarize_by_group"}


def __getattr__(name: str):
    if name in _GENERATION_EXPORTS:
        module = import_module(".generation", __name__)
    elif name in _GROUPED_EXPORTS:
        module = import_module(".grouped", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value

__all__ = [
    "GeneratedCandidate",
    "GenerationBatch",
    "RepeatedGenerationResult",
    "ReactantInput",
    "ReactantView",
    "ReferenceLabels",
    "candidate_fingerprint",
    "candidate_fingerprints",
    "centered_reactant_coordinates",
    "checkpoint_atom_count",
    "deduplicate_generated_candidates",
    "generated_candidate_key",
    "generate_candidate",
    "generate_candidates",
    "generate_repeated_candidates",
    "generate_repeated_candidates_with_accounting",
    "label_replacement_invariant",
    "load_generation_model",
    "load_reactant_inputs",
    "paired_group_comparison",
    "reactant_packed_state",
    "reactant_view_from_sample",
    "reference_labels_from_sample",
    "score_candidates",
    "summarize_by_group",
    "GENERATION_FAILURE_STAGES",
    "GENERATION_STATUSES",
    "INITIAL_STATE_POLICIES",
    "GenerationAttemptRecord",
    "GenerationProtocol",
    "TerminalCandidate",
    "deduplicate_terminal_candidates",
]
