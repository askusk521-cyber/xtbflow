"""Public-data contracts and leakage-safe splitting.

The tensor-backed task views are optional at import time.  Keeping them lazy
allows record admission, pairing, and cache tooling to run without importing
PyTorch, while preserving the same public ``xtbflow.data`` names when a model
consumer requests them.
"""
from __future__ import annotations

from importlib import import_module

from .public_sources import SourceAudit, hash_file, verify_local_asset
from .records import (
    CHNOS_ELEMENTS,
    PUBLIC_CANONICAL_UNITS,
    PublicRecord,
    RecordFilterResult,
    RecordPolicy,
    canonical_hash,
    filter_records,
    load_jsonl,
    write_jsonl,
)
from .splits import SplitError, admitted_records, assign_group_splits, audit_no_group_leakage
from .pairing import EFPair, PAIR_CANONICAL_UNITS, build_ef_pair, failure_pair, geometry_hash
from .cache import append_pair_cache, load_pair_cache
from .track_b import (
    TRACK_B_SCHEMA,
    TrackBFilterResult,
    TrackBLeakageError,
    TrackBRecord,
    audit_track_b_leakage,
    filter_track_b_records,
    load_track_b_jsonl,
    write_track_b_jsonl,
)
from .independent_seeds import (
    INDEPENDENT_SEED_SCHEMA,
    SEED_SPLIT_ROLES,
    IndependentReactantSeed,
    SeedLeakageError,
    audit_seed_leakage,
    load_seed_jsonl,
    write_seed_jsonl,
)
from .attempts import (
    EVIDENCE_STATES,
    SEARCH_ATTEMPT_SCHEMA,
    SearchAttempt,
    append_attempt_jsonl,
    load_attempt_jsonl,
)
from .origin_readiness import (
    ORIGIN_CANDIDATE_SCHEMA,
    ORIGIN_SEED_READINESS_SCHEMA,
    OriginReadinessError,
    audit_origin_candidate_registry,
)

_TASK_VIEW_EXPORTS = (
    "EndpointGeometryInput",
    "GeometryFlowResult",
    "GeometryTarget",
    "GeometryTaskMode",
    "ReactantEventGeometryInput",
)
_PAIRED_EXPORTS = (
    "PairedBatch",
    "PairedPilotLoader",
    "load_paired_pilot",
    "input_view_fingerprint",
    "reactant_input_fingerprint",
)


def __getattr__(name: str):
    if name in _TASK_VIEW_EXPORTS:
        module = import_module(".task_views", __name__)
        for export in _TASK_VIEW_EXPORTS:
            globals()[export] = getattr(module, export)
        return globals()[name]
    if name in _PAIRED_EXPORTS:
        module = import_module(".paired_loader", __name__)
        for export in _PAIRED_EXPORTS:
            globals()[export] = getattr(module, export)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SourceAudit", "hash_file", "verify_local_asset", "PublicRecord", "canonical_hash",
    "PUBLIC_CANONICAL_UNITS", "CHNOS_ELEMENTS", "RecordPolicy", "RecordFilterResult",
    "filter_records", "load_jsonl", "write_jsonl", "SplitError", "admitted_records",
    "assign_group_splits", "audit_no_group_leakage",
    "EFPair", "PAIR_CANONICAL_UNITS", "build_ef_pair", "failure_pair", "geometry_hash", "append_pair_cache", "load_pair_cache",
    "TRACK_B_SCHEMA", "TrackBRecord", "TrackBFilterResult", "TrackBLeakageError",
    "filter_track_b_records", "audit_track_b_leakage", "load_track_b_jsonl", "write_track_b_jsonl",
    *_PAIRED_EXPORTS,
    "INDEPENDENT_SEED_SCHEMA", "SEED_SPLIT_ROLES", "IndependentReactantSeed",
    "SeedLeakageError", "audit_seed_leakage", "load_seed_jsonl", "write_seed_jsonl",
    "SEARCH_ATTEMPT_SCHEMA", "EVIDENCE_STATES", "SearchAttempt", "append_attempt_jsonl", "load_attempt_jsonl",
    "ORIGIN_CANDIDATE_SCHEMA", "ORIGIN_SEED_READINESS_SCHEMA",
    "OriginReadinessError", "audit_origin_candidate_registry",
    *_TASK_VIEW_EXPORTS,
]
