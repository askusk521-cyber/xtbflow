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

_TASK_VIEW_EXPORTS = (
    "EndpointGeometryInput",
    "GeometryFlowResult",
    "GeometryTarget",
    "GeometryTaskMode",
    "ReactantEventGeometryInput",
)


def __getattr__(name: str):
    if name in _TASK_VIEW_EXPORTS:
        module = import_module(".task_views", __name__)
        for export in _TASK_VIEW_EXPORTS:
            globals()[export] = getattr(module, export)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SourceAudit", "hash_file", "verify_local_asset", "PublicRecord", "canonical_hash",
    "PUBLIC_CANONICAL_UNITS", "CHNOS_ELEMENTS", "RecordPolicy", "RecordFilterResult",
    "filter_records", "load_jsonl", "write_jsonl", "SplitError", "admitted_records",
    "assign_group_splits", "audit_no_group_leakage",
    "EFPair", "PAIR_CANONICAL_UNITS", "build_ef_pair", "failure_pair", "geometry_hash", "append_pair_cache", "load_pair_cache",
    *_TASK_VIEW_EXPORTS,
]
