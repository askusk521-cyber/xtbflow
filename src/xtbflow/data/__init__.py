"""Public-data contracts and leakage-safe splitting."""

from .public_sources import SourceAudit, hash_file, verify_local_asset
from .records import PublicRecord, canonical_hash, load_jsonl, write_jsonl
from .splits import SplitError, admitted_records, assign_group_splits, audit_no_group_leakage
from .pairing import EFPair, build_ef_pair, failure_pair, geometry_hash
from .cache import append_pair_cache, load_pair_cache

__all__ = [
    "SourceAudit", "hash_file", "verify_local_asset", "PublicRecord", "canonical_hash",
    "load_jsonl", "write_jsonl", "SplitError", "admitted_records",
    "assign_group_splits", "audit_no_group_leakage",
    "EFPair", "build_ef_pair", "failure_pair", "geometry_hash", "append_pair_cache", "load_pair_cache",
    "EndpointGeometryInput", "GeometryFlowResult", "GeometryTarget", "GeometryTaskMode", "ReactantEventGeometryInput",
]

from .task_views import EndpointGeometryInput, GeometryFlowResult, GeometryTarget, GeometryTaskMode, ReactantEventGeometryInput
