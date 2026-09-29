"""Fail-closed transition-state and endpoint connectivity validation."""

from .connectivity import ConnectivityEvidence, infer_binary_connectivity, observed_event
from .evidence import (
    index_artifacts,
    sanitize_cp2k_calibration_report,
    sanitize_cp2k_convergence_report,
    sanitize_error_message,
    sanitize_public_value,
    sha256_file,
)
from .modes import ModeEvidence, validate_mode
from .reference_bridge import bridge_case_record, force_delta_metrics, summarize_bridge
from .ts_search import (
    TSValidationConfig,
    TSValidationRecord,
    candidate_content_hash,
    run_resumable_validation,
    validate_ts_evidence,
    validation_cache_key,
)

__all__ = [
    "ConnectivityEvidence",
    "infer_binary_connectivity",
    "index_artifacts",
    "sanitize_cp2k_calibration_report",
    "sanitize_cp2k_convergence_report",
    "sanitize_error_message",
    "sanitize_public_value",
    "sha256_file",
    "observed_event",
    "ModeEvidence",
    "validate_mode",
    "TSValidationConfig",
    "TSValidationRecord",
    "candidate_content_hash",
    "validation_cache_key",
    "run_resumable_validation",
    "validate_ts_evidence",
    "bridge_case_record",
    "force_delta_metrics",
    "summarize_bridge",
]
