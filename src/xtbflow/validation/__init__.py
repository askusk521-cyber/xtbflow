"""Fail-closed transition-state and endpoint connectivity validation."""

from .connectivity import ConnectivityEvidence, observed_event
from .modes import CANONICAL_MODE_UNIT, MODE_EVIDENCE_CONTRACT, ModeEvidence, normalize_mode_unit, validate_mode
from .ts_search import TSValidationConfig, TSValidationRecord, candidate_content_hash, run_resumable_validation, validate_ts_evidence, validation_cache_key

__all__ = ["ConnectivityEvidence", "observed_event", "CANONICAL_MODE_UNIT", "MODE_EVIDENCE_CONTRACT", "ModeEvidence", "normalize_mode_unit", "validate_mode", "TSValidationConfig", "TSValidationRecord", "candidate_content_hash", "validation_cache_key", "run_resumable_validation", "validate_ts_evidence"]
