"""Sampling controls."""

from .guidance import GuidanceSchedule, schedule_strength
try:
    from .integrator import control_euler_step, euler_step, serial_euler_step
except ModuleNotFoundError as exc:
    # The first #166 contract is calculator-free and should remain importable
    # in lightweight audit environments without the optional torch runtime.
    if exc.name != "torch":
        raise

    _TORCH_IMPORT_ERROR = exc

    def _missing_torch(*args, **kwargs):
        raise ModuleNotFoundError("torch is required for sampling integrators") from _TORCH_IMPORT_ERROR

    control_euler_step = _missing_torch
    euler_step = _missing_torch
    serial_euler_step = _missing_torch
from .parallel_guidance import (
    GuidanceCost,
    GuidanceLane,
    GuidanceLedger,
    GuidanceLedgerEntry,
    GuidanceOutput,
    GuidanceStatus,
    GenerationSnapshot,
    MergeConfig,
    MergeDecision,
    MergeResult,
    merge_guidance,
    project_masked_event_delta,
    reject_target_derived_fields,
)
from .parallel_replay import (
    DeterministicSyntheticProvider,
    OfflineGuidanceReplay,
    ReplayArtifact,
    ReplayBudget,
    ReplayMode,
    ReplayStep,
    REPLAY_MODES,
    write_replay_artifacts,
)

__all__ = [
    "GuidanceSchedule", "schedule_strength", "control_euler_step", "euler_step", "serial_euler_step",
    "GuidanceCost", "GuidanceLane", "GuidanceLedger", "GuidanceLedgerEntry", "GuidanceOutput",
    "GuidanceStatus", "GenerationSnapshot", "MergeConfig", "MergeDecision", "MergeResult",
    "merge_guidance", "project_masked_event_delta", "reject_target_derived_fields",
    "DeterministicSyntheticProvider", "OfflineGuidanceReplay", "ReplayArtifact", "ReplayBudget",
    "ReplayMode", "ReplayStep", "REPLAY_MODES", "write_replay_artifacts",
]
