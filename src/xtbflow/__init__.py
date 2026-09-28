"""Public package entry point for the xtbflow research adapters.

The inherited ``mechai`` compatibility surface is optional for calculator,
ledger, data, and validation consumers.  Load it lazily so those standard
contracts remain usable in lightweight CPU environments that do not install
PyTorch or the reusable snapshot.
"""
from __future__ import annotations

from importlib import import_module

__version__ = "0.1.0"

_LEGACY_EXPORTS = (
    "BondEdit", "ElectronEdit", "EventOrigin", "EventProposal", "LewisState",
    "ProtonTransfer", "ProposalSet", "RankedProposal", "enumerate_relays",
    "ManifestValidationError", "validate_manifest", "ControlMode", "ControlWiring",
    "InferenceCostBudget", "InferenceCostLedger", "decode_event_geometry",
    "ConditionSource", "EnvironmentCondition", "EquivariantEnvironmentResidual",
    "UniTSConditionedDynamics", "CandidateGeometry", "CandidateGeometrySource",
    "CompatibilityEvidence", "CompatibilityScores", "CoupledProposal",
    "EventGeometryCompatibility", "EventGeometryCondition",
    "compatibility_correspondence_loss", "module", "source_root",
)


def _load_legacy():
    """Load and cache the optional reusable compatibility namespace."""

    module = import_module(".legacy", __name__)
    globals()["legacy"] = module
    for name in module.__all__:
        globals()[name] = getattr(module, name)
    return module


def __getattr__(name: str):
    if name == "legacy" or name in _LEGACY_EXPORTS:
        module = _load_legacy()
        return module if name == "legacy" else getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["__version__", "legacy", *_LEGACY_EXPORTS]
