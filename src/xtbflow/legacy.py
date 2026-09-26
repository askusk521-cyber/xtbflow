"""Stable adapters for the immutable reusable ``mechai`` snapshot.

The research package owns the ``xtbflow`` namespace.  The imported ``mechai``
modules remain an explicit, content-addressed compatibility surface so future
algorithms can evolve without silently rewriting the inherited contracts.
"""
from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys
from types import ModuleType

# Source checkouts do not install the vendor tree on sys.path.  Wheels map the
# same packages directly through pyproject.toml, so this branch is a no-op
# after installation.  Keeping the path adjustment here makes source tests and
# editable installs use precisely the same adapter path as a built wheel.
_VENDOR_ROOT = Path(__file__).resolve().parents[2] / "vendor" / "mechai_reusable"
if _VENDOR_ROOT.is_dir() and str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from mechai.data.events import (  # noqa: E402,F401
    BondEdit,
    ElectronEdit,
    EventOrigin,
    EventProposal,
    LewisState,
    ProtonTransfer,
)
from mechai.data.proposals import (  # noqa: E402,F401
    ProposalSet,
    RankedProposal,
    enumerate_relays,
)
from mechai.data.sample_manifest import (  # noqa: E402,F401
    ManifestValidationError,
    validate_manifest,
)
from mechai.evaluation.event_geometry import (  # noqa: E402,F401
    ControlMode,
    ControlWiring,
    InferenceCostBudget,
    InferenceCostLedger,
    decode_event_geometry,
)
from mechai.models.conditioning import (  # noqa: E402,F401
    ConditionSource,
    EnvironmentCondition,
    EquivariantEnvironmentResidual,
    UniTSConditionedDynamics,
)
from mechai.models.event_geometry import (  # noqa: E402,F401
    CandidateGeometry,
    CandidateGeometrySource,
    CompatibilityEvidence,
    CompatibilityScores,
    CoupledProposal,
    EventGeometryCompatibility,
    EventGeometryCondition,
    compatibility_correspondence_loss,
)


def module(name: str) -> ModuleType:
    """Return a legacy module through an explicit adapter boundary.

    Callers should use this helper when they need a module-level function that
    is not re-exported above.  It keeps the dependency visible and avoids
    copying or mutating the vendor implementation.
    """

    if not name.startswith("mechai."):
        raise ValueError("Only the mechai compatibility namespace is supported")
    return import_module(name)


def source_root() -> Path:
    """Return the installed source directory containing the mechai snapshot."""

    return Path(module("mechai").__file__).resolve().parent


__all__ = [
    "BondEdit",
    "ElectronEdit",
    "EventOrigin",
    "EventProposal",
    "LewisState",
    "ProtonTransfer",
    "ProposalSet",
    "RankedProposal",
    "enumerate_relays",
    "ManifestValidationError",
    "validate_manifest",
    "ControlMode",
    "ControlWiring",
    "InferenceCostBudget",
    "InferenceCostLedger",
    "decode_event_geometry",
    "ConditionSource",
    "EnvironmentCondition",
    "EquivariantEnvironmentResidual",
    "UniTSConditionedDynamics",
    "CandidateGeometry",
    "CandidateGeometrySource",
    "CompatibilityEvidence",
    "CompatibilityScores",
    "CoupledProposal",
    "EventGeometryCompatibility",
    "EventGeometryCondition",
    "compatibility_correspondence_loss",
    "module",
    "source_root",
]
