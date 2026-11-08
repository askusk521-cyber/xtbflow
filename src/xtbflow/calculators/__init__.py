"""Fail-closed energy/force calculator contracts and adapters."""

from .base import (
    CalculatorBackend,
    CalculatorCapabilities,
    CalculatorConvergenceError,
    CalculatorError,
    CalculatorProtocol,
    CalculatorProtocolError,
    CalculatorUnavailable,
    CalculationResult,
    MolecularSystem,
    coerce_backend_output,
    finite_difference_forces,
)
# Optional native adapters keep the lightweight mock/ledger path importable
# when a clean checkout has not installed NumPy or a native calculator yet.
try:
    from .xtb_oracle import GFN2OracleAdapter, XTBOracleAdapter
except ImportError:  # pragma: no cover - exercised in dependency-free smoke environments
    GFN2OracleAdapter = None  # type: ignore[assignment]
    XTBOracleAdapter = None  # type: ignore[assignment]
try:
    from .xtbloom import XTBloomAdapter
except ImportError:  # pragma: no cover - optional adapter dependency
    XTBloomAdapter = None  # type: ignore[assignment]
from .cp2k import BOHR_IN_ANGSTROM, CP2KAdapter, CP2KProtocol, CP2KRunnerResult, parse_cp2k_output, render_cp2k_input
try:
    from .generativeqc import GenerativeQCAdapter
except ImportError:  # pragma: no cover - optional adapter dependency
    GenerativeQCAdapter = None  # type: ignore[assignment]
from .mock import MockCalculator

__all__ = [
    "CalculatorBackend", "CalculatorCapabilities", "CalculatorConvergenceError", "CalculatorError",
    "CalculatorProtocol", "CalculatorProtocolError", "CalculatorUnavailable", "CalculationResult",
    "MolecularSystem", "coerce_backend_output", "finite_difference_forces", "XTBloomAdapter",
    "XTBOracleAdapter", "GFN2OracleAdapter", "BOHR_IN_ANGSTROM", "CP2KAdapter", "CP2KProtocol", "CP2KRunnerResult", "parse_cp2k_output", "render_cp2k_input", "GenerativeQCAdapter", "MockCalculator",
]
