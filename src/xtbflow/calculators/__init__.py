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
from .xtb_oracle import GFN2OracleAdapter, XTBOracleAdapter
from .xtbloom import XTBloomAdapter

__all__ = [
    "CalculatorBackend", "CalculatorCapabilities", "CalculatorConvergenceError", "CalculatorError",
    "CalculatorProtocol", "CalculatorProtocolError", "CalculatorUnavailable", "CalculationResult",
    "MolecularSystem", "coerce_backend_output", "finite_difference_forces", "XTBloomAdapter",
    "XTBOracleAdapter", "GFN2OracleAdapter",
]
