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
from .cp2k import BOHR_IN_ANGSTROM, CP2KAdapter, CP2KProtocol, CP2KRunnerResult, parse_cp2k_output, render_cp2k_input

__all__ = [
    "CalculatorBackend", "CalculatorCapabilities", "CalculatorConvergenceError", "CalculatorError",
    "CalculatorProtocol", "CalculatorProtocolError", "CalculatorUnavailable", "CalculationResult",
    "MolecularSystem", "coerce_backend_output", "finite_difference_forces", "XTBloomAdapter",
    "XTBOracleAdapter", "GFN2OracleAdapter", "BOHR_IN_ANGSTROM", "CP2KAdapter", "CP2KProtocol", "CP2KRunnerResult", "parse_cp2k_output", "render_cp2k_input",
]
