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
from .cp2k import (
    BOHR_IN_ANGSTROM,
    CP2KAdapter,
    CP2KProtocol,
    CP2KRunnerResult,
    parse_cp2k_output,
    render_cp2k_input,
)
from .cp2k_runtime import (
    CP2KRuntimeIdentity,
    build_cp2k_protocol,
    inspect_cp2k_runtime,
    load_cp2k_protocol_document,
    physical_protocol_identity,
    physical_protocol_record,
    protocol_document_sha256,
)
from .xtb_oracle import GFN2OracleAdapter, XTBOracleAdapter
from .xtbloom import XTBloomAdapter

__all__ = [
    "CalculatorBackend",
    "CalculatorCapabilities",
    "CalculatorConvergenceError",
    "CalculatorError",
    "CalculatorProtocol",
    "CalculatorProtocolError",
    "CalculatorUnavailable",
    "CalculationResult",
    "MolecularSystem",
    "coerce_backend_output",
    "finite_difference_forces",
    "XTBloomAdapter",
    "XTBOracleAdapter",
    "GFN2OracleAdapter",
    "BOHR_IN_ANGSTROM",
    "CP2KAdapter",
    "CP2KProtocol",
    "CP2KRunnerResult",
    "parse_cp2k_output",
    "render_cp2k_input",
    "CP2KRuntimeIdentity",
    "build_cp2k_protocol",
    "inspect_cp2k_runtime",
    "load_cp2k_protocol_document",
    "physical_protocol_identity",
    "physical_protocol_record",
    "protocol_document_sha256",
]
