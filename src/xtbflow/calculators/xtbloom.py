"""Explicit xTBloom adapter boundary.

The public repository does not assume an xTBloom Python or CLI API.  Until a
qualified build is injected, this adapter reports the installation as unknown
or unavailable and refuses to invent energies or forces.
"""
from __future__ import annotations

from importlib.util import find_spec
from shutil import which
from typing import Any, Callable, Mapping

from .base import (
    CalculatorBackend,
    CalculatorCapabilities,
    CalculatorError,
    CalculatorProtocol,
    CalculatorProtocolError,
    CalculatorUnavailable,
    CalculationResult,
    MolecularSystem,
    coerce_backend_output,
)


class XTBloomAdapter(CalculatorBackend):
    """Use an explicitly injected, qualified xTBloom evaluator."""

    def __init__(
        self,
        protocol: CalculatorProtocol | None = None,
        evaluator: Callable[[MolecularSystem, str], CalculationResult | Mapping[str, Any]] | None = None,
        *,
        evaluate_fn: Callable[[MolecularSystem, str], CalculationResult | Mapping[str, Any]] | None = None,
        version: str | None = None,
        build_hash: str | None = None,
    ) -> None:
        if evaluator is not None and evaluate_fn is not None:
            raise ValueError("provide either evaluator or evaluate_fn, not both")
        self.protocol = protocol or CalculatorProtocol(
            protocol_id="xtbloom-gfn2-v0",
            calculator="xtbloom",
            method="GFN2-xTB",
        )
        if self.protocol.calculator != "xtbloom":
            raise CalculatorProtocolError("xTBloom adapter requires a protocol with calculator='xtbloom'")
        self._evaluator = evaluator or evaluate_fn
        self.version = version
        self.build_hash = build_hash

    @property
    def capabilities(self) -> CalculatorCapabilities:
        if self._evaluator is not None:
            return CalculatorCapabilities("xtbloom", "pass", "injected evaluator; external installation not asserted", self.version, self.build_hash, tuple(sorted({"energy", "forces", "energy_forces"})), (self.protocol.backend,))
        module = find_spec("xtbloom")
        executable = which("xtbloom")
        if module is None and executable is None:
            return CalculatorCapabilities("xtbloom", "unavailable", "no xtbloom module or executable was observed")
        return CalculatorCapabilities("xtbloom", "unknown", "xtbloom was observed but its API is not qualified", self.version, self.build_hash)

    def evaluate(self, system: MolecularSystem, *, operation: str = "energy_forces") -> CalculationResult:
        if operation not in {"energy", "forces", "energy_forces"}:
            raise CalculatorProtocolError(f"unsupported operation: {operation}")
        if self._evaluator is None:
            raise CalculatorUnavailable("xTBloom is not qualified or injected in this environment")
        try:
            output = self._evaluator(system, operation)
        except CalculatorError:
            raise
        except Exception as exc:
            raise RuntimeError(f"xTBloom evaluator failed before producing a contract result: {exc}") from exc
        return coerce_backend_output(output, system, self.protocol, operation=operation)
