"""Independent original-xTB/tblite GFN2 comparison adapter boundary."""
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


class XTBOracleAdapter(CalculatorBackend):
    """Use a separately injected original xTB or tblite evaluator."""

    def __init__(
        self,
        protocol: CalculatorProtocol | None = None,
        evaluator: Callable[[MolecularSystem, str], CalculationResult | Mapping[str, Any]] | None = None,
        *,
        evaluate_fn: Callable[[MolecularSystem, str], CalculationResult | Mapping[str, Any]] | None = None,
        implementation: str = "xtb",
        version: str | None = None,
        build_hash: str | None = None,
    ) -> None:
        if implementation not in {"xtb", "tblite"}:
            raise ValueError("implementation must be 'xtb' or 'tblite'")
        if evaluator is not None and evaluate_fn is not None:
            raise ValueError("provide either evaluator or evaluate_fn, not both")
        self.implementation = implementation
        self.protocol = protocol or CalculatorProtocol(
            protocol_id=f"{implementation}-gfn2-oracle-v0",
            calculator="xtb_oracle",
            method="GFN2-xTB",
        )
        if self.protocol.calculator != "xtb_oracle":
            raise CalculatorProtocolError("xTB oracle adapter requires a protocol with calculator='xtb_oracle'")
        self._evaluator = evaluator or evaluate_fn
        self.version = version
        self.build_hash = build_hash

    @property
    def capabilities(self) -> CalculatorCapabilities:
        if self._evaluator is not None:
            return CalculatorCapabilities("xtb_oracle", "pass", "injected evaluator; external installation not asserted", self.version, self.build_hash, tuple(sorted({"energy", "forces", "energy_forces"})), (self.protocol.backend,))
        module = find_spec(self.implementation)
        executable = which(self.implementation)
        if module is None and executable is None:
            return CalculatorCapabilities("xtb_oracle", "unavailable", f"no {self.implementation} module or executable was observed")
        return CalculatorCapabilities("xtb_oracle", "unknown", f"{self.implementation} was observed but its API is not qualified", self.version, self.build_hash)

    def evaluate(self, system: MolecularSystem, *, operation: str = "energy_forces") -> CalculationResult:
        if operation not in {"energy", "forces", "energy_forces"}:
            raise CalculatorProtocolError(f"unsupported operation: {operation}")
        if self._evaluator is None:
            raise CalculatorUnavailable(f"independent {self.implementation} backend is not qualified or injected")
        try:
            output = self._evaluator(system, operation)
        except CalculatorError:
            raise
        except Exception as exc:
            raise RuntimeError(f"independent evaluator failed before producing a contract result: {exc}") from exc
        return coerce_backend_output(output, system, self.protocol, operation=operation)


GFN2OracleAdapter = XTBOracleAdapter
