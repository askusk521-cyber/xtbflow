"""Versioned CP2K reference E/F protocol, renderer, parser, and adapter."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
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
)
from xtbflow.runtime.ledger import BudgetTokenRequired, CalculatorCallToken


# CP2K's ``[a.u.]`` force table is Hartree/Bohr.  The public calculator
# contract is Hartree/Angstrom, so conversion belongs at this parser boundary.
BOHR_IN_ANGSTROM = 0.529177210903
FORCE_CONVERSION_VERSION = "bohr_to_angstrom_codata2018"
_SUPPORTED_PARAMETER_KEYS = frozenset({
    "basis_set_file", "pseudopotential_file", "basis_by_element",
    "pseudopotential_by_element", "cell_angstrom", "dispersion_parameter_file",
    "reference_functional", "threads",
})


@dataclass(frozen=True)
class CP2KProtocol:
    """All physical settings needed to reproduce one CP2K E/F protocol."""

    protocol_id: str
    functional: str
    basis_set: str
    pseudopotential: str
    dispersion: str
    cutoff_ry: float
    relative_cutoff_ry: float
    scf_epsilon: float
    max_scf: int
    charge: int
    multiplicity: int
    build_hash: str
    cp2k_version: str
    boundary: str = "isolated"
    solvent_model: str = "none"
    path_status: str = "not_requested"
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("protocol_id", "functional", "basis_set", "pseudopotential", "dispersion", "build_hash", "cp2k_version", "boundary", "solvent_model"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("cutoff_ry", "relative_cutoff_ry", "scf_epsilon"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.max_scf) is not int or self.max_scf < 1:
            raise ValueError("max_scf must be positive")
        if type(self.charge) is not int or type(self.multiplicity) is not int or self.multiplicity < 1:
            raise ValueError("charge and multiplicity must be explicit")
        if self.boundary not in {"isolated", "periodic"}:
            raise ValueError("boundary must be isolated or periodic")
        if self.solvent_model != "none":
            raise ValueError("only solvent_model='none' is implemented; unsupported solvent settings fail closed")
        if self.path_status not in {"not_requested", "not_run", "not_validated", "validated"}:
            raise ValueError("unsupported path validation status")
        if not isinstance(self.parameters, Mapping):
            raise ValueError("parameters must be a mapping")
        unsupported = sorted(set(self.parameters) - _SUPPORTED_PARAMETER_KEYS)
        if unsupported:
            raise ValueError("unsupported CP2K protocol parameters: " + ", ".join(unsupported))
        threads = self.parameters.get("threads")
        if threads is not None and (type(threads) is not int or threads < 1):
            raise ValueError("parameters.threads must be a positive integer")
        for name in ("basis_set_file", "pseudopotential_file"):
            value = self.parameters.get(name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty string when provided")
        for name in ("basis_by_element", "pseudopotential_by_element"):
            value = self.parameters.get(name, {})
            if not isinstance(value, Mapping) or any(not isinstance(key, str) or not isinstance(item, str) or not item.strip() for key, item in value.items()):
                raise ValueError(f"{name} must map element symbols to nonempty labels")
        for name in ("dispersion_parameter_file", "reference_functional"):
            value = self.parameters.get(name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty string when provided")
        cell = self.parameters.get("cell_angstrom")
        if self.boundary == "periodic" and cell is None:
            raise ValueError("periodic CP2K protocols require explicit parameters.cell_angstrom")
        if cell is not None and (not isinstance(cell, (tuple, list)) or len(cell) != 3 or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0 for value in cell)):
            raise ValueError("parameters.cell_angstrom must contain three positive finite lengths")
        if self.dispersion not in {"none", "DFTD3(BJ)"}:
            raise ValueError("unsupported CP2K dispersion setting")
        if self.dispersion == "DFTD3(BJ)" and not self.parameters.get("dispersion_parameter_file"):
            raise ValueError("DFTD3(BJ) requires parameters.dispersion_parameter_file")

    @property
    def identity(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def calculator_protocol(self) -> CalculatorProtocol:
        return CalculatorProtocol(
            protocol_id=self.protocol_id,
            calculator="cp2k",
            method=self.functional,
            parameters=asdict(self),
        )

    def for_state(self, charge: int, multiplicity: int, *, protocol_id: str | None = None) -> "CP2KProtocol":
        if type(charge) is not int:
            raise ValueError("charge must be an explicit integer")
        if type(multiplicity) is not int or multiplicity < 1:
            raise ValueError("multiplicity must be an explicit positive integer")
        return replace(self, protocol_id=protocol_id or f"{self.protocol_id}:q{charge}:m{multiplicity}", charge=charge, multiplicity=multiplicity)


def render_cp2k_input(system: MolecularSystem, protocol: CP2KProtocol, *, project: str = "xtbflow") -> str:
    if system.charge != protocol.charge or system.multiplicity != protocol.multiplicity:
        raise CalculatorProtocolError("system charge/multiplicity does not match the frozen CP2K protocol")
    rows = ["&GLOBAL", f"  PROJECT {project}", "  RUN_TYPE ENERGY_FORCE", "&END GLOBAL", "&FORCE_EVAL", "  METHOD QS", "  &PRINT", "    &FORCES", "    &END FORCES", "  &END PRINT", "  &DFT"]
    # Labels and library filenames are distinct CP2K concepts.  Do not place
    # a basis label in BASIS_SET_FILE_NAME merely because the old adapter had
    # only one string field for both values.
    if protocol.parameters.get("basis_set_file"):
        rows.append(f"    BASIS_SET_FILE_NAME {protocol.parameters['basis_set_file']}")
    if protocol.parameters.get("pseudopotential_file"):
        rows.append(f"    POTENTIAL_FILE_NAME {protocol.parameters['pseudopotential_file']}")
    rows.extend([f"    CHARGE {protocol.charge}", f"    MULTIPLICITY {protocol.multiplicity}"])
    if protocol.multiplicity > 1:
        rows.append("    LSD")
    rows.extend(["    &MGRID", f"      CUTOFF {protocol.cutoff_ry:.12g}", f"      REL_CUTOFF {protocol.relative_cutoff_ry:.12g}", "    &END MGRID", "    &SCF", f"      EPS_SCF {protocol.scf_epsilon:.12g}", f"      MAX_SCF {protocol.max_scf}", "    &END SCF", "    &XC", f"      &XC_FUNCTIONAL {protocol.functional}", "      &END XC_FUNCTIONAL"])
    if protocol.dispersion == "DFTD3(BJ)":
        rows.extend(["      &VDW_POTENTIAL", "        POTENTIAL_TYPE PAIR_POTENTIAL", "        &PAIR_POTENTIAL", "          TYPE DFTD3(BJ)", f"          PARAMETER_FILE_NAME {protocol.parameters['dispersion_parameter_file']}", f"          REFERENCE_FUNCTIONAL {protocol.parameters.get('reference_functional', protocol.functional)}", "        &END PAIR_POTENTIAL", "      &END VDW_POTENTIAL"])
    rows.append("    &END XC")
    if protocol.boundary == "isolated":
        rows.extend(["    &POISSON", "      PERIODIC NONE", "      PSOLVER MT", "    &END POISSON"])
    rows.extend(["  &END DFT", "  &SUBSYS", "    &CELL"])
    cell = protocol.parameters.get("cell_angstrom", (20.0, 20.0, 20.0))
    rows.extend(["      ABC " + " ".join(f"{float(value):.12g}" for value in cell), "      PERIODIC " + ("XYZ" if protocol.boundary == "periodic" else "NONE"), "    &END CELL", "    &COORD"])
    for symbol, coordinate in zip(system.symbols, system.coordinates):
        rows.append("      " + symbol + " " + " ".join(f"{value:.16g}" for value in coordinate))
    rows.append("    &END COORD")
    basis_by_element = protocol.parameters.get("basis_by_element", {})
    potential_by_element = protocol.parameters.get("pseudopotential_by_element", {})
    for symbol in dict.fromkeys(system.symbols):
        rows.extend([f"    &KIND {symbol}", f"      BASIS_SET {basis_by_element.get(symbol, protocol.basis_set)}", f"      POTENTIAL {potential_by_element.get(symbol, protocol.pseudopotential)}", "    &END KIND"])
    rows.extend(["  &END SUBSYS", "&END FORCE_EVAL", ""])
    return "\n".join(rows)


_ENERGY = re.compile(r"ENERGY\|.*?energy.*?([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s*$", re.IGNORECASE | re.MULTILINE)
_FORCE_HEADER = re.compile(r"ATOMIC FORCES.*?\[a\.u\.\]", re.IGNORECASE)
_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
_FORCE_ROW = re.compile(r"^\s*(\d+)\s+\d+\s+([A-Za-z][A-Za-z0-9]*)\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s*$")


def parse_cp2k_output(text: str, system: MolecularSystem, protocol: CP2KProtocol, *, input_file_hash: str) -> CalculationResult:
    if not isinstance(text, str) or not text.strip():
        raise CalculatorProtocolError("CP2K output is empty")
    not_converged = bool(re.search(r"SCF.*(?:NOT CONVERGED|FAILED)|SCF run NOT converged", text, re.IGNORECASE))
    if not_converged:
        return CalculationResult(protocol.calculator_protocol().calculator, protocol.protocol_id, system.input_hash, system.charge, system.multiplicity, "energy_forces", status="not_converged", converged=False, error_category="convergence", error_message="CP2K SCF did not converge", calculator_calls=1, calculator_build_hash=protocol.build_hash, input_file_hash=input_file_hash, path_status=protocol.path_status)
    if "PROGRAM ENDED AT" not in text:
        raise CalculatorProtocolError("CP2K completion marker was not found")
    energy_matches = list(_ENERGY.finditer(text))
    if len(energy_matches) != 1:
        raise CalculatorProtocolError("CP2K energy line was not found")
    energy = float(energy_matches[0].group(1))
    lines = text.splitlines()
    force_headers = [index for index, line in enumerate(lines) if _FORCE_HEADER.search(line)]
    if len(force_headers) != 1:
        raise CalculatorProtocolError("CP2K atomic-force section was not found")
    force_start = force_headers[0]
    forces: list[tuple[float, float, float]] = []
    raw_forces: list[tuple[float, float, float]] = []
    for line in lines[force_start + 1:]:
        if "SUM OF ATOMIC FORCES" in line.upper():
            break
        match = _FORCE_ROW.match(line)
        if match is None:
            continue
        if len(raw_forces) >= len(system.symbols):
            raise CalculatorProtocolError("CP2K force section contains more rows than the requested atom inventory")
        index = int(match.group(1))
        symbol = match.group(2)
        if index != len(raw_forces) + 1 or symbol != system.symbols[len(raw_forces)]:
            raise CalculatorProtocolError("CP2K force rows do not match the requested atom order")
        raw = tuple(float(match.group(axis)) for axis in (3, 4, 5))
        if not all(math.isfinite(value) for value in raw):
            raise CalculatorProtocolError("CP2K force row contains a non-finite value")
        raw_forces.append(raw)
        forces.append(tuple(value / BOHR_IN_ANGSTROM for value in raw))
    if len(forces) != len(system.symbols):
        raise CalculatorProtocolError(f"CP2K force section has {len(forces)} rows; expected {len(system.symbols)}")
    return CalculationResult(protocol.calculator_protocol().calculator, protocol.protocol_id, system.input_hash, system.charge, system.multiplicity, "energy_forces", energy=energy, forces=tuple(forces), calculator_calls=1, calculator_build_hash=protocol.build_hash, input_file_hash=input_file_hash, path_status=protocol.path_status, metadata={"raw_force_unit": "hartree/bohr", "normalized_force_unit": "hartree/angstrom", "force_conversion_version": FORCE_CONVERSION_VERSION, "raw_forces": raw_forces})


class CP2KAdapter(CalculatorBackend):
    """Run CP2K only when an executable or explicitly injected runner exists."""

    def __init__(self, protocol: CP2KProtocol, *, executable: str | None = None, runner: Callable[[str, MolecularSystem], str] | None = None, timeout_seconds: float = 120.0, require_budget_token: bool = False):
        self.cp2k_protocol = protocol
        self.protocol = protocol.calculator_protocol()
        self.executable = executable or shutil.which("cp2k.psmp") or shutil.which("cp2k.popt") or shutil.which("cp2k")
        self.runner = runner
        if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
            raise ValueError("timeout_seconds must be finite and positive")
        self.timeout_seconds = float(timeout_seconds)
        self.require_budget_token = bool(require_budget_token)

    @property
    def capabilities(self) -> CalculatorCapabilities:
        if self.runner is not None:
            return CalculatorCapabilities("cp2k", "unknown", "injected runner; physical installation and protocol qualification are not asserted", self.cp2k_protocol.cp2k_version, self.cp2k_protocol.build_hash, ("energy_forces",), ("cpu",))
        if self.executable is None:
            return CalculatorCapabilities("cp2k", "unavailable", "no CP2K executable was observed")
        return CalculatorCapabilities("cp2k", "unknown", "CP2K executable observed but protocol qualification is pending", self.cp2k_protocol.cp2k_version, self.cp2k_protocol.build_hash, ("energy_forces",), ("cpu",))

    def evaluate(
        self,
        system: MolecularSystem,
        *,
        operation: str = "energy_forces",
        budget_token: CalculatorCallToken | None = None,
    ) -> CalculationResult:
        if operation != "energy_forces":
            raise CalculatorProtocolError("CP2K v0 adapter supports energy_forces only")
        rendered = render_cp2k_input(system, self.cp2k_protocol)
        input_hash = hashlib.sha256(rendered.encode()).hexdigest()
        if self.runner is not None:
            if self.require_budget_token and budget_token is None:
                raise BudgetTokenRequired("CP2K production calls require a calculator budget token")
            if budget_token is not None:
                budget_token.consume()
            try:
                text = self.runner(rendered, system)
            except Exception as exc:
                raise CalculatorError(f"CP2K runner failed: {exc}") from exc
            return parse_cp2k_output(text, system, self.cp2k_protocol, input_file_hash=input_hash)
        if self.executable is None:
            raise CalculatorUnavailable("CP2K is not installed or injected")
        if self.require_budget_token and budget_token is None:
            raise BudgetTokenRequired("CP2K production calls require a calculator budget token")
        if budget_token is not None:
            budget_token.consume()
        with tempfile.TemporaryDirectory(prefix="xtbflow-cp2k-") as directory:
            root = Path(directory)
            input_path = root / "input.inp"
            output_path = root / "output.out"
            input_path.write_text(rendered, encoding="utf-8")
            try:
                completed = subprocess.run([self.executable, "-i", str(input_path), "-o", str(output_path)], check=False, capture_output=True, text=True, timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                raise CalculatorError("CP2K execution timed out") from exc
            except OSError as exc:
                raise CalculatorUnavailable(f"CP2K executable could not be started: {exc}") from exc
            if completed.returncode != 0:
                raise CalculatorError(f"CP2K exited with code {completed.returncode}: {completed.stderr[-500:]}")
            if not output_path.exists():
                raise CalculatorError("CP2K completed without producing an output file")
            return parse_cp2k_output(output_path.read_text(encoding="utf-8", errors="replace"), system, self.cp2k_protocol, input_file_hash=input_hash)
