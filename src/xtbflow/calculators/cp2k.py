"""Versioned CP2K reference E/F protocol, renderer, parser, and adapter."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
        if self.path_status not in {"not_requested", "not_run", "not_validated", "validated"}:
            raise ValueError("unsupported path validation status")
        if not isinstance(self.parameters, Mapping):
            raise ValueError("parameters must be a mapping")

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


def render_cp2k_input(system: MolecularSystem, protocol: CP2KProtocol, *, project: str = "xtbflow") -> str:
    if system.charge != protocol.charge or system.multiplicity != protocol.multiplicity:
        raise CalculatorProtocolError("system charge/multiplicity does not match the frozen CP2K protocol")
    rows = ["&GLOBAL", f"  PROJECT {project}", "  RUN_TYPE ENERGY_FORCE", "&END GLOBAL", "&FORCE_EVAL", "  METHOD QS", "  &DFT", f"    BASIS_SET_FILE_NAME {protocol.basis_set}", f"    POTENTIAL_FILE_NAME {protocol.pseudopotential}", f"    CHARGE {protocol.charge}", f"    MULTIPLICITY {protocol.multiplicity}", f"    &SCF", f"      EPS_SCF {protocol.scf_epsilon:.12g}", f"      MAX_SCF {protocol.max_scf}", "    &END SCF", "    &XC", f"      &XC_FUNCTIONAL {protocol.functional}", "      &END XC_FUNCTIONAL", f"      &VDW_POTENTIAL", f"        POTENTIAL_TYPE {protocol.dispersion}", "      &END VDW_POTENTIAL", "    &END XC", "  &END DFT", "  &SUBSYS"]
    if protocol.boundary == "periodic":
        rows.extend(["    &CELL", "      ABC 20 20 20", "    &END CELL"])
    rows.append("    &COORD")
    for symbol, coordinate in zip(system.symbols, system.coordinates):
        rows.append("      " + symbol + " " + " ".join(f"{value:.16g}" for value in coordinate))
    rows.extend(["    &END COORD", "  &END SUBSYS", "&END FORCE_EVAL", ""])
    return "\n".join(rows)


_ENERGY = re.compile(r"ENERGY\|.*?energy.*?([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s*$", re.IGNORECASE)
_FORCE_HEADER = re.compile(r"ATOMIC FORCES.*?\[a\.u\.\]", re.IGNORECASE)
_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


def parse_cp2k_output(text: str, system: MolecularSystem, protocol: CP2KProtocol, *, input_file_hash: str) -> CalculationResult:
    if not isinstance(text, str) or not text.strip():
        raise CalculatorProtocolError("CP2K output is empty")
    energy_match = _ENERGY.search(text)
    not_converged = bool(re.search(r"SCF.*(?:NOT CONVERGED|FAILED)|SCF run NOT converged", text, re.IGNORECASE))
    if not_converged:
        return CalculationResult(protocol.calculator_protocol().calculator, protocol.protocol_id, system.input_hash, system.charge, system.multiplicity, "energy_forces", status="not_converged", converged=False, error_category="convergence", error_message="CP2K SCF did not converge", calculator_calls=1, calculator_build_hash=protocol.build_hash, input_file_hash=input_file_hash, path_status=protocol.path_status)
    if energy_match is None:
        raise CalculatorProtocolError("CP2K energy line was not found")
    energy = float(energy_match.group(1))
    lines = text.splitlines()
    force_start = next((index for index, line in enumerate(lines) if _FORCE_HEADER.search(line)), None)
    if force_start is None:
        raise CalculatorProtocolError("CP2K atomic-force section was not found")
    forces: list[tuple[float, float, float]] = []
    for line in lines[force_start + 1:]:
        numbers = re.findall(_FLOAT, line)
        if len(numbers) < 3:
            if forces:
                break
            continue
        # CP2K force rows carry atom/kind/element columns before the final xyz.
        try:
            candidate = tuple(float(value) for value in numbers[-3:])
        except ValueError:
            continue
        if all(math.isfinite(value) for value in candidate):
            forces.append(candidate)
        if len(forces) == len(system.symbols):
            break
    if len(forces) != len(system.symbols):
        raise CalculatorProtocolError(f"CP2K force section has {len(forces)} rows; expected {len(system.symbols)}")
    return CalculationResult(protocol.calculator_protocol().calculator, protocol.protocol_id, system.input_hash, system.charge, system.multiplicity, "energy_forces", energy=energy, forces=tuple(forces), calculator_calls=1, calculator_build_hash=protocol.build_hash, input_file_hash=input_file_hash, path_status=protocol.path_status)


class CP2KAdapter(CalculatorBackend):
    """Run CP2K only when an executable or explicitly injected runner exists."""

    def __init__(self, protocol: CP2KProtocol, *, executable: str | None = None, runner: Callable[[str, MolecularSystem], str] | None = None, timeout_seconds: float = 120.0):
        self.cp2k_protocol = protocol
        self.protocol = protocol.calculator_protocol()
        self.executable = executable or shutil.which("cp2k.psmp") or shutil.which("cp2k.popt") or shutil.which("cp2k")
        self.runner = runner
        if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
            raise ValueError("timeout_seconds must be finite and positive")
        self.timeout_seconds = float(timeout_seconds)

    @property
    def capabilities(self) -> CalculatorCapabilities:
        if self.runner is not None:
            return CalculatorCapabilities("cp2k", "pass", "injected runner; installation not asserted", self.cp2k_protocol.cp2k_version, self.cp2k_protocol.build_hash, ("energy_forces",), ("cpu",))
        if self.executable is None:
            return CalculatorCapabilities("cp2k", "unavailable", "no CP2K executable was observed")
        return CalculatorCapabilities("cp2k", "unknown", "CP2K executable observed but protocol qualification is pending", self.cp2k_protocol.cp2k_version, self.cp2k_protocol.build_hash, ("energy_forces",), ("cpu",))

    def evaluate(self, system: MolecularSystem, *, operation: str = "energy_forces") -> CalculationResult:
        if operation != "energy_forces":
            raise CalculatorProtocolError("CP2K v0 adapter supports energy_forces only")
        rendered = render_cp2k_input(system, self.cp2k_protocol)
        input_hash = hashlib.sha256(rendered.encode()).hexdigest()
        if self.runner is not None:
            try:
                text = self.runner(rendered, system)
            except Exception as exc:
                raise CalculatorError(f"CP2K runner failed: {exc}") from exc
            return parse_cp2k_output(text, system, self.cp2k_protocol, input_file_hash=input_hash)
        if self.executable is None:
            raise CalculatorUnavailable("CP2K is not installed or injected")
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
            if completed.returncode != 0 and not output_path.exists():
                raise CalculatorError(f"CP2K exited with code {completed.returncode}: {completed.stderr[-500:]}")
            return parse_cp2k_output(output_path.read_text(encoding="utf-8", errors="replace"), system, self.cp2k_protocol, input_file_hash=input_hash)
