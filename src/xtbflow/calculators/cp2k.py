"""Versioned CP2K reference E/F protocol, renderer, parser, and adapter."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from contextlib import contextmanager
import hashlib
import json
import math
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Iterator, Mapping

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
_SUPPORTED_PARAMETER_KEYS = frozenset(
    {
        "basis_set_file",
        "pseudopotential_file",
        "basis_by_element",
        "pseudopotential_by_element",
        "cell_angstrom",
    }
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
        if self.solvent_model != "none":
            raise ValueError("only solvent_model='none' is implemented; unsupported solvent settings fail closed")
        if self.path_status not in {"not_requested", "not_run", "not_validated", "validated"}:
            raise ValueError("unsupported path validation status")
        if not isinstance(self.parameters, Mapping):
            raise ValueError("parameters must be a mapping")
        if any(not isinstance(key, str) or not key.strip() for key in self.parameters):
            raise ValueError("CP2K protocol parameter names must be nonempty strings")
        unsupported = sorted(set(self.parameters) - _SUPPORTED_PARAMETER_KEYS)
        if unsupported:
            raise ValueError(
                "unsupported CP2K protocol parameters would not affect rendered input: "
                + ", ".join(str(item) for item in unsupported)
            )
        for name in ("basis_set_file", "pseudopotential_file"):
            value = self.parameters.get(name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty string when provided")
        for name in ("basis_by_element", "pseudopotential_by_element"):
            value = self.parameters.get(name, {})
            if not isinstance(value, Mapping) or any(not isinstance(key, str) or not isinstance(item, str) or not item.strip() for key, item in value.items()):
                raise ValueError(f"{name} must map element symbols to nonempty labels")
        cell = self.parameters.get("cell_angstrom")
        if self.boundary == "periodic" and cell is None:
            raise ValueError("periodic CP2K protocols require explicit parameters.cell_angstrom")
        if cell is not None and (not isinstance(cell, (tuple, list)) or len(cell) != 3 or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0 for value in cell)):
            raise ValueError("parameters.cell_angstrom must contain three positive finite lengths")
        if self.dispersion not in {"none", "DFTD3(BJ)"}:
            raise ValueError("unsupported CP2K dispersion setting")

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
    rows = ["&GLOBAL", f"  PROJECT {project}", "  RUN_TYPE ENERGY_FORCE", "&END GLOBAL", "&FORCE_EVAL", "  METHOD QS", "  &DFT"]
    # Labels and library filenames are distinct CP2K concepts.  Do not place
    # a basis label in BASIS_SET_FILE_NAME merely because the old adapter had
    # only one string field for both values.
    if protocol.parameters.get("basis_set_file"):
        rows.append(f"    BASIS_SET_FILE_NAME {protocol.parameters['basis_set_file']}")
    if protocol.parameters.get("pseudopotential_file"):
        rows.append(f"    POTENTIAL_FILE_NAME {protocol.parameters['pseudopotential_file']}")
    rows.extend([f"    CHARGE {protocol.charge}", f"    MULTIPLICITY {protocol.multiplicity}", "    &MGRID", f"      CUTOFF {protocol.cutoff_ry:.12g}", f"      REL_CUTOFF {protocol.relative_cutoff_ry:.12g}", "    &END MGRID", "    &SCF", f"      EPS_SCF {protocol.scf_epsilon:.12g}", f"      MAX_SCF {protocol.max_scf}", "    &END SCF", "    &XC", f"      &XC_FUNCTIONAL {protocol.functional}", "      &END XC_FUNCTIONAL"])
    if protocol.dispersion == "DFTD3(BJ)":
        rows.extend(["      &VDW_POTENTIAL", "        POTENTIAL_TYPE PAIR_POTENTIAL", "        &PAIR_POTENTIAL", "          TYPE DFTD3(BJ)", "        &END PAIR_POTENTIAL", "      &END VDW_POTENTIAL"])
    rows.append("    &END XC")
    if protocol.boundary == "isolated":
        rows.extend(["    &POISSON", "      PERIODIC NONE", "    &END POISSON"])
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
_SCF_CONVERGED = re.compile(r"(?:SCF[^\n]*?(?:converged|convergence achieved)|(?:converged|convergence achieved)[^\n]*?SCF)", re.IGNORECASE)
_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
_FORCE_ROW = re.compile(r"^\s*(\d+)\s+\d+\s+([A-Za-z][A-Za-z0-9]*)\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s*$")


def parse_cp2k_output(
    text: str,
    system: MolecularSystem,
    protocol: CP2KProtocol,
    *,
    input_file_hash: str,
    metadata_extra: Mapping[str, Any] | None = None,
) -> CalculationResult:
    if not isinstance(text, str) or not text.strip():
        raise CalculatorProtocolError("CP2K output is empty")
    not_converged = bool(re.search(r"SCF.*(?:NOT CONVERGED|FAILED)|SCF run NOT converged", text, re.IGNORECASE))
    if not_converged:
        metadata = dict(metadata_extra or {})
        metadata.update({"scf_convergence_evidence": "negative_failure_marker"})
        return CalculationResult(protocol.calculator_protocol().calculator, protocol.protocol_id, system.input_hash, system.charge, system.multiplicity, "energy_forces", status="not_converged", converged=False, error_category="convergence", error_message="CP2K SCF did not converge", calculator_calls=1, calculator_build_hash=protocol.build_hash, input_file_hash=input_file_hash, path_status=protocol.path_status, metadata=metadata)
    if "PROGRAM ENDED AT" not in text:
        raise CalculatorProtocolError("CP2K completion marker was not found")
    if not _SCF_CONVERGED.search(text):
        raise CalculatorProtocolError("CP2K positive SCF convergence marker was not found")
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
    metadata = dict(metadata_extra or {})
    metadata.update({"raw_force_unit": "hartree/bohr", "normalized_force_unit": "hartree/angstrom", "force_conversion_version": FORCE_CONVERSION_VERSION, "raw_forces": raw_forces, "scf_convergence_evidence": "positive_marker"})
    return CalculationResult(protocol.calculator_protocol().calculator, protocol.protocol_id, system.input_hash, system.charge, system.multiplicity, "energy_forces", energy=energy, forces=tuple(forces), calculator_calls=1, calculator_build_hash=protocol.build_hash, input_file_hash=input_file_hash, path_status=protocol.path_status, metadata=metadata)


@dataclass(frozen=True)
class CP2KRunnerResult:
    """Explicit result returned by an injected runner.

    A plain output string remains accepted for small compatibility fixtures.  A
    structured result is required when a runner needs to expose return codes,
    stdout, stderr, and the completed CP2K output separately.
    """

    output: str
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.output, str):
            raise ValueError("CP2K runner output must be text")
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise ValueError("CP2K runner stdout/stderr must be text")
        if type(self.returncode) is not int:
            raise ValueError("CP2K runner returncode must be an integer")


def _coerce_runner_result(value: str | CP2KRunnerResult | Mapping[str, Any]) -> CP2KRunnerResult:
    if isinstance(value, CP2KRunnerResult):
        return value
    if isinstance(value, str):
        return CP2KRunnerResult(output=value)
    if isinstance(value, Mapping):
        try:
            return CP2KRunnerResult(
                output=value["output"],
                stdout=value.get("stdout", ""),
                stderr=value.get("stderr", ""),
                returncode=value.get("returncode", 0),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CalculatorProtocolError(f"invalid structured CP2K runner result: {exc}") from exc
    raise CalculatorProtocolError("CP2K runner must return text or a structured result")


class CP2KAdapter(CalculatorBackend):
    """Run CP2K only when an executable or explicitly injected runner exists."""

    def __init__(
        self,
        protocol: CP2KProtocol,
        *,
        executable: str | None = None,
        runner: Callable[[str, MolecularSystem], str | CP2KRunnerResult | Mapping[str, Any]] | None = None,
        timeout_seconds: float = 120.0,
        require_budget_token: bool = False,
        artifact_dir: str | Path | None = None,
        require_artifacts: bool = False,
    ):
        self.cp2k_protocol = protocol
        self.protocol = protocol.calculator_protocol()
        observed_executable = executable or shutil.which("cp2k.psmp") or shutil.which("cp2k.popt") or shutil.which("cp2k")
        if observed_executable is not None and Path(observed_executable).expanduser().is_file():
            observed_executable = str(Path(observed_executable).expanduser().resolve())
        self.executable = observed_executable
        self.runner = runner
        if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
            raise ValueError("timeout_seconds must be finite and positive")
        self.timeout_seconds = float(timeout_seconds)
        self.require_budget_token = bool(require_budget_token)
        self.artifact_dir = Path(artifact_dir).expanduser() if artifact_dir is not None else None
        self.require_artifacts = bool(require_artifacts)
        if self.require_artifacts and self.artifact_dir is None:
            raise ValueError("require_artifacts requires artifact_dir")
        if self.artifact_dir is not None:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            self.artifact_dir = self.artifact_dir.resolve()

    @property
    def capabilities(self) -> CalculatorCapabilities:
        if self.runner is not None:
            return CalculatorCapabilities("cp2k", "unknown", "injected runner; physical installation and protocol qualification are not asserted", self.cp2k_protocol.cp2k_version, self.cp2k_protocol.build_hash, ("energy", "forces", "energy_forces"), ("cpu",), "callable")
        if not self._executable_observed():
            return CalculatorCapabilities("cp2k", "unavailable", "no CP2K executable was observed", qualification="unavailable")
        return CalculatorCapabilities("cp2k", "unknown", "CP2K executable observed but protocol qualification is pending", self.cp2k_protocol.cp2k_version, self.cp2k_protocol.build_hash, ("energy", "forces", "energy_forces"), ("cpu",), "installed")

    def _executable_observed(self) -> bool:
        if self.executable is None:
            return False
        return Path(self.executable).is_file() or shutil.which(self.executable) is not None

    def evaluate(
        self,
        system: MolecularSystem,
        *,
        operation: str = "energy_forces",
        budget_token: CalculatorCallToken | None = None,
    ) -> CalculationResult:
        if operation not in {"energy", "forces", "energy_forces"}:
            raise CalculatorProtocolError(f"CP2K v0 adapter does not support operation: {operation}")
        rendered = render_cp2k_input(system, self.cp2k_protocol)
        input_hash = hashlib.sha256(rendered.encode()).hexdigest()
        if self.runner is not None:
            if self.require_budget_token and budget_token is None:
                raise BudgetTokenRequired("CP2K production calls require a calculator budget token")
            if budget_token is not None:
                budget_token.consume()
            return self._evaluate_runner(rendered, system, input_hash, operation)
        if self.executable is None:
            raise CalculatorUnavailable("CP2K is not installed or injected")
        if self.require_budget_token and budget_token is None:
            raise BudgetTokenRequired("CP2K production calls require a calculator budget token")
        if budget_token is not None:
            budget_token.consume()
        with self._artifact_workspace() as root:
            input_path = root / "input.inp"
            output_path = root / "output.out"
            input_path.write_text(rendered, encoding="utf-8")
            try:
                completed = subprocess.run(
                    [self.executable, "-i", input_path.name, "-o", output_path.name],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    cwd=root,
                )
            except subprocess.TimeoutExpired as exc:
                self._write_subprocess_stream(root / "stdout.txt", exc.stdout)
                self._write_subprocess_stream(root / "stderr.txt", exc.stderr)
                raise CalculatorError("CP2K execution timed out") from exc
            except OSError as exc:
                raise CalculatorUnavailable(f"CP2K executable could not be started: {exc}") from exc
            (root / "stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
            (root / "stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
            if completed.returncode != 0:
                raise CalculatorError(f"CP2K exited with code {completed.returncode}: {completed.stderr[-500:]}")
            if not output_path.exists():
                raise CalculatorError("CP2K completed without producing an output file")
            metadata = self._artifact_metadata(root, persisted=self.artifact_dir is not None)
            parsed = parse_cp2k_output(output_path.read_text(encoding="utf-8", errors="replace"), system, self.cp2k_protocol, input_file_hash=input_hash, metadata_extra=metadata)
            return self._select_operation(parsed, operation)

    @staticmethod
    def _write_subprocess_stream(path: Path, value: str | bytes | None) -> None:
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        path.write_text(value or "", encoding="utf-8")

    def _artifact_metadata(self, root: Path, *, persisted: bool) -> dict[str, Any]:
        return {
            "artifact_persistence": "persistent" if persisted else "temporary",
            "artifact_directory": str(root) if persisted else None,
            "artifact_files": sorted(path.name for path in root.iterdir() if path.is_file()),
        }

    @contextmanager
    def _artifact_workspace(self) -> Iterator[Path]:
        if self.artifact_dir is not None:
            yield Path(tempfile.mkdtemp(prefix="xtbflow-cp2k-", dir=self.artifact_dir)).resolve()
            return
        with tempfile.TemporaryDirectory(prefix="xtbflow-cp2k-") as directory:
            yield Path(directory)

    @staticmethod
    def _select_operation(result: CalculationResult, operation: str) -> CalculationResult:
        if operation == "energy":
            return replace(result, operation="energy", forces=None)
        if operation == "forces":
            return replace(result, operation="forces", energy=None)
        return result

    def _evaluate_runner(self, rendered: str, system: MolecularSystem, input_hash: str, operation: str) -> CalculationResult:
        with self._artifact_workspace() as root:
            (root / "input.inp").write_text(rendered, encoding="utf-8")
            try:
                result = _coerce_runner_result(self.runner(rendered, system))
            except Exception as exc:
                (root / "stdout.txt").write_text("", encoding="utf-8")
                (root / "stderr.txt").write_text(str(exc), encoding="utf-8")
                raise CalculatorError(f"CP2K runner failed: {exc}") from exc
            (root / "stdout.txt").write_text(result.stdout, encoding="utf-8")
            (root / "stderr.txt").write_text(result.stderr, encoding="utf-8")
            (root / "output.out").write_text(result.output, encoding="utf-8")
            if result.returncode != 0:
                raise CalculatorError(f"CP2K exited with code {result.returncode}: {result.stderr[-500:]}")
            metadata = self._artifact_metadata(root, persisted=self.artifact_dir is not None)
            parsed = parse_cp2k_output(result.output, system, self.cp2k_protocol, input_file_hash=input_hash, metadata_extra=metadata)
            return self._select_operation(parsed, operation)
