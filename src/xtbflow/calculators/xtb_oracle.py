"""Independent original-xTB/tblite GFN2 comparison adapter boundary.

The public contract uses Angstrom coordinates, Hartree energies and
Hartree/Angstrom forces.  Both direct native APIs used here consume atomic-unit
coordinates and return Hartree/Bohr gradients; their adapters perform the
conversion at this boundary and retain implementation identity in result
metadata.  The ASE tblite wrapper is deliberately not used by this module.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
from importlib.util import find_spec
from shutil import which
from typing import Any, Callable, Mapping

import numpy as np

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
)
from xtbflow.runtime.ledger import BudgetTokenRequired, CalculatorCallToken


BOHR_IN_ANGSTROM = 0.529177210903
KELVIN_TO_HARTREE = 3.166811563e-6
ADAPTER_REVISION = "xtb-oracle-native-units-v1"


def _package_identity(implementation: str) -> tuple[str | None, str | None, str | None]:
    """Return version, module path and hashes for the observed implementation.

    The native interface module is included for tblite when available.  Hashing
    only the package ``__init__`` can miss a changed Python-to-native boundary,
    so the identity deliberately covers the file that defines ``Calculator``.
    """

    try:
        module = importlib.import_module(implementation)
    except Exception:
        return None, None, None
    path = getattr(module, "__file__", None)
    version = getattr(module, "__version__", None)
    try:
        distribution = "xtb" if implementation == "xtb" else "tblite"
        version = version or importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        pass
    identity_paths = [path] if path else []
    if implementation == "tblite":
        try:
            interface = importlib.import_module("tblite.interface")
        except Exception:
            interface = None
        interface_path = getattr(interface, "__file__", None)
        if interface_path:
            identity_paths.append(interface_path)
    digest = hashlib.sha256()
    observed = False
    for identity_path in identity_paths:
        try:
            with open(identity_path, "rb") as handle:
                contents = handle.read()
            digest.update(str(identity_path).encode())
            digest.update(contents)
            observed = True
        except OSError:
            continue
    build_hash = digest.hexdigest() if observed else None
    return str(version) if version is not None else None, path, build_hash


def _adapter_code_hash() -> str:
    """Hash this adapter source so cache identities include conversion code."""

    try:
        with open(__file__, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return "unavailable"


def _required_parameter(parameters: Mapping[str, Any], name: str) -> Any:
    """Read a protocol setting without silently selecting a backend default."""

    if name not in parameters:
        raise CalculatorProtocolError(f"protocol.parameters must explicitly define {name}")
    return parameters[name]


def _validate_environment(system: MolecularSystem) -> None:
    """Reject periodic inputs; the oracle path is molecular and non-periodic."""

    periodic = system.environment.get("periodic", False)
    if type(periodic) is not bool:
        raise CalculatorProtocolError("environment.periodic must be a boolean when provided")
    if periodic:
        raise CalculatorProtocolError("the GFN2 oracle adapter only supports non-periodic molecular systems")


def _xtb_evaluator(system: MolecularSystem, operation: str, parameters: Mapping[str, Any], protocol: CalculatorProtocol) -> CalculationResult:
    """Evaluate one system through the original xTB Python API."""

    from xtb.interface import Calculator, Param

    _validate_environment(system)
    numbers = np.asarray([_atomic_number(symbol) for symbol in system.symbols], dtype=np.int32)
    positions = np.asarray(system.coordinates, dtype=np.float64) / BOHR_IN_ANGSTROM
    # The original Python bindings identify the method with an enum even
    # though the public protocol uses the canonical ``GFN2-xTB`` spelling.
    calc = Calculator(Param.GFN2xTB, numbers, positions, system.charge, system.multiplicity - 1)
    calc.set_verbosity("muted")
    calc.set_accuracy(float(_required_parameter(parameters, "accuracy")))
    calc.set_max_iterations(int(_required_parameter(parameters, "max_iterations")))
    calc.set_electronic_temperature(int(_required_parameter(parameters, "electronic_temperature")))
    solvent = _required_parameter(parameters, "solvent")
    if solvent not in {None, "none", "None"}:
        raise CalculatorProtocolError("direct xTB oracle currently requires solvent=None")
    result = calc.singlepoint()
    energy = float(result.get_energy())
    gradient = np.asarray(result.get_gradient(), dtype=np.float64)
    # dE/dR_angstrom = dE/dR_bohr / (angstrom-per-bohr).
    forces = -gradient / BOHR_IN_ANGSTROM
    return _normalized_result(system, operation, energy, forces, implementation="xtb", native_gradient_unit="hartree/bohr", force_conversion="-gradient_hartree_per_bohr / 0.529177210903 = force_hartree_per_angstrom", protocol=protocol)


def _tblite_evaluator(system: MolecularSystem, operation: str, parameters: Mapping[str, Any], protocol: CalculatorProtocol) -> CalculationResult:
    """Evaluate one system through tblite's direct GFN2 interface."""

    from tblite.interface import Calculator

    _validate_environment(system)
    numbers = np.asarray([_atomic_number(symbol) for symbol in system.symbols], dtype=np.int32)
    # The direct native interface consumes Bohr, unlike the public contract.
    positions = np.asarray(system.coordinates, dtype=np.float64) / BOHR_IN_ANGSTROM
    calc = Calculator("GFN2-xTB", numbers, positions, charge=system.charge, uhf=system.multiplicity - 1)
    calc.set("verbosity", 0)
    calc.set("accuracy", float(_required_parameter(parameters, "accuracy")))
    calc.set("max-iter", int(_required_parameter(parameters, "max_iterations")))
    # tblite's temperature setter uses Hartree while the shared protocol
    # records the conventional Kelvin value used by xTB.
    calc.set("temperature", float(_required_parameter(parameters, "electronic_temperature")) * KELVIN_TO_HARTREE)
    solvent = _required_parameter(parameters, "solvent")
    if solvent not in {None, "none", "None"}:
        raise CalculatorProtocolError("direct tblite oracle currently requires solvent=None")
    result = calc.singlepoint()
    energy = float(result.get("energy"))
    gradient = np.asarray(result.get("gradient"), dtype=np.float64)
    forces = -gradient / BOHR_IN_ANGSTROM
    return _normalized_result(system, operation, energy, forces, implementation="tblite", native_gradient_unit="hartree/bohr", force_conversion="-gradient_hartree_per_bohr / 0.529177210903 = force_hartree_per_angstrom", protocol=protocol)


def _normalized_result(system: MolecularSystem, operation: str, energy: float, forces: np.ndarray, *, implementation: str, native_gradient_unit: str, force_conversion: str, protocol: CalculatorProtocol) -> CalculationResult:
    if operation == "energy":
        forces_value = None
    else:
        forces_value = tuple(tuple(float(value) for value in row) for row in forces.tolist())
    return CalculationResult(
        calculator=protocol.calculator,
        protocol_id=protocol.protocol_id,
        input_hash=system.input_hash,
        charge=system.charge,
        multiplicity=system.multiplicity,
        operation=operation,
        energy=energy if operation in {"energy", "energy_forces"} else None,
        forces=forces_value,
        metadata={
            "implementation": implementation,
            "adapter_revision": ADAPTER_REVISION,
            "adapter_code_hash": _adapter_code_hash(),
            "public_coordinate_unit": "angstrom",
            "native_coordinate_unit": "bohr",
            "native_energy_unit": "hartree",
            "native_gradient_unit": native_gradient_unit,
            "normalized_force_unit": "hartree/angstrom",
            "force_conversion": force_conversion,
        },
    )


def _atomic_number(symbol: str) -> int:
    """Resolve an element symbol without adding a chemistry dependency."""

    symbols = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split()
    try:
        return symbols.index(symbol) + 1
    except ValueError as exc:
        raise CalculatorProtocolError(f"unsupported element symbol: {symbol}") from exc


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
        require_budget_token: bool = False,
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
        self._direct_backend = False
        observed_version, self.module_path, observed_hash = _package_identity(implementation)
        self.version = version or observed_version
        self.build_hash = build_hash or observed_hash
        self.require_budget_token = bool(require_budget_token)
        required = {"accuracy", "max_iterations", "electronic_temperature", "solvent"}
        if self._evaluator is None and implementation in {"xtb", "tblite"} and required.issubset(self.protocol.parameters):
            self._direct_backend = True
            self._evaluator = lambda item, operation: (
                _xtb_evaluator(item, operation, self.protocol.parameters, self.protocol)
                if implementation == "xtb"
                else _tblite_evaluator(item, operation, self.protocol.parameters, self.protocol)
            )

    @property
    def capabilities(self) -> CalculatorCapabilities:
        if self._evaluator is not None:
            if self._direct_backend:
                detail = "direct backend is callable; numerical qualification is not asserted"
                qualification = "installed" if self.version and self.build_hash else "callable"
            else:
                detail = "injected evaluator; external installation and numerical qualification are not asserted"
                qualification = "callable"
            return CalculatorCapabilities(
                "xtb_oracle",
                "unknown",
                detail,
                self.version,
                self.build_hash,
                tuple(sorted({"energy", "forces", "energy_forces"})),
                (self.protocol.backend,),
                qualification,
            )
        module = find_spec(self.implementation)
        executable = which(self.implementation)
        if module is None and executable is None:
            return CalculatorCapabilities("xtb_oracle", "unavailable", f"no {self.implementation} module or executable was observed", qualification="unavailable")
        return CalculatorCapabilities("xtb_oracle", "unknown", f"{self.implementation} was observed but its API is not qualified", self.version, self.build_hash, qualification="installed")

    def evaluate(
        self,
        system: MolecularSystem,
        *,
        operation: str = "energy_forces",
        budget_token: CalculatorCallToken | None = None,
    ) -> CalculationResult:
        if operation not in {"energy", "forces", "energy_forces"}:
            raise CalculatorProtocolError(f"unsupported operation: {operation}")
        if self._evaluator is None:
            raise CalculatorUnavailable(f"independent {self.implementation} backend is not qualified or injected")
        if self.require_budget_token and budget_token is None:
            raise BudgetTokenRequired("xTB oracle production calls require a calculator budget token")
        if budget_token is not None:
            budget_token.consume()
        try:
            output = self._evaluator(system, operation)
        except CalculatorError:
            raise
        except Exception as exc:
            if self._direct_backend and ("converg" in str(exc).lower() or "scf" in str(exc).lower()):
                raise CalculatorConvergenceError(f"independent evaluator did not converge: {exc}") from exc
            raise RuntimeError(f"independent evaluator failed before producing a contract result: {exc}") from exc
        return coerce_backend_output(output, system, self.protocol, operation=operation)


GFN2OracleAdapter = XTBOracleAdapter
