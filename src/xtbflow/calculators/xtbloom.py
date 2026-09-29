"""Explicit xTBloom adapter boundary.

The public repository does not assume an xTBloom Python or CLI API.  Until a
qualified build is injected, this adapter reports the installation as unknown
or unavailable and refuses to invent energies or forces.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
from importlib.util import find_spec
from pathlib import Path
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
ADAPTER_REVISION = "xtbloom-native-units-v1"


def _package_identity() -> tuple[str | None, str | None, str | None]:
    """Identify the installed wrapper sources without claiming qualification."""

    try:
        module = importlib.import_module("xtbloom")
    except Exception:
        return None, None, None
    version = getattr(module, "__version__", None)
    try:
        version = version or importlib.metadata.version("xtbloom")
    except importlib.metadata.PackageNotFoundError:
        pass
    module_path = getattr(module, "__file__", None)
    paths = [module_path]
    # The public Calculator wrapper lives in interface.py; library.py carries
    # the ctypes ABI loader. Hash both so a changed Python/native boundary
    # cannot silently reuse a stale calculator identity.
    for name in ("xtbloom.interface", "xtbloom.library"):
        try:
            submodule = importlib.import_module(name)
        except Exception:
            continue
        paths.append(getattr(submodule, "__file__", None))
        if name == "xtbloom.library":
            locator = getattr(submodule, "library_path", None)
            if callable(locator):
                try:
                    native_path = locator()
                except Exception:
                    native_path = None
                if native_path and Path(native_path).is_file():
                    paths.append(str(native_path))
    if module_path:
        package_dir = Path(module_path).parent
        for runtime_dir in (package_dir / "lib", package_dir / "lib64", package_dir / "bin"):
            paths.extend(str(path) for path in sorted(runtime_dir.glob("libxtbloom.so*")))
            paths.extend(str(path) for path in sorted(runtime_dir.glob("libxtbloom.dylib*")))
            paths.extend(str(path) for path in sorted(runtime_dir.glob("xtbloom.dll")))
    digest = hashlib.sha256()
    observed = False
    for path in paths:
        if not path:
            continue
        try:
            digest.update(str(path).encode())
            with open(path, "rb") as handle:
                digest.update(handle.read())
            observed = True
        except OSError:
            continue
    return (
        str(version) if version is not None else None,
        getattr(module, "__file__", None),
        digest.hexdigest() if observed else None,
    )


def _required_parameter(parameters: Mapping[str, Any], name: str) -> Any:
    if name not in parameters:
        raise CalculatorProtocolError(f"protocol.parameters must explicitly define {name}")
    return parameters[name]


def _adapter_code_hash() -> str:
    """Hash this adapter so normalized results retain conversion provenance."""

    try:
        with open(__file__, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return "unavailable"


def _validate_environment(system: MolecularSystem) -> None:
    periodic = system.environment.get("periodic", False)
    if type(periodic) is not bool:
        raise CalculatorProtocolError("environment.periodic must be a boolean when provided")
    if periodic:
        raise CalculatorProtocolError("the xTBloom adapter currently supports non-periodic molecular systems only")


def _atomic_number(symbol: str) -> int:
    symbols = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split()
    try:
        return symbols.index(symbol) + 1
    except ValueError as exc:
        raise CalculatorProtocolError(f"unsupported element symbol: {symbol}") from exc


def _xtbloom_evaluator(system: MolecularSystem, operation: str, parameters: Mapping[str, Any], protocol: CalculatorProtocol) -> CalculationResult:
    """Evaluate one public Å input through xTBloom's atomic-unit API."""

    from xtbloom import Calculator

    _validate_environment(system)
    method = protocol.method
    if method not in {"GFN1", "GFN1-xTB", "GFN2", "GFN2-xTB"}:
        raise CalculatorProtocolError(
            "xTBloom supports only GFN1-xTB or GFN2-xTB methods"
        )
    numbers = np.asarray([_atomic_number(symbol) for symbol in system.symbols], dtype=np.int32)
    positions_bohr = np.asarray(system.coordinates, dtype=np.float64) / BOHR_IN_ANGSTROM
    initialization = _required_parameter(parameters, "initialization")
    if initialization not in {"fresh", "warm"}:
        raise CalculatorProtocolError("xTBloom initialization must be 'fresh' or 'warm'")
    calc_kwargs = {
        "charge": system.charge,
        "multiplicity": system.multiplicity,
        "backend": protocol.backend,
        "max_scc_iterations": int(_required_parameter(parameters, "max_scc_iterations")),
        "charge_tolerance": float(_required_parameter(parameters, "charge_tolerance")),
        "energy_tolerance": float(_required_parameter(parameters, "energy_tolerance")),
        "electronic_temperature": float(_required_parameter(parameters, "electronic_temperature")),
        "warm_start": initialization == "warm",
    }
    if "cpu_threads" in parameters:
        calc_kwargs["cpu_threads"] = int(parameters["cpu_threads"])
    with Calculator(method, numbers, positions_bohr, **calc_kwargs) as calc:
        result = calc.singlepoint()
    energy = float(result.get("energy"))
    native_forces = np.asarray(result.get("forces"), dtype=np.float64)
    forces = native_forces / BOHR_IN_ANGSTROM
    return CalculationResult(
        calculator=protocol.calculator,
        protocol_id=protocol.protocol_id,
        input_hash=system.input_hash,
        charge=system.charge,
        multiplicity=system.multiplicity,
        operation=operation,
        energy=energy if operation in {"energy", "energy_forces"} else None,
        forces=None if operation == "energy" else tuple(tuple(float(value) for value in row) for row in forces.tolist()),
        metadata={
            "implementation": "xtbloom",
            "adapter_revision": ADAPTER_REVISION,
            "adapter_code_hash": _adapter_code_hash(),
            "method": method,
            "native_coordinate_unit": "bohr",
            "native_force_unit": "hartree/bohr",
            "normalized_coordinate_unit": "angstrom",
            "normalized_force_unit": "hartree/angstrom",
            "force_conversion": "native_force_hartree_per_bohr / 0.529177210903 = force_hartree_per_angstrom",
            "initialization": initialization,
        },
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
        require_budget_token: bool = False,
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
        self._direct_backend = False
        observed_version, self.module_path, observed_hash = _package_identity()
        self.version = version or observed_version
        self.build_hash = build_hash or observed_hash
        self.require_budget_token = bool(require_budget_token)
        required = {"max_scc_iterations", "charge_tolerance", "energy_tolerance", "electronic_temperature", "initialization"}
        if self._evaluator is None and required.issubset(self.protocol.parameters):
            try:
                module = importlib.import_module("xtbloom")
            except Exception:
                pass
            else:
                if callable(getattr(module, "Calculator", None)):
                    self._direct_backend = True
                    self._evaluator = lambda item, operation: _xtbloom_evaluator(item, operation, self.protocol.parameters, self.protocol)

    @property
    def capabilities(self) -> CalculatorCapabilities:
        if self._evaluator is not None:
            qualification = "installed" if self._direct_backend else "callable"
            detail = "direct backend is callable; numerical qualification is not asserted" if self._direct_backend else "injected evaluator; external installation and numerical qualification are not asserted"
            return CalculatorCapabilities(
                "xtbloom",
                "unknown",
                detail,
                self.version,
                self.build_hash,
                tuple(sorted({"energy", "forces", "energy_forces"})),
                (self.protocol.backend,),
                qualification,
            )
        module = find_spec("xtbloom")
        executable = which("xtbloom")
        if module is None and executable is None:
            return CalculatorCapabilities("xtbloom", "unavailable", "no xtbloom module or executable was observed", qualification="unavailable")
        return CalculatorCapabilities("xtbloom", "unknown", "xtbloom was observed but its API is not qualified", self.version, self.build_hash, qualification="installed")

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
            raise CalculatorUnavailable("xTBloom is not qualified or injected in this environment")
        if self.require_budget_token and budget_token is None:
            raise BudgetTokenRequired("xTBloom production calls require a calculator budget token")
        if budget_token is not None:
            budget_token.consume()
        try:
            output = self._evaluator(system, operation)
        except CalculatorError:
            raise
        except Exception as exc:
            if self._direct_backend and ("converg" in str(exc).lower() or "scc" in str(exc).lower()):
                raise CalculatorConvergenceError(f"xTBloom evaluator did not converge: {exc}") from exc
            raise RuntimeError(f"xTBloom evaluator failed before producing a contract result: {exc}") from exc
        return coerce_backend_output(output, system, self.protocol, operation=operation)
