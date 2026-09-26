from __future__ import annotations

import pytest

from xtbflow.calculators import CP2KAdapter, CP2KProtocol, CalculatorProtocolError, MolecularSystem, render_cp2k_input


def protocol(**changes):
    values = dict(
        protocol_id="cp2k-toy-v1",
        functional="PBE",
        basis_set="BASIS_MOLOPT",
        pseudopotential="GTH-PBE",
        dispersion="DFTD3(BJ)",
        cutoff_ry=400.0,
        relative_cutoff_ry=50.0,
        scf_epsilon=1e-7,
        max_scf=80,
        charge=0,
        multiplicity=1,
        build_hash="cp2k-fixture-build",
        cp2k_version="2025.1-fixture",
    )
    values.update(changes)
    return CP2KProtocol(**values)


def system():
    return MolecularSystem(("O", "H", "H"), ((0.0, 0.0, 0.0), (0.95, 0.0, 0.0), (-0.2, 0.9, 0.0)), 0, 1)


def output():
    return """  ENERGY| Total FORCE_EVAL ( QS ) energy (a.u.):        -75.123456789
  ATOMIC FORCES in [a.u.]
  # Atom   Kind   Element      X              Y              Z
      1      1      O        1.0            2.0            3.0
      2      2      H        4.0            5.0            6.0
      3      2      H        7.0            8.0            9.0
  SUM OF ATOMIC FORCES
"""


def test_cp2k_parser_returns_energy_force_identity_and_units():
    adapter = CP2KAdapter(protocol(), runner=lambda rendered, item: output())
    result = adapter.evaluate(system())
    assert result.status == "success"
    assert result.energy == pytest.approx(-75.123456789)
    assert result.forces == ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0))
    assert result.calculator_build_hash == "cp2k-fixture-build"
    assert result.input_file_hash
    assert result.path_status == "not_requested"


def test_protocol_and_rendered_input_change_when_physics_setting_changes():
    first = protocol()
    changed = protocol(cutoff_ry=500.0)
    assert first.identity != changed.identity
    assert render_cp2k_input(system(), first) != render_cp2k_input(system(), changed)


def test_scf_failure_and_truncated_output_fail_closed():
    failed = CP2KAdapter(protocol(), runner=lambda rendered, item: "SCF run NOT converged")
    result = failed.evaluate(system())
    assert result.status == "not_converged"
    assert result.error_category == "convergence"
    truncated = CP2KAdapter(protocol(), runner=lambda rendered, item: output().split("      3")[0])
    with pytest.raises(CalculatorProtocolError, match="force section"):
        truncated.evaluate(system())


def test_missing_protocol_state_and_backend_are_explicit():
    with pytest.raises(ValueError, match="functional"):
        protocol(functional="")
    unavailable = CP2KAdapter(protocol(), executable="definitely-not-cp2k")
    assert unavailable.capabilities.status in {"unavailable", "unknown"}
    with pytest.raises(Exception):
        unavailable.evaluate(system())
