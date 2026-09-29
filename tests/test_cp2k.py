from __future__ import annotations

import pytest

from xtbflow.calculators import (
    BOHR_IN_ANGSTROM,
    CP2KAdapter,
    CP2KProtocol,
    CalculatorProtocolError,
    MolecularSystem,
    render_cp2k_input,
)


def protocol(**changes):
    values = dict(
        protocol_id="cp2k-toy-v1",
        functional="PBE",
        basis_set="DZVP-MOLOPT-GTH",
        pseudopotential="GTH-PBE",
        dispersion="DFTD3(BJ)",
        cutoff_ry=400.0,
        relative_cutoff_ry=50.0,
        scf_epsilon=1e-7,
        max_scf=80,
        charge=0,
        multiplicity=1,
        build_hash="cp2k-fixture-build",
        cp2k_version="2024.2-fixture",
        parameters={
            "basis_set_file": "/tmp/BASIS_MOLOPT",
            "pseudopotential_file": "/tmp/GTH_POTENTIALS",
            "dispersion_parameter_file": "/tmp/dftd3.dat",
            "reference_functional": "PBE",
            "threads": 2,
        },
    )
    values.update(changes)
    return CP2KProtocol(**values)


def system():
    return MolecularSystem(
        ("O", "H", "H"),
        ((0.0, 0.0, 0.0), (0.95, 0.0, 0.0), (-0.2, 0.9, 0.0)),
        0,
        1,
    )


def output():
    return """  ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]:        -75.123456789
  SCF run converged in 12 steps
  ATOMIC FORCES in [a.u.]
  # Atom   Kind   Element      X              Y              Z
      1      1      O        1.0            2.0            3.0
      2      2      H        4.0            5.0            6.0
      3      2      H        7.0            8.0            9.0
  SUM OF ATOMIC FORCES
  PROGRAM ENDED AT 2026-01-01 00:00:00
"""


def test_renderer_contains_isolated_poisson_force_and_d3_settings():
    rendered = render_cp2k_input(system(), protocol())
    assert "PSOLVER MT" in rendered
    assert "&FORCES" in rendered
    assert "TYPE DFTD3(BJ)" in rendered
    assert "REFERENCE_FUNCTIONAL PBE" in rendered


def test_open_shell_renderer_enables_lsd():
    open_shell_system = MolecularSystem(system().symbols, system().coordinates, 1, 2)
    rendered = render_cp2k_input(open_shell_system, protocol().for_state(1, 2))
    assert "    LSD" in rendered


def test_cp2k_parser_returns_energy_force_identity_and_units():
    adapter = CP2KAdapter(protocol(), runner=lambda rendered, item: output())
    result = adapter.evaluate(system())
    assert result.status == "success"
    assert result.energy == pytest.approx(-75.123456789)
    expected_forces = tuple(
        tuple(value / BOHR_IN_ANGSTROM for value in row)
        for row in ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0))
    )
    assert all(actual == pytest.approx(expected) for actual, expected in zip(result.forces, expected_forces))
    assert result.metadata["raw_force_unit"] == "hartree/bohr"
    assert result.metadata["normalized_force_unit"] == "hartree/angstrom"


def test_scf_failure_and_truncated_output_fail_closed():
    failed = CP2KAdapter(protocol(), runner=lambda rendered, item: "SCF run NOT converged")
    result = failed.evaluate(system())
    assert result.status == "not_converged"
    assert result.error_category == "convergence"
    truncated = CP2KAdapter(
        protocol(),
        runner=lambda rendered, item: output().replace(
            "      3      2      H        7.0            8.0            9.0\n", ""
        ),
    )
    with pytest.raises(CalculatorProtocolError, match="force section"):
        truncated.evaluate(system())


def test_dispersion_requires_a_parameter_file():
    with pytest.raises(ValueError, match="dispersion_parameter_file"):
        protocol(parameters={})


def test_state_derivation_keeps_charge_spin_in_protocol_identity():
    charged = protocol().for_state(1, 2)
    assert charged.charge == 1
    assert charged.multiplicity == 2
    assert charged.protocol_id.endswith(":q1:m2")
    assert charged.identity != protocol().identity


def test_periodic_protocol_requires_and_renders_explicit_cell():
    with pytest.raises(ValueError, match="cell_angstrom"):
        protocol(boundary="periodic")
    periodic = protocol(boundary="periodic", parameters={**protocol().parameters, "cell_angstrom": (12.0, 13.0, 14.0)})
    rendered = render_cp2k_input(system(), periodic)
    assert "ABC 12 13 14" in rendered
    assert "PERIODIC XYZ" in rendered


def test_runtime_execution_fields_are_not_physical_protocol_parameters():
    base = protocol()
    with pytest.raises(ValueError, match="would not affect rendered input"):
        protocol(parameters={**base.parameters, "threads": 8})
    with pytest.raises(ValueError, match="would not affect rendered input"):
        protocol(parameters={**base.parameters, "executable": "/tmp/cp2k.psmp"})
