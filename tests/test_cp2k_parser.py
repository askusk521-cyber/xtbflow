from __future__ import annotations

from pathlib import Path

import pytest

from xtbflow.calculators import BOHR_IN_ANGSTROM, CP2KAdapter, CP2KProtocol, CP2KRunnerResult, CalculatorError, CalculatorProtocolError, MolecularSystem, finite_difference_forces, render_cp2k_input


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
        parameters={
            "dispersion_parameter_file": "/tmp/dftd3.dat",
            "reference_functional": "PBE",
        },
    )
    values.update(changes)
    return CP2KProtocol(**values)


def system():
    return MolecularSystem(("O", "H", "H"), ((0.0, 0.0, 0.0), (0.95, 0.0, 0.0), (-0.2, 0.9, 0.0)), 0, 1)


def output():
    return """  ENERGY| Total FORCE_EVAL ( QS ) energy (a.u.):        -75.123456789
  SCF run converged in 12 steps
  ATOMIC FORCES in [a.u.]
  # Atom   Kind   Element      X              Y              Z
      1      1      O        1.0            2.0            3.0
      2      2      H        4.0            5.0            6.0
      3      2      H        7.0            8.0            9.0
  SUM OF ATOMIC FORCES
  PROGRAM ENDED AT 2026-01-01 00:00:00
"""


def test_cp2k_parser_returns_energy_force_identity_and_units():
    adapter = CP2KAdapter(protocol(), runner=lambda rendered, item: output())
    result = adapter.evaluate(system())
    assert result.status == "success"
    assert result.energy == pytest.approx(-75.123456789)
    expected_forces = tuple(tuple(value / BOHR_IN_ANGSTROM for value in row) for row in ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)))
    assert all(actual == pytest.approx(expected) for actual, expected in zip(result.forces, expected_forces))
    assert result.metadata["raw_force_unit"] == "hartree/bohr"
    assert result.metadata["normalized_force_unit"] == "hartree/angstrom"
    assert result.calculator_build_hash == "cp2k-fixture-build"
    assert result.input_file_hash
    assert result.path_status == "not_requested"


def test_protocol_and_rendered_input_change_when_physics_setting_changes():
    first = protocol()
    changed = protocol(cutoff_ry=500.0)
    assert first.identity != changed.identity
    assert render_cp2k_input(system(), first) != render_cp2k_input(system(), changed)


def test_unapplied_protocol_parameters_fail_closed():
    with pytest.raises(ValueError, match="would not affect rendered input"):
        protocol(parameters={"solvent_epsilon": 78.4})


def test_positive_scf_convergence_is_required():
    without_marker = output().replace("  SCF run converged in 12 steps\n", "")
    with pytest.raises(CalculatorProtocolError, match="positive SCF convergence"):
        CP2KAdapter(protocol(), runner=lambda rendered, item: without_marker).evaluate(system())


def test_structured_runner_persists_input_stdout_stderr_and_output(tmp_path):
    artifacts = tmp_path / "cp2k-artifacts"
    adapter = CP2KAdapter(
        protocol(),
        runner=lambda rendered, item: CP2KRunnerResult(output=output(), stdout="launcher stdout\n", stderr="launcher stderr\n"),
        artifact_dir=artifacts,
        require_artifacts=True,
    )
    result = adapter.evaluate(system())
    artifact_directory = result.metadata["artifact_directory"]
    assert result.metadata["artifact_persistence"] == "persistent"
    assert artifact_directory
    root = Path(artifact_directory)
    assert (root / "input.inp").read_text(encoding="utf-8")
    assert (root / "stdout.txt").read_text(encoding="utf-8") == "launcher stdout\n"
    assert (root / "stderr.txt").read_text(encoding="utf-8") == "launcher stderr\n"
    assert (root / "output.out").read_text(encoding="utf-8") == output()


def test_nonzero_structured_runner_exit_is_rejected_and_artifacts_remain(tmp_path):
    adapter = CP2KAdapter(
        protocol(),
        runner=lambda rendered, item: {"output": output(), "stderr": "fatal", "returncode": 2},
        artifact_dir=tmp_path,
        require_artifacts=True,
    )
    with pytest.raises(CalculatorError, match="exited with code 2"):
        adapter.evaluate(system())
    assert list(tmp_path.glob("xtbflow-cp2k-*/output.out"))


def test_energy_and_force_operations_share_an_angstrom_finite_difference_contract():
    def harmonic_runner(rendered, item):
        coordinates_bohr = [[value / BOHR_IN_ANGSTROM for value in row] for row in item.coordinates]
        energy = 0.5 * sum(value * value for row in coordinates_bohr for value in row)
        rows = [
            f"      {index}      {1 if symbol == 'O' else 2}      {symbol}        {row[0]}            {row[1]}            {row[2]}"
            for index, (symbol, row) in enumerate(zip(item.symbols, coordinates_bohr), start=1)
        ]
        return "\n".join(
            [
                f"  ENERGY| Total FORCE_EVAL ( QS ) energy (a.u.):        {-energy}",
                "  SCF run converged in 1 steps",
                "  ATOMIC FORCES in [a.u.]",
                *rows,
                "  SUM OF ATOMIC FORCES",
                "  PROGRAM ENDED AT 2026-01-01 00:00:00",
                "",
            ]
        )

    adapter = CP2KAdapter(protocol(), runner=harmonic_runner)
    energy = adapter.evaluate(system(), operation="energy")
    forces = adapter.evaluate(system(), operation="forces")
    assert energy.operation == "energy" and energy.forces is None
    assert forces.operation == "forces" and forces.energy is None
    estimated = finite_difference_forces(adapter, system(), step=1e-5)
    assert any(abs(value) > 1e-8 for row in forces.forces for value in row)
    for estimated_row, force_row in zip(estimated, forces.forces):
        assert estimated_row == pytest.approx(force_row, rel=1e-5, abs=1e-5)


@pytest.mark.parametrize("relative_artifact_dir", [True, False])
def test_real_subprocess_resolves_artifact_workspace_and_relative_executable(tmp_path, monkeypatch, relative_artifact_dir):
    executable = tmp_path / "fake-cp2k"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import argparse, pathlib\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('-i')\n"
        "parser.add_argument('-o')\n"
        "args = parser.parse_args()\n"
        "input_path = pathlib.Path(args.i)\n"
        "assert input_path.is_file(), input_path\n"
        f"pathlib.Path(args.o).write_text({output()!r})\n"
        "print('fake stdout')\n",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | 0o111)
    monkeypatch.chdir(tmp_path)
    artifact_dir = Path("runs/cp2k") if relative_artifact_dir else tmp_path / "runs-absolute" / "cp2k"
    adapter = CP2KAdapter(protocol(), executable="fake-cp2k", artifact_dir=artifact_dir, require_artifacts=True)
    result = adapter.evaluate(system())
    root = Path(result.metadata["artifact_directory"])
    assert root.is_absolute()
    assert (root / "input.inp").is_file()
    assert (root / "output.out").read_text(encoding="utf-8") == output()
    assert (root / "stdout.txt").read_text(encoding="utf-8") == "fake stdout\n"


@pytest.mark.parametrize("emit_output", [True, False])
def test_real_subprocess_timeout_preserves_logs_and_timeout_classification(tmp_path, emit_output):
    executable = tmp_path / "fake-cp2k-timeout"
    output_lines = (
        "print('before-timeout-stdout', flush=True)\n"
        "print('before-timeout-stderr', file=sys.stderr, flush=True)\n"
        if emit_output
        else ""
    )
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, time\n"
        + output_lines
        + "time.sleep(10)\n",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | 0o111)
    artifacts = tmp_path / "timeout-artifacts"
    adapter = CP2KAdapter(
        protocol(),
        executable=str(executable),
        timeout_seconds=0.1,
        artifact_dir=artifacts,
        require_artifacts=True,
    )
    with pytest.raises(CalculatorError, match="timed out"):
        adapter.evaluate(system())
    roots = list(artifacts.glob("xtbflow-cp2k-*"))
    assert len(roots) == 1
    stdout = (roots[0] / "stdout.txt").read_text(encoding="utf-8")
    stderr = (roots[0] / "stderr.txt").read_text(encoding="utf-8")
    if emit_output:
        assert "before-timeout-stdout" in stdout
        assert "before-timeout-stderr" in stderr
    else:
        assert stdout == ""
        assert stderr == ""


def test_scf_failure_and_truncated_output_fail_closed():
    failed = CP2KAdapter(protocol(), runner=lambda rendered, item: "SCF run NOT converged")
    result = failed.evaluate(system())
    assert result.status == "not_converged"
    assert result.error_category == "convergence"
    truncated = CP2KAdapter(protocol(), runner=lambda rendered, item: output().replace("      3      2      H        7.0            8.0            9.0\n", ""))
    with pytest.raises(CalculatorProtocolError, match="force section"):
        truncated.evaluate(system())


def test_missing_protocol_state_and_backend_are_explicit():
    with pytest.raises(ValueError, match="functional"):
        protocol(functional="")
    unavailable = CP2KAdapter(protocol(), executable="definitely-not-cp2k")
    assert unavailable.capabilities.status in {"unavailable", "unknown"}
    assert unavailable.capabilities.qualification == "unavailable"
    assert CP2KAdapter(protocol(), runner=lambda rendered, item: output()).capabilities.qualification == "callable"
    with pytest.raises(Exception):
        unavailable.evaluate(system())


def test_cp2k_parser_requires_completion_and_preserves_atom_order():
    with pytest.raises(CalculatorProtocolError, match="completion marker"):
        CP2KAdapter(protocol(), runner=lambda rendered, item: output().replace("  PROGRAM ENDED AT 2026-01-01 00:00:00", "")).evaluate(system())
    reordered = output().replace("      1      1      O", "      2      1      O", 1)
    with pytest.raises(CalculatorProtocolError, match="atom order"):
        CP2KAdapter(protocol(), runner=lambda rendered, item: reordered).evaluate(system())
