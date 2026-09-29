from __future__ import annotations

import json

import pytest

from xtbflow.calculators import (
    CP2KRuntimeIdentity,
    build_cp2k_protocol,
    inspect_cp2k_runtime,
    load_cp2k_protocol_document,
    physical_protocol_identity,
    physical_protocol_record,
)


def document():
    return {
        "protocol": {
            "protocol_id": "cp2k-reference-v0.1",
            "functional": "PBE",
            "basis_set": "DZVP-MOLOPT-GTH",
            "pseudopotential": "GTH-PBE",
            "dispersion": "DFTD3(BJ)",
            "cutoff_ry": 600,
            "relative_cutoff_ry": 50,
            "scf_epsilon": 1e-7,
            "max_scf": 80,
            "charge": "REQUIRED_PER_SYSTEM",
            "multiplicity": "REQUIRED_PER_SYSTEM",
            "build_hash": "cp2k-2024.2-8222f9c",
            "cp2k_version": "2024.2",
            "boundary": "isolated",
            "solvent_model": "none",
            "path_status": "not_requested",
            "parameters": {
                "basis_set_file": "BASIS_MOLOPT",
                "pseudopotential_file": "GTH_POTENTIALS",
                "dispersion_parameter_file": "dftd3.dat",
                "reference_functional": "PBE",
                "basis_by_element": {"H": "DZVP-MOLOPT-GTH"},
                "pseudopotential_by_element": {"H": "GTH-PBE"},
            },
        }
    }


def runtime(tmp_path):
    for name in ("BASIS_MOLOPT", "GTH_POTENTIALS", "dftd3.dat"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    return CP2KRuntimeIdentity(
        executable="/opt/cp2k.psmp",
        version="2024.2",
        source_revision="8222f9c",
        build_hash="cp2k-2024.2-8222f9c",
        executable_sha256="fixture",
        data_dir=str(tmp_path),
    )


def test_build_protocol_binds_state_and_resolves_data_files(tmp_path):
    protocol = build_cp2k_protocol(
        document(), runtime(tmp_path), charge=1, multiplicity=2
    )
    assert protocol.protocol_id.endswith(":q1:m2")
    assert protocol.charge == 1
    assert protocol.multiplicity == 2
    assert protocol.parameters["basis_set_file"] == str(
        (tmp_path / "BASIS_MOLOPT").resolve()
    )
    assert protocol.build_hash == "cp2k-2024.2-8222f9c"


def test_build_protocol_rejects_runtime_identity_mismatch(tmp_path):
    observed = runtime(tmp_path)
    bad = CP2KRuntimeIdentity(
        executable=observed.executable,
        version="2025.1",
        source_revision=observed.source_revision,
        build_hash="cp2k-2025.1-8222f9c",
        executable_sha256=observed.executable_sha256,
        data_dir=observed.data_dir,
    )
    with pytest.raises(ValueError, match="version mismatch"):
        build_cp2k_protocol(document(), bad, charge=0, multiplicity=1)


def test_load_protocol_document_requires_protocol_object(tmp_path):
    path = tmp_path / "protocol.yaml"
    path.write_text(json.dumps({"status": "missing"}), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol object"):
        load_cp2k_protocol_document(path)


def test_inspect_runtime_parses_version_and_hides_private_paths(tmp_path):
    executable = tmp_path / "cp2k.psmp"
    executable.write_text(
        "#!/bin/sh\necho 'CP2K version 2024.2'\necho 'Source code revision 8222f9c'\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    data_dir = tmp_path / "share"
    data_dir.mkdir()
    observed = inspect_cp2k_runtime(executable, data_dir=data_dir)
    assert observed.version == "2024.2"
    assert observed.source_revision == "8222f9c"
    assert observed.build_hash == "cp2k-2024.2-8222f9c"
    public = observed.public_record()
    assert public["executable_name"] == "cp2k.psmp"
    assert "data_dir" not in public
    assert str(tmp_path) not in str(public)


def test_build_protocol_rejects_build_mismatch(tmp_path):
    payload = document()
    payload["protocol"]["build_hash"] = "cp2k-2024.2-other"
    with pytest.raises(ValueError, match="build mismatch"):
        build_cp2k_protocol(payload, runtime(tmp_path), charge=0, multiplicity=1)


def test_physical_protocol_identity_is_path_independent(tmp_path):
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()
    left = runtime(left_dir)
    right = runtime(right_dir)
    left_identity = physical_protocol_identity(
        document(), left, charge=0, multiplicity=1
    )
    right_identity = physical_protocol_identity(
        document(), right, charge=0, multiplicity=1
    )
    assert left_identity == right_identity
    record = physical_protocol_record(
        document(), left, charge=0, multiplicity=1
    )
    assert record["parameters"]["basis_set_file"]["name"] == "BASIS_MOLOPT"
    assert "sha256" in record["parameters"]["basis_set_file"]
    assert str(tmp_path) not in str(record)


def test_physical_protocol_identity_changes_with_state(tmp_path):
    observed = runtime(tmp_path)
    neutral = physical_protocol_identity(
        document(), observed, charge=0, multiplicity=1
    )
    charged = physical_protocol_identity(
        document(), observed, charge=1, multiplicity=2
    )
    assert neutral != charged
