from __future__ import annotations

from xtbflow.validation import (
    ConnectivityEvidence,
    infer_binary_connectivity,
    observed_event,
    index_artifacts,
    sanitize_cp2k_calibration_report,
    sanitize_cp2k_convergence_report,
    sha256_file,
)


def test_endpoint_event_is_derived_from_calculated_connectivity():
    minus = infer_binary_connectivity(
        ("H", "H"), ((0.0, 0.0, 0.0), (0.7, 0.0, 0.0))
    )
    plus = infer_binary_connectivity(
        ("H", "H"), ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    )
    assert observed_event(
        ConnectivityEvidence(reactant_bonds=minus, product_bonds=plus)
    ) == ((0, 1, -1),)


def test_artifact_index_is_relative_and_hashed(tmp_path):
    root = tmp_path / "private-artifacts"
    nested = root / "water" / "run-1"
    nested.mkdir(parents=True)
    artifact = nested / "output.out"
    artifact.write_text("PROGRAM ENDED AT\n", encoding="utf-8")
    records = index_artifacts(root)
    assert records == [
        {
            "relative_path": "water/run-1/output.out",
            "size_bytes": artifact.stat().st_size,
            "sha256": sha256_file(artifact),
        }
    ]
    assert str(tmp_path) not in str(records)


def test_calibration_report_removes_artifact_root_and_case_paths():
    payload = {
        "schema": "xtbflow-cp2k-calibration-smoke/v1",
        "status": "pass",
        "calculator": "cp2k",
        "artifact_root": "/private/root",
        "cases": [
            {
                "system_id": "water",
                "status": "success",
                "metadata": {
                    "artifact_directory": "/private/root/water/run",
                    "artifact_persistence": "persistent",
                },
            }
        ],
    }
    public = sanitize_cp2k_calibration_report(
        payload,
        execution_source_commit="execution123",
        publication_source_commit="publication123",
        private_report_sha256="report-hash",
        calibration_script_sha256="script-hash",
        adapter_sha256="adapter-hash",
        artifacts=[
            {
                "relative_path": "water/run/output.out",
                "size_bytes": 10,
                "sha256": "artifact-hash",
            }
        ],
    )
    assert "artifact_root" not in public
    assert public["scientific_qualification"] is False
    assert public["execution_source_commit"] == "execution123"
    assert public["publication_source_commit"] == "publication123"
    assert public["cases"][0]["metadata"] == {
        "artifact_persistence": "persistent",
        "artifact_directory_recorded_in_private_run": True,
    }
    assert "/private/root" not in str(public)
    assert public["artifact_index"][0]["relative_path"] == "water/run/output.out"


def test_public_failure_text_redacts_absolute_paths():
    payload = {
        "schema": "xtbflow-cp2k-calibration-smoke/v1",
        "status": "fail",
        "cases": [
            {
                "status": "failure",
                "metadata": {
                    "artifact_directory": "/home/private-user/run",
                    "error": "CP2K failed while reading /scratch/private-user/input.inp",
                },
            }
        ],
    }
    public = sanitize_cp2k_calibration_report(
        payload,
        execution_source_commit="execution123",
        publication_source_commit="publication123",
        private_report_sha256="report-hash",
        calibration_script_sha256="script-hash",
        adapter_sha256="adapter-hash",
        artifacts=[{"relative_path": "run/error.out", "size_bytes": 1, "sha256": "x"}],
    )
    encoded = str(public)
    assert "/home/" not in encoded
    assert "/scratch/" not in encoded
    assert "[private-path]" in encoded


def test_convergence_report_removes_private_paths_and_retains_ledger_hash():
    payload = {
        "schema": "xtbflow-cp2k-convergence-smoke/v2",
        "status": "fail",
        "artifact_root": "/private/convergence",
        "calculator_calls": 3,
        "runs": [
            {
                "cutoff_ry": cutoff,
                "status": "success",
                "metadata": {
                    "artifact_directory": f"/private/convergence/{cutoff}",
                    "artifact_persistence": "persistent",
                },
            }
            for cutoff in (500.0, 600.0, 700.0)
        ],
        "comparisons_to_highest_cutoff": [
            {
                "cutoff_ry": 600.0,
                "reference_cutoff_ry": 700.0,
                "abs_energy_delta_hartree": 2.0e-5,
                "max_force_delta_hartree_per_angstrom": 2.0e-6,
            }
        ],
        "claim_limits": ["bounded ladder"],
    }
    public = sanitize_cp2k_convergence_report(
        payload,
        execution_source_commit="execution123",
        publication_source_commit="publication123",
        private_report_sha256="report-hash",
        ledger_sha256="ledger-hash",
        convergence_script_sha256="script-hash",
        adapter_sha256="adapter-hash",
        artifacts=[
            {
                "relative_path": "cutoff-500/output.out",
                "size_bytes": 10,
                "sha256": "artifact-hash",
            }
        ],
    )
    assert public["status"] == "fail"
    assert public["execution_source_commit"] == "execution123"
    assert public["publication_source_commit"] == "publication123"
    assert public["ledger_sha256"] == "ledger-hash"
    assert public["scientific_qualification"] is False
    assert "artifact_root" not in public
    assert "/private/" not in str(public)
    assert public["runs"][0]["metadata"][
        "artifact_directory_recorded_in_private_run"
    ] is True


def test_convergence_report_requires_multiple_runs():
    import pytest

    with pytest.raises(ValueError, match="at least two runs"):
        sanitize_cp2k_convergence_report(
            {
                "runs": [{"status": "success"}],
                "comparisons_to_highest_cutoff": [],
            },
            execution_source_commit="execution123",
            publication_source_commit="publication123",
            private_report_sha256="report-hash",
            ledger_sha256="ledger-hash",
            convergence_script_sha256="script-hash",
            adapter_sha256="adapter-hash",
            artifacts=[],
        )
