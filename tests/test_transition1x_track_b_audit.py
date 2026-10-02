from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pytest

from xtbflow.data.transition1x_audit import (
    Transition1xAuditError,
    audit_transition1x_pickle,
)


def _object_array(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def _dataset(*, mismatched_product_atoms: bool = False):
    numbers_a = np.asarray([6, 1, 1, 1, 1], dtype=np.int64)
    numbers_b = np.asarray([8, 1, 1], dtype=np.int64)
    product_a = numbers_a.copy()
    if mismatched_product_atoms:
        product_a[0] = 7

    def branch(first_numbers):
        numbers = [first_numbers, numbers_b]
        positions = [
            np.arange(first_numbers.size * 3, dtype=np.float64).reshape(-1, 3) / 10,
            np.arange(numbers_b.size * 3, dtype=np.float64).reshape(-1, 3) / 20,
        ]
        return {
            "positions": _object_array(positions),
            "charges": _object_array(numbers),
            "num_atoms": np.asarray([first_numbers.size, numbers_b.size]),
            "rxn": np.asarray(["rxn-a", "rxn-b"]),
        }

    return {
        "reactant": branch(numbers_a),
        "product": branch(product_a),
        "transition_state": branch(numbers_a),
        "use_ind": np.asarray([0], dtype=np.int64),
        "single_fragment": np.asarray([True, True]),
    }


def _write_pickle(path: Path, payload) -> str:
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit(path: Path, expected_sha256: str):
    return audit_transition1x_pickle(
        path,
        expected_sha256=expected_sha256,
        expected_size_bytes=path.stat().st_size,
        source_revision="pinned-test-revision",
        source_locator="https://example.test/transition1x",
        license_record="CC-BY-4.0",
        upstream_code_commit="a" * 40,
    )


def test_transition1x_audit_hash_gates_endpoint_structure(tmp_path):
    asset = tmp_path / "train_rpsb_all.pkl"
    expected_sha256 = _write_pickle(asset, _dataset())
    report = _audit(asset, expected_sha256)

    assert report["source"]["identity_verified_before_deserialization"] is True
    assert report["source"]["asset_name"] == asset.name
    assert report["source"]["deserializer"] == "restricted_numpy_pickle"
    assert report["source"]["archive_code_executed"] is False
    assert str(tmp_path) not in str(report)
    assert report["structure"]["record_count"] == 2
    assert report["structure"]["endpoint_atom_rows_match"] is True
    split = report["structure"]["split"]
    assert {key: split[key] for key in (
        "representation", "selected_count", "complement_count"
    )} == {
        "representation": "index_list",
        "selected_count": 1,
        "complement_count": 1,
    }
    assert len(split["selected_indices_sha256"]) == 64
    assert report["field_evidence"]["charges_field_interpretation"] == "atomic_numbers"
    assert report["field_evidence"]["formal_charge_field_present"] is False
    assert report["field_evidence"]["multiplicity_field_present"] is False
    assert report["track_b_gate"]["event_geometry_training_ready"] is False
    assert report["track_b_gate"]["admission"] == "quarantine_diagnostic_only"
    assert report["scientific_claim_allowed"] is False


def test_transition1x_audit_refuses_hash_mismatch_before_unpickling(tmp_path):
    asset = tmp_path / "untrusted.pkl"
    asset.write_bytes(b"not a pickle")
    with pytest.raises(Transition1xAuditError, match="refusing to unpickle"):
        _audit(asset, "0" * 64)


def test_transition1x_audit_rejects_endpoint_atom_row_mismatch(tmp_path):
    asset = tmp_path / "mismatch.pkl"
    expected_sha256 = _write_pickle(
        asset,
        _dataset(mismatched_product_atoms=True),
    )
    with pytest.raises(Transition1xAuditError, match="atom rows differ"):
        _audit(asset, expected_sha256)


def test_transition1x_audit_restricted_unpickler_blocks_foreign_globals(tmp_path):
    asset = tmp_path / "foreign.pkl"
    expected_sha256 = _write_pickle(asset, Path("not-numeric"))
    with pytest.raises(Transition1xAuditError, match="blocked pickle global"):
        _audit(asset, expected_sha256)


def test_transition1x_audit_rejects_declared_record_count_mismatch(tmp_path):
    asset = tmp_path / "count-mismatch.pkl"
    expected_sha256 = _write_pickle(asset, _dataset())
    with pytest.raises(Transition1xAuditError, match="record count mismatch"):
        audit_transition1x_pickle(
            asset,
            expected_sha256=expected_sha256,
            expected_size_bytes=asset.stat().st_size,
            source_revision="pinned-test-revision",
            source_locator="https://example.test/transition1x",
            license_record="unspecified",
            upstream_code_commit="a" * 40,
            expected_record_count=3,
        )
