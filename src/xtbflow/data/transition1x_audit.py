"""Hash-gated structural audit for the public Transition1x endpoint pickle.

The public pickle is useful for endpoint-conditioned geometry diagnostics, but
it does not by itself provide the electronic-state, event-label, or family
metadata required by the Track-B joint event--geometry gate.  This module
therefore verifies source identity before restricted deserialization and
reports observed fields without promoting records into a scientific manifest.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


TRANSITION1X_AUDIT_SCHEMA = "xtbflow-transition1x-source-audit/v1"
_REQUIRED_BRANCHES = ("reactant", "product", "transition_state")
_REQUIRED_BRANCH_FIELDS = ("positions", "charges")
_SHARED_ALIGNMENT_FIELDS = ("rxn", "formula", "num_atoms")
_SHA256_LENGTH = 64
_MD5_LENGTH = 32


class Transition1xAuditError(ValueError):
    """Raised when source identity or the endpoint triple contract fails."""


class _RestrictedUnpickler(pickle.Unpickler):
    """Permit only NumPy reconstruction primitives used by the public asset."""

    _ALLOWED_GLOBALS = {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy.core.numeric", "_frombuffer"),
        ("numpy._core.numeric", "_frombuffer"),
    }

    def find_class(self, module: str, name: str):
        if (module, name) not in self._ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"blocked pickle global {module}.{name}")
        return super().find_class(module, name)


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_numeric_pickle(path: Path) -> Any:
    try:
        with path.open("rb") as handle:
            return _RestrictedUnpickler(handle).load()
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError) as exc:
        raise Transition1xAuditError(
            f"restricted Transition1x deserialization failed: {exc}"
        ) from exc


def load_transition1x_pickle(
    path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int | None = None,
) -> Mapping[str, Any]:
    """Load a pinned numeric Transition1x pickle after byte identity checks.

    This is intentionally a small public boundary for development consumers.
    It performs the same hash gate and restricted NumPy-only deserialization as
    :func:`audit_transition1x_pickle`, but does not claim that the resulting
    payload satisfies the Track-B scientific admission contract.
    """

    if not isinstance(expected_sha256, str) or len(expected_sha256) != _SHA256_LENGTH:
        raise Transition1xAuditError("expected_sha256 must be a lowercase SHA-256")
    if any(character not in "0123456789abcdef" for character in expected_sha256):
        raise Transition1xAuditError("expected_sha256 must be a lowercase SHA-256")
    if not path.is_file():
        raise Transition1xAuditError("Transition1x asset does not exist")
    observed_size = path.stat().st_size
    if expected_size_bytes is not None and observed_size != expected_size_bytes:
        raise Transition1xAuditError(
            f"asset size mismatch: expected {expected_size_bytes}, observed {observed_size}"
        )
    observed_sha256 = hash_file(path)
    if observed_sha256 != expected_sha256:
        raise Transition1xAuditError("asset SHA-256 mismatch; refusing to unpickle")
    payload = _load_numeric_pickle(path)
    return _require_mapping(payload, "Transition1x payload")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Transition1xAuditError(f"{name} must be a mapping")
    return value


def _record_sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            raise Transition1xAuditError(f"{name} must contain a record axis")
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    raise Transition1xAuditError(f"{name} must be an array or sequence")


def _normalise_split(value: Any, record_count: int) -> dict[str, Any]:
    sequence = _record_sequence(value, "use_ind")
    items = np.asarray(sequence)
    if items.ndim != 1:
        raise Transition1xAuditError("use_ind must be one-dimensional")
    if items.size == record_count and items.dtype.kind == "b":
        selected = np.flatnonzero(items).astype(int).tolist()
        representation = "boolean_mask"
    else:
        try:
            selected = [int(item) for item in items.tolist()]
        except (TypeError, ValueError) as exc:
            raise Transition1xAuditError("use_ind must contain indices or booleans") from exc
        if len(set(selected)) != len(selected):
            raise Transition1xAuditError("use_ind contains duplicate indices")
        if any(index < 0 or index >= record_count for index in selected):
            raise Transition1xAuditError("use_ind contains an out-of-range index")
        representation = "index_list"
    return {
        "representation": representation,
        "selected_count": len(selected),
        "complement_count": record_count - len(selected),
        "selected_indices_sha256": hashlib.sha256(
            ",".join(str(index) for index in selected).encode("ascii")
        ).hexdigest(),
    }


def _branch_sequences(
    dataset: Mapping[str, Any],
) -> tuple[
    int,
    dict[str, Mapping[str, Any]],
    dict[str, Sequence[Any]],
    dict[str, Sequence[Any]],
]:
    branches: dict[str, Mapping[str, Any]] = {}
    positions: dict[str, Sequence[Any]] = {}
    atomic_numbers: dict[str, Sequence[Any]] = {}
    record_count: int | None = None
    for branch_name in _REQUIRED_BRANCHES:
        branch = _require_mapping(dataset.get(branch_name), branch_name)
        missing = [field for field in _REQUIRED_BRANCH_FIELDS if field not in branch]
        if missing:
            raise Transition1xAuditError(
                f"{branch_name} is missing required fields: {missing}"
            )
        branch_positions = _record_sequence(
            branch["positions"], f"{branch_name}.positions"
        )
        branch_atomic_numbers = _record_sequence(
            branch["charges"], f"{branch_name}.charges"
        )
        if len(branch_positions) != len(branch_atomic_numbers):
            raise Transition1xAuditError(
                f"{branch_name} positions/charges record counts differ"
            )
        if record_count is None:
            record_count = len(branch_positions)
        elif len(branch_positions) != record_count:
            raise Transition1xAuditError("endpoint branches have different record counts")
        branches[branch_name] = branch
        positions[branch_name] = branch_positions
        atomic_numbers[branch_name] = branch_atomic_numbers
    if record_count is None or record_count < 1:
        raise Transition1xAuditError("Transition1x asset contains no endpoint records")
    return record_count, branches, positions, atomic_numbers


def _audit_shared_alignment(
    record_count: int,
    branches: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for field in _SHARED_ALIGNMENT_FIELDS:
        presence = {name: field in branch for name, branch in branches.items()}
        if any(presence.values()) and not all(presence.values()):
            raise Transition1xAuditError(
                f"{field} is present in only some endpoint branches"
            )
        if not any(presence.values()):
            report[field] = {"present": False, "aligned": None}
            continue
        sequences = {
            name: _record_sequence(branch[field], f"{name}.{field}")
            for name, branch in branches.items()
        }
        if any(len(sequence) != record_count for sequence in sequences.values()):
            raise Transition1xAuditError(f"{field} record count differs from positions")
        reference = sequences["reactant"]
        for branch_name in ("product", "transition_state"):
            for index in range(record_count):
                if not np.array_equal(
                    np.asarray(reference[index]),
                    np.asarray(sequences[branch_name][index]),
                ):
                    raise Transition1xAuditError(
                        f"record {index} {field} differs between reactant and {branch_name}"
                    )
        report[field] = {"present": True, "aligned": True}
    return report


def _audit_endpoint_rows(
    record_count: int,
    positions: Mapping[str, Sequence[Any]],
    atomic_numbers: Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    atom_counts: list[int] = []
    element_counts: Counter[int] = Counter()
    for index in range(record_count):
        row_positions: dict[str, np.ndarray] = {}
        row_numbers: dict[str, np.ndarray] = {}
        for branch_name in _REQUIRED_BRANCHES:
            coordinates = np.asarray(positions[branch_name][index], dtype=np.float64)
            numbers = np.asarray(atomic_numbers[branch_name][index])
            if coordinates.ndim != 2 or coordinates.shape[1] != 3:
                raise Transition1xAuditError(
                    f"record {index} {branch_name} positions are not N x 3"
                )
            if numbers.ndim != 1 or numbers.shape[0] != coordinates.shape[0]:
                raise Transition1xAuditError(
                    f"record {index} {branch_name} atomic numbers do not match positions"
                )
            if not np.isfinite(coordinates).all():
                raise Transition1xAuditError(
                    f"record {index} {branch_name} positions are not finite"
                )
            if numbers.dtype.kind not in "iu" or np.any(numbers < 1):
                raise Transition1xAuditError(
                    f"record {index} {branch_name} charges are not atomic numbers"
                )
            row_positions[branch_name] = coordinates
            row_numbers[branch_name] = numbers.astype(np.int64, copy=False)
        reference_numbers = row_numbers["reactant"]
        reference_shape = row_positions["reactant"].shape
        for branch_name in ("product", "transition_state"):
            if row_positions[branch_name].shape != reference_shape:
                raise Transition1xAuditError(
                    f"record {index} coordinate shapes differ between reactant and {branch_name}"
                )
            if not np.array_equal(reference_numbers, row_numbers[branch_name]):
                raise Transition1xAuditError(
                    f"record {index} atom rows differ between reactant and {branch_name}"
                )
        atom_counts.append(int(reference_numbers.shape[0]))
        element_counts.update(int(value) for value in reference_numbers.tolist())
    return {
        "records_checked": record_count,
        "atom_count_min": min(atom_counts),
        "atom_count_max": max(atom_counts),
        "atomic_number_counts": {
            str(number): count for number, count in sorted(element_counts.items())
        },
        "endpoint_atom_rows_match": True,
        "endpoint_coordinate_shapes_match": True,
        "all_coordinates_finite": True,
    }


def audit_transition1x_pickle(
    path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int | None,
    source_revision: str,
    source_locator: str,
    license_record: str,
    upstream_code_commit: str,
    expected_record_count: int | None = None,
    expected_md5: str | None = None,
    license_status: str = "unknown",
) -> dict[str, Any]:
    """Audit one exact public pickle after checking its byte identity.

    Deserialization is permitted only after exact size/SHA-256 checks and uses a
    NumPy-only restricted unpickler.  The result remains source evidence rather
    than an admitted Track-B corpus.
    """

    if not path.is_file():
        raise Transition1xAuditError("Transition1x asset does not exist")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise Transition1xAuditError("expected_sha256 must be a lowercase SHA-256")
    if expected_md5 is not None and (
        len(expected_md5) != _MD5_LENGTH
        or any(character not in "0123456789abcdef" for character in expected_md5)
    ):
        raise Transition1xAuditError("expected_md5 must be a lowercase MD5")
    if expected_record_count is not None and expected_record_count < 1:
        raise Transition1xAuditError("expected_record_count must be positive")

    observed_size = path.stat().st_size
    if expected_size_bytes is not None and observed_size != expected_size_bytes:
        raise Transition1xAuditError(
            f"asset size mismatch: expected {expected_size_bytes}, observed {observed_size}"
        )
    observed_sha256 = hash_file(path)
    if observed_sha256 != expected_sha256:
        raise Transition1xAuditError("asset SHA-256 mismatch; refusing to unpickle")
    observed_md5 = hash_file(path, "md5") if expected_md5 is not None else None
    if expected_md5 is not None and observed_md5 != expected_md5:
        raise Transition1xAuditError("asset MD5 mismatch; refusing to unpickle")

    dataset = _require_mapping(_load_numeric_pickle(path), "Transition1x payload")
    record_count, branches, positions, atomic_numbers = _branch_sequences(dataset)
    if expected_record_count is not None and record_count != expected_record_count:
        raise Transition1xAuditError(
            f"record count mismatch: expected {expected_record_count}, observed {record_count}"
        )
    row_audit = _audit_endpoint_rows(record_count, positions, atomic_numbers)
    alignment_audit = _audit_shared_alignment(record_count, branches)
    if "use_ind" not in dataset:
        raise Transition1xAuditError("Transition1x payload is missing use_ind")
    split_audit = _normalise_split(dataset["use_ind"], record_count)

    branch_fields = {
        name: sorted(str(key) for key in branch.keys())
        for name, branch in branches.items()
    }
    common_branch_fields = {
        str(key).lower()
        for key in set.intersection(*(set(branch.keys()) for branch in branches.values()))
    }
    top_level_fields_lower = {str(key).lower() for key in dataset.keys()}
    evidence_fields = common_branch_fields | top_level_fields_lower
    event_fields = sorted(
        str(key)
        for key in dataset.keys()
        if str(key).lower() in {"event", "event_label", "bond_edits", "delta_be"}
    )
    family_fields = sorted(
        str(key)
        for key in dataset.keys()
        if str(key).lower() in {"family", "family_id", "reaction_family"}
    )
    license_is_explicit = license_status not in {"unknown", "unspecified", "missing"}

    return {
        "schema_version": TRANSITION1X_AUDIT_SCHEMA,
        "source": {
            "source_id": "transition1x_preprocessed",
            "source_locator": source_locator,
            "source_revision": source_revision,
            "license_record": license_record,
            "license_status": license_status,
            "upstream_code_commit": upstream_code_commit,
            "asset_name": path.name,
            "size_bytes": observed_size,
            "sha256": observed_sha256,
            "md5": observed_md5,
            "identity_verified_before_deserialization": True,
            "deserializer": "restricted_numpy_pickle",
            "archive_code_executed": False,
        },
        "structure": {
            "record_count": record_count,
            "top_level_fields": sorted(str(key) for key in dataset.keys()),
            "branch_fields": branch_fields,
            "shared_alignment": alignment_audit,
            "split": split_audit,
            **row_audit,
        },
        "field_evidence": {
            "positions_unit": "angstrom",
            "positions_unit_source": (
                "pinned upstream code at upstream_code_commit; not embedded "
                "as a per-record unit field"
            ),
            "charges_field_interpretation": "atomic_numbers",
            "formal_charge_field_present": bool(
                evidence_fields & {"formal_charge", "total_charge", "molecular_charge"}
            ),
            "multiplicity_field_present": "multiplicity" in evidence_fields,
            "event_label_fields": event_fields,
            "family_fields": family_fields,
            "explicit_license_present": license_is_explicit,
            "atom_correspondence": (
                "same row order and atomic numbers across R/P/TS; not an independently "
                "certified chemical atom mapping"
            ),
        },
        "track_b_gate": {
            "source_identity_verified": True,
            "endpoint_geometry_diagnostic_ready": True,
            "event_geometry_training_ready": False,
            "admission": "quarantine_diagnostic_only",
            "blocking_reasons": [
                "explicit formal charge is absent",
                "explicit multiplicity is absent",
                "trusted Track-B event labels are absent",
                "family-certified holdout metadata is absent",
                "same-row endpoint correspondence is not a certified atom mapping",
                "the pinned source record has no explicit redistribution license",
            ],
        },
        "scientific_claim_allowed": False,
        "claim_boundary": (
            "Source-identity and endpoint-structure evidence only. This report does "
            "not admit Track-B records, validate transition states, or support a "
            "joint-versus-serial model claim."
        ),
    }
