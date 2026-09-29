"""Endpoint-conditioned Transition1x geometry diagnostics.

This evaluator preserves the public row order and uses a proper-rotation
Kabsch alignment. It does not infer bonds, events, charge, multiplicity or
mechanism, and it treats the published complement of ``use_ind`` as a
diagnostic split rather than an audited family holdout.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np


GUESS_FIELDS = ("ts_guess", "ts_guess_true", "ts_guess_sbv1", "ts_guess_NEBCI-xtb")


class Transition1xGeometryError(ValueError):
    """Raised when a Transition1x geometry evaluation input is malformed."""


def _positions(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise Transition1xGeometryError(f"{name} must be finite [N,3] coordinates")
    return array


def geometry_metrics(predicted: Any, reference: Any) -> dict[str, float]:
    """Return proper-rotation atom RMSD and pair-distance MAE."""

    predicted_array = _positions(predicted, "predicted")
    reference_array = _positions(reference, "reference")
    if predicted_array.shape != reference_array.shape or len(predicted_array) == 0:
        raise Transition1xGeometryError("prediction and reference must have the same non-empty shape")
    centered_predicted = predicted_array - predicted_array.mean(axis=0)
    centered_reference = reference_array - reference_array.mean(axis=0)
    left, _, right_transpose = np.linalg.svd(
        centered_predicted.T @ centered_reference,
        full_matrices=False,
    )
    parity = float(np.linalg.det(left @ right_transpose))
    rotation = left @ np.diag([1.0, 1.0, parity]) @ right_transpose
    aligned = centered_predicted @ rotation
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - centered_reference) ** 2, axis=1))))
    predicted_distances = np.linalg.norm(predicted_array[:, None, :] - predicted_array[None, :, :], axis=-1)
    reference_distances = np.linalg.norm(reference_array[:, None, :] - reference_array[None, :, :], axis=-1)
    upper = np.triu_indices(len(predicted_array), 1)
    pair_error = np.abs(predicted_distances[upper] - reference_distances[upper])
    return {
        "atom_mapped_rmsd_angstrom": rmsd,
        "pair_distance_mae_angstrom": float(np.mean(pair_error)) if len(pair_error) else 0.0,
    }


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise Transition1xGeometryError("metric values must be a non-empty finite sequence")
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def _split_summary(rows: Sequence[Mapping[str, float]], indices: Sequence[int]) -> dict[str, dict[str, float | int]]:
    return {
        metric: _summary(rows[index][metric] for index in indices)
        for metric in ("atom_mapped_rmsd_angstrom", "pair_distance_mae_angstrom")
    }


def _indices_sha256(indices: Sequence[int]) -> str:
    return hashlib.sha256(",".join(str(index) for index in indices).encode("ascii")).hexdigest()


def evaluate_transition1x_guesses(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate all published guesses against the audited reference TS rows."""

    required = {"reactant", "transition_state", "product", "use_ind", *GUESS_FIELDS}
    missing = sorted(required - set(payload))
    if missing:
        raise Transition1xGeometryError(f"missing Transition1x fields: {missing}")
    try:
        n_records = len(payload["transition_state"]["rxn"])
        fit_indices = sorted({int(index) for index in payload["use_ind"]})
    except (KeyError, TypeError, ValueError) as exc:
        raise Transition1xGeometryError("malformed Transition1x split fields") from exc
    if n_records < 1 or len(fit_indices) >= n_records or any(index < 0 or index >= n_records for index in fit_indices):
        raise Transition1xGeometryError("invalid Transition1x use_ind split")
    complement_indices = [index for index in range(n_records) if index not in set(fit_indices)]
    reference = payload["transition_state"]["positions"]
    metrics: dict[str, dict[str, Any]] = {}
    for field in GUESS_FIELDS:
        try:
            rows = [geometry_metrics(payload[field][index], reference[index]) for index in range(n_records)]
        except (KeyError, IndexError, TypeError, Transition1xGeometryError) as exc:
            raise Transition1xGeometryError(f"invalid geometry field {field}") from exc
        metrics[field] = {
            "split": {
                "published_use_ind": _split_summary(rows, fit_indices),
                "complement_diagnostic": _split_summary(rows, complement_indices),
            },
            "field_scope": (
                "near-reference diagnostic; do not use as independent input"
                if field == "ts_guess_true"
                else "published TS-initial-guess field"
            ),
        }

    def energy_summary(field: str, indices: Sequence[int]) -> dict[str, float | int]:
        try:
            values = np.asarray(
                payload["transition_state"][field], dtype=np.float64
            ) - np.asarray(payload["reactant"][field], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise Transition1xGeometryError("reference energy fields are malformed") from exc
        return _summary(values[list(indices)])

    return {
        "dataset": "Transition1x preprocessed Reaction Dataset",
        "n_records": n_records,
        "split": {
            "published_use_ind": {
                "n": len(fit_indices),
                "indices_sha256": _indices_sha256(fit_indices),
            },
            "complement_diagnostic": {
                "n": len(complement_indices),
                "indices_sha256": _indices_sha256(complement_indices),
            },
        },
        "alignment_contract": {
            "atom_row_order": "published row order",
            "atomic_numbers": "charges field, as audited; not formal charges",
            "rotation": "proper Kabsch alignment",
            "atom_permutations": False,
            "event_or_mechanism_inference": False,
        },
        "metrics": metrics,
        "reference_energy_diagnostics": {
            "reactant_to_transition_state": {
                "published_use_ind": energy_summary("wB97x_6-31G(d).energy", fit_indices),
                "complement_diagnostic": energy_summary("wB97x_6-31G(d).energy", complement_indices),
                "unit": "published energy unit",
            },
        },
        "claim_limits": [
            "The complement is a diagnostic split from the published index list, not an independently audited reaction-family split.",
            "ts_guess_true is near the reference TS and is retained only as an oracle-like diagnostic, not an independent model input.",
            "The data contain no explicit bond graph, formal charge, multiplicity, or product-free event label.",
            "Geometry error and endpoint energy gaps are not DFT/IRC success, reaction rates, or chemical flux.",
        ],
        "new_quantum_calculations": False,
    }

