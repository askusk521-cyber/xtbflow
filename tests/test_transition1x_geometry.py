from __future__ import annotations

import numpy as np
import pytest

from xtbflow.evaluation.transition1x_geometry import (
    Transition1xGeometryError,
    evaluate_transition1x_guesses,
    geometry_metrics,
)


def _rotated(value: np.ndarray) -> np.ndarray:
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return value @ rotation


def _payload():
    reference = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    guesses = {
        "ts_guess": [reference + 0.2, reference + 0.2],
        "ts_guess_true": [reference.copy(), reference.copy()],
        "ts_guess_sbv1": [_rotated(reference), _rotated(reference)],
        "ts_guess_NEBCI-xtb": [reference + 0.1, reference + 0.1],
    }
    branch = {
        "rxn": ["rxn-0", "rxn-1"],
        "positions": [reference, reference],
        "wB97x_6-31G(d).energy": [1.0, 1.0],
    }
    payload = {
        "reactant": {**branch, "wB97x_6-31G(d).energy": [0.0, 0.0]},
        "transition_state": branch,
        "product": branch,
        "use_ind": [0],
        **guesses,
    }
    return payload


def test_geometry_metrics_aligns_translation_and_rotation():
    reference = _payload()["transition_state"]["positions"][0]
    metrics = geometry_metrics(_rotated(reference) + 8.0, reference)
    assert metrics["atom_mapped_rmsd_angstrom"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["pair_distance_mae_angstrom"] == pytest.approx(0.0, abs=1e-12)


def test_evaluator_preserves_oracle_scope_and_split_metrics():
    report = evaluate_transition1x_guesses(_payload())
    assert report["split"]["published_use_ind"]["n"] == 1
    assert report["metrics"]["ts_guess_true"]["field_scope"].startswith("near-reference")
    assert report["reference_energy_diagnostics"]["reactant_to_transition_state"]["complement_diagnostic"]["mean"] == pytest.approx(1.0)


def test_evaluator_rejects_missing_guess_field():
    payload = _payload()
    del payload["ts_guess"]
    with pytest.raises(Transition1xGeometryError, match="missing"):
        evaluate_transition1x_guesses(payload)
