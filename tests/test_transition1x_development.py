from __future__ import annotations

import numpy as np
import pytest

from xtbflow.data.transition1x_development import (
    Transition1xDevelopmentError,
    conserved_event_delta,
    development_sample,
    development_split,
    infer_binary_bonds,
)


def _payload():
    numbers = np.asarray([6, 1, 1, 1, 1], dtype=np.int64)
    reactant = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    product = reactant.copy()
    product[1] = [0.0, 0.0, 1.0]
    transition = (reactant + product) / 2
    branch = {
        "charges": [numbers],
        "positions": [reactant],
        "formula": ["CH4"],
        "rxn": ["rxn-0"],
    }
    return {
        "reactant": branch,
        "product": {**branch, "positions": [product]},
        "transition_state": {**branch, "positions": [transition]},
    }


def test_binary_bonds_are_symmetric_and_diagonal_free():
    payload = _payload()
    bonds = infer_binary_bonds(payload["reactant"]["charges"][0], payload["reactant"]["positions"][0])
    assert np.array_equal(bonds, bonds.T)
    assert np.count_nonzero(np.diag(bonds)) == 0
    assert bonds[0, 1] == 1.0


def test_event_delta_conserves_weighted_packed_sum():
    payload = _payload()
    event = conserved_event_delta(
        payload["reactant"]["charges"][0],
        payload["reactant"]["positions"][0],
        payload["product"]["positions"][0],
    )
    assert event["evidence"] == "derived_under_contract"
    assert event["conservation_residual"] == pytest.approx(0.0, abs=1e-12)
    assert event["delta_matrix"].shape == (5, 5)


def test_development_sample_marks_product_derived_split_and_claim_limit():
    sample = development_sample(_payload(), 0)
    assert sample["split"] == development_split("CH4")
    assert sample["claim_limit"].startswith("quarantine")
    assert sample["event"]["representation"].endswith("v1")


def test_development_rejects_unsupported_element():
    payload = _payload()
    payload["reactant"]["charges"][0][0] = 9
    with pytest.raises(Transition1xDevelopmentError, match="H/C/N/O"):
        development_sample(payload, 0)

