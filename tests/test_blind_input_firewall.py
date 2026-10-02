from __future__ import annotations

import pytest

from xtbflow.evaluation.blind import (
    InputFirewallError,
    assert_same_blind_input,
    make_blind_input,
)


def test_blind_input_is_stable_under_target_metadata_changes():
    left = make_blind_input({"reactant_coordinates": [[0.0, 0.0, 0.0]], "charge": 0})
    right = make_blind_input({"reactant_coordinates": [[0.0, 0.0, 0.0]], "charge": 0})
    assert left.fingerprint == right.fingerprint
    assert_same_blind_input(left, right)


def test_nested_target_fields_are_rejected():
    with pytest.raises(InputFirewallError, match="target-derived"):
        make_blind_input({"reactant": {"coordinates": [], "reference_ts": []}})


def test_different_inputs_fail_comparison():
    left = make_blind_input({"reactant_coordinates": [[0.0, 0.0, 0.0]]})
    right = make_blind_input({"reactant_coordinates": [[0.1, 0.0, 0.0]]})
    with pytest.raises(InputFirewallError, match="differ"):
        assert_same_blind_input(left, right)
