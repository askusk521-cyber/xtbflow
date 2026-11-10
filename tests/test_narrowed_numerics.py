import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('diagnostic', Path(__file__).parents[1]/'scripts/diagnose_narrowed_numerics.py')
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_saddle_success_and_deliberate_wrong_mode_and_transport_counterexamples():
    good, wrong_mode, wrong_map = diagnostic.analytic_fixtures()
    assert good['after_saddle_distance'] < good['before_saddle_distance']
    assert wrong_mode['after_saddle_distance'] > wrong_mode['before_saddle_distance']
    assert wrong_map['first_unstable_coordinate_after'] > wrong_map['first_unstable_coordinate_before']
    assert all(r['alpha_zero_equivalent'] and r['normalized_unit_rescaling_invariant'] for r in (good, wrong_mode, wrong_map))


def test_retained_pairs_reject_missing_and_duplicate_arms():
    row = dict(parent_id='p', seed=1, arm='no_op')
    with pytest.raises(ValueError, match='denominator'):
        diagnostic.retained_steps([row])
    with pytest.raises(ValueError, match='duplicate'):
        diagnostic.retained_steps([row, row])
