from __future__ import annotations

from dataclasses import replace

import pytest

from mechai.data.events import LewisState
from xtbflow.proposals import (
    CoreIntent,
    CoreRuleConfig,
    RoleSet,
    build_proposal_pool,
    enumerate_core_events,
    infer_roles,
)


def state(symbols, be):
    return LewisState(tuple(symbols), tuple(tuple(row) for row in be), 0, 1)


def test_addition_bookkeeping_is_conserved():
    source = state(("C", "O"), [[4, 0], [0, 6]])
    roles = RoleSet(nucleophiles=(1,), electrophiles=(0,))
    result = enumerate_core_events(source, roles, config=CoreRuleConfig(intents=(CoreIntent.ADDITION,)))
    assert len(result.candidates) == 1
    target = result.candidates[0].event.apply(source)
    assert target.be == ((4, 1), (1, 4))
    assert result.candidates[0].event.origin.value == "input_hypothesis"


def test_substitution_break_and_make_is_one_conserved_event():
    source = state(("C", "O", "N", "H", "H", "H", "H", "H"), [
        [2, 1, 0, 1, 0, 0, 0, 0],
        [1, 4, 0, 0, 1, 0, 0, 0],
        [0, 0, 2, 0, 0, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
    ])
    roles = RoleSet(nucleophiles=(2,), electrophiles=(0,), leaving_groups=(1,))
    result = enumerate_core_events(source, roles, config=CoreRuleConfig(intents=(CoreIntent.SUBSTITUTION,)))
    assert len(result.candidates) == 1
    target = result.candidates[0].event.apply(source)
    assert target.be[0][1] == 0
    assert target.be[0][2] == 1


def test_bond_break_preserves_electron_count_and_rejects_invalid_scope():
    source = state(("C", "O"), [[2, 1], [1, 6]])
    result = enumerate_core_events(source, RoleSet(), config=CoreRuleConfig(intents=(CoreIntent.BOND_BREAK,)))
    assert len(result.candidates) == 2
    assert {candidate.event.apply(source).be for candidate in result.candidates} == {
        ((4, 0), (0, 6)),
        ((2, 0), (0, 8)),
    }


def test_inferred_roles_use_only_reactant_state():
    source = state(("O", "H", "H", "C"), [[4, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 4]])
    roles = infer_roles(source)
    assert roles.water_oxygens == (0,)
    assert 0 in roles.nucleophiles
    assert roles.provenance == "reactant-rule-v1"


def test_hydronium_is_not_misclassified_as_neutral_water():
    matrix = [
        [2, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
    ]
    source = LewisState(("O", "H", "H", "H"), tuple(tuple(row) for row in matrix), 1, 1)
    assert infer_roles(source).water_oxygens == ()


def test_equal_score_group_is_not_atom_order_truncated():
    source = state(("C", "O", "O"), [[4, 0, 0], [0, 6, 0], [0, 0, 6]])
    roles = RoleSet(nucleophiles=(1, 2), electrophiles=(0,))
    result = enumerate_core_events(source, roles, config=CoreRuleConfig(intents=(CoreIntent.ADDITION,), max_candidates=1))
    assert result.truncated is True
    assert result.n_boundary_ties_withheld == 2
    assert not result.candidates


def test_max_bond_order_rejects_out_of_scope_endpoint():
    source = state(("C", "O"), [[4, 2], [2, 2]])
    roles = RoleSet(nucleophiles=(1,), electrophiles=(0,))
    with pytest.raises(ValueError, match="maximum bond order"):
        enumerate_core_events(
            source,
            roles,
            config=CoreRuleConfig(intents=(CoreIntent.ADDITION,), max_bond_order=1),
        )


def test_hydrogen_cannot_be_used_as_a_heavy_atom_role():
    source = state(("C", "H", "H", "H", "H"), [
        [0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
    ])
    with pytest.raises(ValueError, match="heavy-atom core roles"):
        enumerate_core_events(source, RoleSet(nucleophiles=(1,), electrophiles=(0,)))


def test_config_accepts_serialized_intent_names():
    config = CoreRuleConfig(intents=("bond_break",))
    assert config.intents == (CoreIntent.BOND_BREAK,)


def test_pool_adds_bounded_relay_candidates_without_target_inputs():
    n = 7
    matrix = [[0] * n for _ in range(n)]
    matrix[0][0] = 0
    matrix[0][1] = matrix[1][0] = 2
    matrix[0][2] = matrix[2][0] = 1
    matrix[0][3] = matrix[3][0] = 1
    matrix[1][1] = 4
    matrix[4][4] = 4
    matrix[4][5] = matrix[5][4] = 1
    matrix[4][6] = matrix[6][4] = 1
    source = state(("C", "O", "H", "H", "O", "H", "H"), matrix)
    roles = RoleSet(nucleophiles=(4,), electrophiles=(0,), donors=(4,), acceptors=(1,), water_oxygens=(4,))
    coordinates = [(float(i), 0.0, 0.0) for i in range(n)]
    first = build_proposal_pool(
        source,
        coordinates,
        roles=roles,
        core_config=CoreRuleConfig(intents=(CoreIntent.SUBSTITUTION, CoreIntent.BOND_BREAK)),
        max_participating_waters=1,
    )
    assert first.core.candidates
    assert first.relays
    assert all(item.event.origin.value == "input_hypothesis" for item in first.relays)
    assert first.relay_attempts > 0


def test_core_pool_is_invariant_to_unseen_product_or_ts_metadata():
    source = state(("C", "O"), [[4, 0], [0, 6]])
    roles = RoleSet(nucleophiles=(1,), electrophiles=(0,))
    baseline = enumerate_core_events(source, roles)
    # The proposal API has no product/TS parameters; attaching unrelated
    # metadata to a caller-side record cannot affect this reactant-only result.
    record = {"reactant": source, "product": object(), "ts": object()}
    assert enumerate_core_events(record["reactant"], roles) == baseline


def test_atom_permutation_preserves_core_endpoint_set():
    source = state(("C", "O", "O"), [[4, 0, 0], [0, 6, 0], [0, 0, 6]])
    roles = RoleSet(nucleophiles=(1, 2), electrophiles=(0,))
    before = enumerate_core_events(source, roles, config=CoreRuleConfig(intents=(CoreIntent.ADDITION,), max_candidates=8))
    order = (0, 2, 1)
    inverse = {old: new for new, old in enumerate(order)}
    permuted = replace(source, symbols=tuple(source.symbols[i] for i in order), be=tuple(tuple(source.be[i][j] for j in order) for i in order))
    after = enumerate_core_events(
        permuted,
        RoleSet(nucleophiles=tuple(sorted(inverse[i] for i in roles.nucleophiles)), electrophiles=(inverse[0],)),
        config=CoreRuleConfig(intents=(CoreIntent.ADDITION,), max_candidates=8),
    )
    expected = {
        tuple(tuple(endpoint[i][j] for j in order) for i in order)
        for item in before.candidates
        for endpoint in (item.event.apply(source).be,)
    }
    assert expected == {item.event.apply(permuted).be for item in after.candidates}
