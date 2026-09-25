"""Invariants and budget behavior of source-only relay enumeration."""
from dataclasses import replace
import unittest

from mechai.data.events import BondEdit, ElectronEdit, EventProposal, LewisState
from mechai.data.proposals import enumerate_relays


def hydration_fixture(water_count=2):
    n = 4 + 3 * water_count
    matrix = [[0] * n for _ in range(n)]
    matrix[1][1] = 4
    for i, j, value in [(0, 1, 2), (0, 2, 1), (0, 3, 1)]:
        matrix[i][j] = matrix[j][i] = value
    for o in range(4, n, 3):
        matrix[o][o] = 4
        for h in (o + 1, o + 2):
            matrix[o][h] = matrix[h][o] = 1
    state = LewisState(('C', 'O', 'H', 'H') + ('O', 'H', 'H') * water_count,
                       tuple(map(tuple, matrix)), 0, 1)
    core = EventProposal((BondEdit(0, 1, -1), BondEdit(0, 4, 1)),
                         (ElectronEdit(1, 2), ElectronEdit(4, -2)), provenance='synthetic-core')
    xyz = [(0.19 * i * i, 0.31 * i, (-1) ** i * .47) for i in range(n)]
    return state, core, xyz


class ProposalTests(unittest.TestCase):
    def test_all_direct_and_one_extra_water_choices_preserve_atoms(self):
        state, core, xyz = hydration_fixture(3)
        result = enumerate_relays(state, core, xyz, donors=(4,), acceptors=(1,), water_oxygens=(4, 7, 10))
        self.assertEqual(result.n_valid_before_cap, 11)  # core + 2 direct + 2*2*2 relay variants
        self.assertFalse(result.truncated)
        for item in result.candidates:
            self.assertEqual(item.event.apply(state).symbols, state.symbols)
            self.assertLessEqual(len(item.participating_water_oxygens), 2)
            self.assertIn(4, item.participating_water_oxygens)

    def test_water_budget_counts_attacking_and_intermediate_water(self):
        state, core, xyz = hydration_fixture()
        result = enumerate_relays(state, core, xyz, donors=(4,), acceptors=(1,), water_oxygens=(4, 7), max_participating_waters=1)
        self.assertEqual(result.n_valid_before_cap, 3)
        self.assertTrue(all(len(p.event.transfers) <= 1 for p in result.candidates))
        self.assertEqual(len(enumerate_relays(state, core, xyz, donors=(4,), acceptors=(1,),
                                            water_oxygens=(4, 7), max_participating_waters=0).candidates), 0)

    def test_budget_does_not_break_a_symmetric_tie_by_atom_id(self):
        state, core, xyz = hydration_fixture()
        xyz = [(0., 0., 0.)] * len(xyz)
        result = enumerate_relays(state, core, xyz, donors=(4,), acceptors=(1,), water_oxygens=(4, 7), max_candidates=3)
        self.assertEqual(result.n_valid_before_cap, 7)
        self.assertEqual(len(result.candidates), 0)
        self.assertEqual(result.n_boundary_ties_withheld, 7)
        self.assertTrue(result.truncated)

    def test_four_water_paths_are_complete_conserved_and_bounded(self):
        state, core, xyz = hydration_fixture(4)
        result = enumerate_relays(state, core, xyz, donors=(4,), acceptors=(1,),
                                  water_oxygens=(4, 7, 10, 13), max_participating_waters=4,
                                  max_candidates=200)
        # 1 core + 2 H choices * sum(P(3, k) * 2**k, k=0..3).
        self.assertEqual(result.n_valid_before_cap, 159)
        self.assertEqual(result.n_enumerated, result.enumeration_attempt_bound)
        self.assertFalse(result.truncated)
        self.assertEqual({len(p.event.transfers) for p in result.candidates}, {0, 1, 2, 3, 4})
        for item in result.candidates:
            endpoint = item.event.apply(state)
            self.assertEqual(sum(map(sum, endpoint.be)), sum(map(sum, state.be)))
            self.assertLessEqual(len(item.participating_water_oxygens), 4)

    def test_oversized_search_refuses_before_compiling_partial_proposals(self):
        from unittest.mock import patch
        state, core, xyz = hydration_fixture(4)
        with patch.object(EventProposal, 'apply', autospec=True, return_value=state) as compile_event:
            with self.assertRaisesRegex(ValueError, 'enumeration attempt budget'):
                enumerate_relays(state, core, xyz, donors=(4,), acceptors=(1,),
                                 water_oxygens=(4, 7, 10, 13), max_participating_waters=4,
                                 max_enumeration_attempts=100)
            self.assertEqual(compile_event.call_count, 1)  # Only validates the input core.

    def test_four_water_cap_is_invariant_under_rigid_motion_and_water_order(self):
        state, core, xyz = hydration_fixture(4)
        kwargs = dict(donors=(4,), acceptors=(1,), max_participating_waters=4, max_candidates=32)
        first = enumerate_relays(state, core, xyz, water_oxygens=(4, 7, 10, 13), **kwargs)
        moved = [(z + 2., x - 3., y + 5.) for x, y, z in xyz]
        second = enumerate_relays(state, core, moved, water_oxygens=(13, 10, 7, 4), **kwargs)
        self.assertEqual({p.event.apply(state).be for p in first.candidates},
                         {p.event.apply(state).be for p in second.candidates})
        self.assertEqual(first.n_boundary_ties_withheld, second.n_boundary_ties_withheld)

    def test_donor_without_h_does_not_trigger_factorial_path_iteration(self):
        from unittest.mock import patch
        state, core, xyz = hydration_fixture(4)
        with patch('mechai.data.proposals.permutations', side_effect=AssertionError('Must skip H-free donors')):
            result = enumerate_relays(state, core, xyz, donors=(1,), acceptors=(4,),
                                      water_oxygens=(4, 7, 10, 13), max_participating_waters=4)
        self.assertEqual(result.n_enumerated, 1)

    def test_permutation_preserves_capped_candidate_set(self):
        state, core, xyz = hydration_fixture(3)
        kwargs = dict(donors=(4,), acceptors=(1,), water_oxygens=(4, 7, 10), max_candidates=5,
                      max_participating_waters=4)
        before = enumerate_relays(state, core, xyz, **kwargs)
        order = [0, 1, 3, 2, 10, 12, 11, 7, 9, 8, 4, 6, 5]
        inv = {old: new for new, old in enumerate(order)}
        changed_state = replace(state, symbols=tuple(state.symbols[i] for i in order),
                                be=tuple(tuple(state.be[i][j] for j in order) for i in order))
        changed_core = replace(core, bond_edits=tuple(BondEdit(inv[e.i], inv[e.j], e.delta) for e in core.bond_edits),
                               electron_edits=tuple(ElectronEdit(inv[e.atom], e.delta) for e in core.electron_edits))
        after = enumerate_relays(changed_state, changed_core, [xyz[i] for i in order], donors=(inv[4],),
                                 acceptors=(inv[1],), water_oxygens=tuple(inv[i] for i in (4, 7, 10)),
                                 max_candidates=5, max_participating_waters=4)
        expected = {tuple(tuple(item.event.apply(state).be[i][j] for j in order) for i in order) for item in before.candidates}
        self.assertEqual(expected, {item.event.apply(changed_state).be for item in after.candidates})

    def test_bad_water_roles_and_target_oracle_rejected(self):
        from mechai.data.events import EventOrigin
        state, core, xyz = hydration_fixture()
        for proposal, water in [(core, (1, 4)), (replace(core, origin=EventOrigin.TARGET_ORACLE), (4, 7))]:
            with self.assertRaises(ValueError):
                enumerate_relays(state, proposal, xyz, donors=(4,), acceptors=(1,), water_oxygens=water)


if __name__ == '__main__':
    unittest.main()
