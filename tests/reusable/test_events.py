"""Chemical bookkeeping fixtures; no synthetic example is a validated TS/path."""
from dataclasses import replace
import unittest

from mechai.data.events import BondEdit, ElectronEdit, EventOrigin, EventProposal, LewisState, ProtonTransfer


def waters(count=2):
    """Explicit neutral water Lewis matrices with OHH atom ordering."""
    matrix = [[0] * (3 * count) for _ in range(3 * count)]
    for o in range(0, 3 * count, 3):
        matrix[o][o] = 4
        for h in (o + 1, o + 2):
            matrix[o][h] = matrix[h][o] = 1
    return LewisState(("O", "H", "H") * count, tuple(map(tuple, matrix)), 0, 1)


class EventTests(unittest.TestCase):
    def test_proton_transfer_preserves_inventory_charge_electrons(self):
        state = waters()
        event = EventProposal(transfers=(ProtonTransfer(0, 1, 3),), provenance="synthetic-test")
        target = event.apply(state)
        self.assertEqual(target.symbols, state.symbols)
        self.assertEqual(sum(map(sum, target.be)), sum(map(sum, state.be)))
        self.assertEqual(target.formal_charges, (-1, 0, 0, 1, 0, 0))
        self.assertEqual(state.be[0][1], 1)

    def test_relay_preserves_middle_water_charge(self):
        state = waters(3)
        proposal = EventProposal(transfers=(ProtonTransfer(0, 1, 3), ProtonTransfer(3, 4, 6)),
                                 provenance="synthetic-relay")
        target = proposal.apply(state)
        self.assertEqual(target.formal_charges[::3], (-1, 0, 1))
        self.assertEqual(target.be[3][3], state.be[3][3])

    def test_heavy_atom_edits_and_electron_edits_apply_together(self):
        matrix = [[0] * 7 for _ in range(7)]
        matrix[1][1] = matrix[4][4] = 4
        for i, j, value in ((0, 1, 2), (0, 2, 1), (0, 3, 1), (4, 5, 1), (4, 6, 1)):
            matrix[i][j] = matrix[j][i] = value
        state = LewisState(("C", "O", "H", "H", "O", "H", "H"), tuple(map(tuple, matrix)), 0, 1)
        proposal = EventProposal(bond_edits=(BondEdit(0, 1, -1), BondEdit(0, 4, 1)),
                                 electron_edits=(ElectronEdit(1, 2), ElectronEdit(4, -2)),
                                 provenance="synthetic-addition")
        self.assertEqual(proposal.apply(state).formal_charges, (0, -1, 0, 0, 1, 0, 0))

    def test_nonconservation_and_incompatible_hydrogen_edits_fail(self):
        state = waters()
        for proposal in (
            EventProposal(electron_edits=(ElectronEdit(0, 2),), provenance="bad-count"),
            EventProposal(transfers=(ProtonTransfer(0, 4, 3),), provenance="wrong-h"),
            EventProposal(transfers=(ProtonTransfer(0, 1, 3),) * 2, provenance="duplicate-h"),
            EventProposal(transfers=(ProtonTransfer(0, 1, 3),), bond_edits=(BondEdit(0, 1, -1),), provenance="duplicate-edit"),
        ):
            with self.subTest(proposal=proposal), self.assertRaises(ValueError):
                proposal.apply(state)

    def test_explicit_scope_and_oracle_guard(self):
        for state in (replace(waters(), charge=None), replace(waters(), multiplicity=2)):
            with self.assertRaises(ValueError):
                state.validate()
        proposal = EventProposal(origin=EventOrigin.TARGET_ORACLE, provenance="target-derived-test")
        with self.assertRaises(ValueError):
            proposal.apply(waters())
        self.assertEqual(proposal.apply(waters(), allow_oracle=True), waters())

    def test_matrix_symmetry_and_integer_electrons(self):
        state = waters()
        for value in (0, 1.5):
            matrix = [list(row) for row in state.be]
            matrix[0][1] = value
            with self.assertRaises(ValueError):
                replace(state, be=tuple(map(tuple, matrix))).validate()


if __name__ == "__main__":
    unittest.main()
