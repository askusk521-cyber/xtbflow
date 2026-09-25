"""Explicit, closed-shell Lewis bookkeeping for proposed elementary events.

BE off-diagonals are integer bond orders; diagonals count nonbonding electrons,
not lone pairs. Conservation is necessary, not proof of a feasible reaction or
an allowed valence state. Fractional TS bond orders must never be rounded into
this endpoint/intent representation. Atom indices are identities, not features.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


VALENCE_ELECTRONS = {"H": 1, "C": 4, "N": 5, "O": 6, "S": 6}
ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16}


class EventOrigin(str, Enum):
    """Audit tag only: callers must verify that claimed inputs are available."""

    INPUT_HYPOTHESIS = "input_hypothesis"
    ARCHIVED_SOURCE = "archived_source"
    TARGET_ORACLE = "target_oracle"


@dataclass(frozen=True)
class LewisState:
    """An explicit-H endpoint state with fixed composition and total charge.

    V0 supports CHNOS singlets with paired nonbonding electrons. We deliberately
    do not impose an octet on sulfur or infer missing hydrogens/charge. Domain
    valence checks and stereochemical identity must be supplied by the dataset.
    """

    symbols: tuple[str, ...]
    be: tuple[tuple[int, ...], ...]
    charge: int
    multiplicity: int

    def validate(self) -> None:
        """Reject unsupported scope and broken bookkeeping; do not certify chemistry."""
        n = len(self.symbols)
        if not n or any(s not in VALENCE_ELECTRONS for s in self.symbols):
            raise ValueError("Explicit supported CHNOS atoms are required")
        if type(self.charge) is not int or type(self.multiplicity) is not int or self.multiplicity != 1:
            raise ValueError("V0 requires explicit integer charge and singlet multiplicity")
        if len(self.be) != n or any(len(row) != n for row in self.be):
            raise ValueError("BE dimensions must match the atom inventory")
        if any(type(v) is not int or v < 0 for row in self.be for v in row):
            raise ValueError("BE entries must be nonnegative integers")
        if any(self.be[i][j] != self.be[j][i] for i in range(n) for j in range(n)):
            raise ValueError("BE must be symmetric")
        if any(self.be[i][i] % 2 for i in range(n)):
            raise ValueError("Unpaired nonbonding electrons are outside V0 scope")
        if sum(map(sum, self.be)) != sum(VALENCE_ELECTRONS[s] for s in self.symbols) - self.charge:
            raise ValueError("BE electron count is inconsistent with total charge")
        for i, s in enumerate(self.symbols):
            if s == "H" and self.be[i][i] + 2 * (sum(self.be[i]) - self.be[i][i]) > 2:
                raise ValueError("Endpoint hydrogen cannot have multiple covalent bonds")

    @property
    def formal_charges(self) -> tuple[int, ...]:
        """Row sums allocate one electron per bond order to each atom."""
        self.validate()
        return tuple(VALENCE_ELECTRONS[s] - sum(self.be[i]) for i, s in enumerate(self.symbols))


@dataclass(frozen=True)
class BondEdit:
    """One undirected bond-order change; diagonal changes use ElectronEdit."""

    i: int
    j: int
    delta: int


@dataclass(frozen=True)
class ElectronEdit:
    """Change in nonbonding electron count, measured in electrons."""

    atom: int
    delta: int


@dataclass(frozen=True)
class ProtonTransfer:
    """Move an explicitly identified H and heterolytically redistribute its bond pair.

    A relay contains multiple such transfers of different H atoms. This list
    specifies endpoint changes, not their physical timing or concertedness.
    """

    donor: int
    hydrogen: int
    acceptor: int


@dataclass(frozen=True)
class EventProposal:
    """A supplied candidate hypothesis; no automatic event enumeration is implied.

    Heavy-atom edits and explicit proton transfers are combined atomically so
    temporary intermediate matrices need not satisfy endpoint constraints.
    ``provenance`` identifies the rule or archive record used to propose it.
    """

    bond_edits: tuple[BondEdit, ...] = ()
    electron_edits: tuple[ElectronEdit, ...] = ()
    transfers: tuple[ProtonTransfer, ...] = ()
    origin: EventOrigin = EventOrigin.INPUT_HYPOTHESIS
    provenance: str = ""

    def apply(self, state: LewisState, *, allow_oracle: bool = False) -> LewisState:
        """Apply one complete event with exact electron/atom/charge conservation."""
        state.validate()
        if not isinstance(self.origin, EventOrigin) or not self.provenance.strip():
            raise ValueError("An explicit origin and nonempty provenance are required")
        if self.origin == EventOrigin.TARGET_ORACLE and not allow_oracle:
            raise ValueError("Target-derived event requires an explicit oracle diagnostic")
        n = len(state.symbols)
        matrix = [list(row) for row in state.be]

        def index(i: int) -> None:
            if type(i) is not int or not 0 <= i < n:
                raise ValueError("Event atom index is outside the fixed inventory")

        def change(i: int, j: int, value: int) -> None:
            index(i)
            index(j)
            if type(value) is not int or value == 0:
                raise ValueError("Each edit must have a nonzero integer delta")
            matrix[i][j] += value
            if i != j:
                matrix[j][i] += value

        used_h = set()
        for transfer in self.transfers:
            d, h, a = transfer.donor, transfer.hydrogen, transfer.acceptor
            for i in (d, h, a):
                index(i)
            if len({d, h, a}) != 3 or h in used_h:
                raise ValueError("Each transferred hydrogen requires one distinct donor and acceptor")
            if state.symbols[h] != "H" or state.symbols[d] == "H" or state.symbols[a] == "H":
                raise ValueError("Proton transfer must identify explicit H and heavy partners")
            if state.be[d][h] != 1 or state.be[a][h] != 0:
                raise ValueError("The source endpoint must contain donor-H but not acceptor-H")
            used_h.add(h)
            change(d, h, -1)
            change(a, h, 1)
            change(d, d, 2)
            change(a, a, -2)
        seen_bonds = set()
        for edit in self.bond_edits:
            index(edit.i)
            index(edit.j)
            key = tuple(sorted((edit.i, edit.j)))
            if edit.i == edit.j or key in seen_bonds or used_h.intersection(key):
                raise ValueError("Duplicate/diagonal bond edits or double-counted transferred H")
            seen_bonds.add(key)
            change(edit.i, edit.j, edit.delta)
        seen_electrons = set()
        for edit in self.electron_edits:
            if edit.atom in seen_electrons:
                raise ValueError("Combine repeated electron edits before proposing an event")
            seen_electrons.add(edit.atom)
            change(edit.atom, edit.atom, edit.delta)
        target = LewisState(state.symbols, tuple(map(tuple, matrix)), state.charge, state.multiplicity)
        target.validate()
        return target
