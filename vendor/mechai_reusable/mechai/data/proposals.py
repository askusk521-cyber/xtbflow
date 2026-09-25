"""Bounded explicit-H relay proposals from a supplied reaction intent.

This enumerator does not discover heavy-atom reaction centers. It augments one
source-supplied core edit with zero/direct/water-mediated proton transfers. All
environment atoms are retained, all proposals are hypotheses, and the geometric
ordering is a heuristic rather than an energy or a feasibility label.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from math import dist, isfinite, perm
from typing import Sequence

from .events import EventProposal, LewisState, ProtonTransfer


@dataclass(frozen=True)
class RankedProposal:
    """One conservative bookkeeping proposal and its source-only geometric cost."""

    event: EventProposal
    heuristic_cost: float
    participating_water_oxygens: tuple[int, ...]


@dataclass(frozen=True)
class ProposalSet:
    """Capped proposal set with explicit coverage limitations.

    ``n_valid_before_cap`` counts unique BE outcomes from this rule family, not
    physical channels. At a budget boundary a whole tied group is withheld if
    it will not fit; atom IDs never decide which equivalent candidate survives.
    """

    candidates: tuple[RankedProposal, ...]
    n_enumerated: int
    n_invalid_bookkeeping: int
    n_valid_before_cap: int
    truncated: bool
    n_boundary_ties_withheld: int
    enumeration_attempt_bound: int


def enumerate_relays(
    state: LewisState,
    core: EventProposal,
    coordinates: Sequence[Sequence[float]],
    *,
    donors: Sequence[int],
    acceptors: Sequence[int],
    water_oxygens: Sequence[int],
    max_participating_waters: int = 2,
    max_candidates: int = 32,
    include_no_transfer: bool = True,
    max_enumeration_attempts: int = 100_000,
) -> ProposalSet:
    """Add relay endpoint changes using distinct H atoms from the source state.

    A donor/acceptor water counts toward the water limit, as do intermediate
    waters. A zero-transfer alternative can represent a charged intermediate;
    its existence/stability is not assumed. Cost is the sum of squared excess
    H--acceptor distances above 1.2 Angstrom. It only orders proposals; there is
    no distance rejection cutoff. Equal costs are quantized at 1e-10 A^2 for
    stable whole-tie budget handling under rigid transforms and atom relabeling.

    Water-chain length is variable up to the supplied bound. Before compiling
    any event, a conservative combinatorial bound must fit the search budget.
    An oversized search raises instead of returning an atom-order-biased prefix.
    The output candidate budget and this computation limit are separate limits.
    """
    state.validate()
    if core.transfers:
        raise ValueError("The supplied core must not already contain proton transfers")
    # Scope/provenance and the core's conservation are checked before enumeration.
    core.apply(state)
    n = len(state.symbols)
    if len(coordinates) != n or any(len(row) != 3 or not all(isfinite(v) for v in row) for row in coordinates):
        raise ValueError("Finite source coordinates in Angstrom are required for every atom")
    if type(max_participating_waters) is not int or max_participating_waters < 0:
        raise ValueError("The water-participation bound must be a nonnegative integer")
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError("The candidate budget must be a positive integer")
    if type(max_enumeration_attempts) is not int or max_enumeration_attempts < 1:
        raise ValueError("The enumeration attempt budget must be a positive integer")
    for indices in (donors, acceptors, water_oxygens):
        if len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < n for i in indices):
            raise ValueError("Role indices must be unique and inside the inventory")
    if any(state.symbols[i] == "H" for i in (*donors, *acceptors)):
        raise ValueError("Donors and acceptors must be heavy atoms")
    neighbors = {i: tuple(j for j in range(n) if state.be[i][j] and j != i) for i in range(n)}
    for o in water_oxygens:
        if (state.symbols[o] != "O" or state.be[o][o] != 4 or len(neighbors[o]) != 2
                or any(state.symbols[h] != "H" or state.be[o][h] != 1 for h in neighbors[o])):
            raise ValueError("Water role requires a neutral explicit OHH source molecule")
    water = set(water_oxygens)
    edited_atoms = {i for e in core.bond_edits for i in (e.i, e.j)} | {e.atom for e in core.electron_edits}
    core_waters = {o for o in water if edited_atoms.intersection((o, *neighbors[o]))}
    hydrogen_choices = {i: tuple(h for h in neighbors[i] if state.symbols[h] == "H" and state.be[i][h] == 1)
                        for i in range(n)}

    def path_limit(donor: int, acceptor: int, available: Sequence[int]) -> int:
        # Intermediate core waters may already count in the fixed set. Bounding
        # only by endpoint waters is conservative; the exact union is checked
        # below. Every intermediate is a distinct, validated neutral OHH.
        return min(len(available), max_participating_waters - len({donor, acceptor} & water))

    attempt_bound = int(include_no_transfer and len(core_waters) <= max_participating_waters)
    for donor in donors:
        for acceptor in acceptors:
            if (donor == acceptor or not hydrogen_choices[donor]
                    or len(core_waters | ({donor, acceptor} & water)) > max_participating_waters):
                continue
            available = [w for w in water_oxygens if w not in (donor, acceptor)]
            for length in range(path_limit(donor, acceptor, available) + 1):
                attempt_bound += perm(len(available), length) * len(hydrogen_choices[donor]) * 2 ** length
                if attempt_bound > max_enumeration_attempts:
                    raise ValueError("Relay search exceeds the enumeration attempt budget; no partial candidates returned")
    pool: dict[tuple, RankedProposal] = {}
    attempted = invalid = 0

    def consider(transfers: tuple[ProtonTransfer, ...]) -> None:
        nonlocal attempted, invalid
        attempted += 1
        event = EventProposal(core.bond_edits, core.electron_edits, transfers,
                              core.origin, core.provenance + ":bounded-relay-v1")
        try:
            target = event.apply(state)
        except ValueError:
            invalid += 1
            return
        participants = tuple(sorted(core_waters | water.intersection(
            {i for t in transfers for i in (t.donor, t.acceptor)})))
        cost = sum(max(dist(coordinates[t.hydrogen], coordinates[t.acceptor]) - 1.2, 0.) ** 2 for t in transfers)
        proposal = RankedProposal(event, cost, participants)
        # Equivalent descriptions of the same mapped endpoint edit get one slot.
        # Different actual trajectories with the same edit are a geometry task.
        if target.be not in pool or cost < pool[target.be].heuristic_cost:
            pool[target.be] = proposal

    if include_no_transfer and len(core_waters) <= max_participating_waters:
        consider(())
    for donor in donors:
        for acceptor in acceptors:
            if donor == acceptor or not hydrogen_choices[donor]:
                continue
            fixed_waters = len(core_waters | ({donor, acceptor} & water))
            if fixed_waters > max_participating_waters:
                continue
            available = [w for w in water_oxygens if w not in (donor, acceptor)]
            for length in range(path_limit(donor, acceptor, available) + 1):
                for middle in permutations(available, length):
                    if len(core_waters | ({donor, acceptor, *middle} & water)) > max_participating_waters:
                        continue
                    path = (donor, *middle, acceptor)
                    choices = [hydrogen_choices[d] for d in path[:-1]]
                    for hydrogens in product(*choices):
                        if len(set(hydrogens)) == len(hydrogens):
                            consider(tuple(ProtonTransfer(d, h, a) for d, h, a in zip(path[:-1], hydrogens, path[1:])))
    buckets: dict[float, list[RankedProposal]] = {}
    for proposal in pool.values():
        buckets.setdefault(round(proposal.heuristic_cost, 10), []).append(proposal)
    chosen = []
    withheld = 0
    for score in sorted(buckets):
        group = buckets[score]
        if len(chosen) + len(group) > max_candidates:
            withheld = len(group)
            break
        chosen.extend(group)
    return ProposalSet(tuple(chosen), attempted, invalid, len(pool), len(chosen) < len(pool), withheld, attempt_bound)
