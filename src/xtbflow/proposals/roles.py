"""Reactant-only role hypotheses for the strong core-event baseline."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from mechai.data.events import LewisState


class AtomRole(str, Enum):
    NUCLEOPHILE = "nucleophile"
    ELECTROPHILE = "electrophile"
    LEAVING_GROUP = "leaving_group"
    DONOR = "proton_donor"
    ACCEPTOR = "proton_acceptor"
    WATER_OXYGEN = "neutral_water_oxygen"


@dataclass(frozen=True)
class RoleSet:
    """Explicit role candidates; roles are hypotheses, not chemical labels."""

    nucleophiles: tuple[int, ...] = ()
    electrophiles: tuple[int, ...] = ()
    leaving_groups: tuple[int, ...] = ()
    donors: tuple[int, ...] = ()
    acceptors: tuple[int, ...] = ()
    water_oxygens: tuple[int, ...] = ()
    provenance: str = "reactant-rule-v1"

    def __post_init__(self) -> None:
        all_values = (self.nucleophiles, self.electrophiles, self.leaving_groups, self.donors, self.acceptors, self.water_oxygens)
        for values in all_values:
            if tuple(sorted(set(values))) != values or any(type(i) is not int or i < 0 for i in values):
                raise ValueError("role indices must be sorted unique nonnegative integers")
        if not self.provenance.strip():
            raise ValueError("role provenance is required")


def infer_roles(state: LewisState) -> RoleSet:
    """Generate conservative role hypotheses from the current Lewis state only.

    The rule intentionally over-generates eligible CHNOS atoms and leaves
    physical feasibility to later validation. It never reads products, TSs or
    target-derived solvent identities.
    """

    state.validate()
    neighbors = {i: [j for j, order in enumerate(state.be[i]) if j != i and order > 0] for i in range(len(state.symbols))}
    nucleophiles = tuple(i for i, symbol in enumerate(state.symbols) if symbol in {"N", "O", "S"} and state.be[i][i] >= 2)
    electrophiles = tuple(i for i, symbol in enumerate(state.symbols) if symbol in {"C", "N", "S"} and state.be[i][i] >= 1)
    leaving_groups = tuple(i for i, symbol in enumerate(state.symbols) if symbol in {"N", "O", "S"} and neighbors[i])
    donors = tuple(i for i in nucleophiles if any(state.symbols[j] == "H" and state.be[i][j] == 1 for j in neighbors[i]))
    acceptors = nucleophiles
    water_oxygens = tuple(
        i for i, symbol in enumerate(state.symbols)
        if symbol == "O" and state.be[i][i] == 4 and len(neighbors[i]) == 2
        and all(state.symbols[j] == "H" and state.be[i][j] == 1 for j in neighbors[i])
    )
    return RoleSet(nucleophiles, electrophiles, leaving_groups, donors, acceptors, water_oxygens)
