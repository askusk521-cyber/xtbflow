"""Bounded reactant-only heavy-atom event rules.

The rules create bookkeeping hypotheses and call ``EventProposal.apply`` for
conservation validation. They never inspect products, reference TSs, energy,
forces or target-selected waters, and a surviving candidate is not a pathway
claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product

from mechai.data.events import BondEdit, ElectronEdit, EventOrigin, EventProposal, LewisState

from .roles import RoleSet


class CoreIntent(str, Enum):
    ADDITION = "addition"
    SUBSTITUTION = "substitution"
    BOND_BREAK = "bond_break"


@dataclass(frozen=True)
class CoreRuleConfig:
    intents: tuple[CoreIntent, ...] = (CoreIntent.ADDITION, CoreIntent.SUBSTITUTION, CoreIntent.BOND_BREAK)
    max_candidates: int = 64
    max_attempts: int = 10_000
    max_bond_order: int = 3

    def __post_init__(self) -> None:
        # Config files use readable strings while callers often use the enum;
        # normalize both forms before enumeration so provenance and ordering
        # never depend on the caller's representation.
        normalized = tuple(intent if isinstance(intent, CoreIntent) else CoreIntent(intent) for intent in self.intents)
        object.__setattr__(self, "intents", normalized)
        if not self.intents or len(set(self.intents)) != len(self.intents):
            raise ValueError("at least one unique core intent is required")
        if type(self.max_candidates) is not int or self.max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if type(self.max_bond_order) is not int or self.max_bond_order < 1:
            raise ValueError("max_bond_order must be positive")


@dataclass(frozen=True)
class CoreProposal:
    event: EventProposal
    intent: CoreIntent
    heuristic_score: int
    provenance: str


@dataclass(frozen=True)
class CoreProposalSet:
    candidates: tuple[CoreProposal, ...]
    n_attempted: int
    n_invalid: int
    n_valid_before_cap: int
    truncated: bool
    n_boundary_ties_withheld: int
    attempt_bound: int


def _bond_order(state: LewisState, i: int, j: int) -> int:
    return state.be[i][j]


def _within_bond_order(state: LewisState, max_bond_order: int) -> bool:
    """Check the configured discrete bond-order scope for an endpoint."""

    return all(
        i == j or state.be[i][j] <= max_bond_order
        for i in range(len(state.symbols))
        for j in range(len(state.symbols))
    )


def _candidate_key(state: LewisState, proposal: CoreProposal) -> tuple:
    edits = tuple(sorted((state.symbols[e.i], state.symbols[e.j], e.delta) for e in proposal.event.bond_edits))
    electrons = tuple(sorted((state.symbols[e.atom], e.delta) for e in proposal.event.electron_edits))
    return (proposal.heuristic_score, proposal.intent.value, edits, electrons)


def enumerate_core_events(state: LewisState, roles: RoleSet, *, config: CoreRuleConfig | None = None) -> CoreProposalSet:
    """Enumerate bounded addition, substitution and bond-breaking hypotheses."""

    state.validate()
    config = config or CoreRuleConfig()
    if not _within_bond_order(state, config.max_bond_order):
        raise ValueError("reactant exceeds the configured maximum bond order")
    n = len(state.symbols)
    for group in (roles.nucleophiles, roles.electrophiles, roles.leaving_groups):
        if any(i >= n for i in group):
            raise ValueError("role index outside the fixed atom inventory")
        if any(state.symbols[i] == "H" for i in group):
            raise ValueError("heavy-atom core roles cannot contain explicit hydrogen")
    attempt_bound = 0
    for intent in config.intents:
        if intent == CoreIntent.ADDITION:
            attempt_bound += len(roles.nucleophiles) * len(roles.electrophiles)
        elif intent == CoreIntent.SUBSTITUTION:
            attempt_bound += len(roles.nucleophiles) * len(roles.electrophiles) * max(1, len(roles.leaving_groups))
        else:
            attempt_bound += sum(1 for i in range(n) for j in range(i + 1, n) if state.be[i][j])
    if attempt_bound > config.max_attempts:
        raise ValueError("core-rule enumeration exceeds attempt budget; no partial candidates returned")

    attempted = invalid = 0
    pool: dict[tuple, CoreProposal] = {}

    def consider(intent: CoreIntent, event: EventProposal, score: int) -> None:
        nonlocal attempted, invalid
        attempted += 1
        try:
            target = event.apply(state)
            if not _within_bond_order(target, config.max_bond_order):
                raise ValueError("candidate exceeds the configured maximum bond order")
        except ValueError:
            invalid += 1
            return
        proposal = CoreProposal(event, intent, score, event.provenance)
        key = tuple(tuple(row) for row in target.be)
        old = pool.get(key)
        if old is None or _candidate_key(state, proposal) < _candidate_key(state, old):
            pool[key] = proposal

    for intent in config.intents:
        if intent == CoreIntent.ADDITION:
            for nu, electrophile in product(roles.nucleophiles, roles.electrophiles):
                if nu == electrophile or _bond_order(state, nu, electrophile) != 0:
                    continue
                event = EventProposal(
                    bond_edits=(BondEdit(nu, electrophile, 1),),
                    # In the closed-shell BE representation a new single bond
                    # consumes one electron pair from one endpoint. Splitting
                    # that pair as -1/-1 would create forbidden odd diagonal
                    # entries, so the rule records the donor-pair hypothesis.
                    electron_edits=(ElectronEdit(nu, -2),),
                    origin=EventOrigin.INPUT_HYPOTHESIS,
                    provenance="core-rule-v1:addition",
                )
                consider(intent, event, 2)
        elif intent == CoreIntent.SUBSTITUTION:
            for nu, electrophile, leaving in product(roles.nucleophiles, roles.electrophiles, roles.leaving_groups):
                if len({nu, electrophile, leaving}) < 3 or _bond_order(state, nu, electrophile) != 0 or _bond_order(state, electrophile, leaving) != 1:
                    continue
                event = EventProposal(
                    bond_edits=(BondEdit(nu, electrophile, 1), BondEdit(electrophile, leaving, -1)),
                    origin=EventOrigin.INPUT_HYPOTHESIS,
                    provenance="core-rule-v1:substitution",
                )
                consider(intent, event, 1)
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    if not _bond_order(state, i, j) or state.symbols[i] == "H" or state.symbols[j] == "H":
                        continue
                    # A heterolytic bond break assigns the released pair to
                    # one endpoint. Both assignments are explicit hypotheses;
                    # neither is presented as a mechanism or charge verdict.
                    for recipient in (i, j):
                        event = EventProposal(
                            bond_edits=(BondEdit(i, j, -1),),
                            electron_edits=(ElectronEdit(recipient, 2),),
                            origin=EventOrigin.INPUT_HYPOTHESIS,
                            provenance=f"core-rule-v1:bond-break:electron-to-{recipient}",
                        )
                        consider(intent, event, 3)

    buckets: dict[int, list[CoreProposal]] = {}
    for proposal in pool.values():
        buckets.setdefault(proposal.heuristic_score, []).append(proposal)
    chosen: list[CoreProposal] = []
    withheld = 0
    for score in sorted(buckets):
        group = sorted(buckets[score], key=lambda p: _candidate_key(state, p))
        if len(chosen) + len(group) > config.max_candidates:
            withheld += len(group)
            break
        chosen.extend(group)
    return CoreProposalSet(tuple(chosen), attempted, invalid, len(pool), bool(withheld), withheld, attempt_bound)
