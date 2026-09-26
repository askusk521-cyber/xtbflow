"""Combine reactant-only core events with the existing bounded relay enumerator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from mechai.data.events import LewisState
from mechai.data.proposals import RankedProposal, enumerate_relays

from .core_rules import CoreProposalSet, CoreRuleConfig, enumerate_core_events
from .roles import RoleSet, infer_roles


@dataclass(frozen=True)
class ProposalPool:
    core: CoreProposalSet
    relays: tuple[RankedProposal, ...]
    relay_attempts: int
    relay_invalid: int
    relay_truncated: bool


def build_proposal_pool(
    state: LewisState,
    coordinates: Sequence[Sequence[float]],
    *,
    roles: RoleSet | None = None,
    core_config: CoreRuleConfig | None = None,
    max_participating_waters: int = 2,
    max_relay_candidates: int = 32,
    max_enumeration_attempts: int = 100_000,
) -> ProposalPool:
    """Build a hypothesis pool without consulting target-derived information."""

    if type(max_enumeration_attempts) is not int or max_enumeration_attempts < 1:
        raise ValueError("max_enumeration_attempts must be a positive integer")
    roles = roles or infer_roles(state)
    core = enumerate_core_events(state, roles, config=core_config)
    relays_by_endpoint: dict[tuple[tuple[int, ...], ...], RankedProposal] = {}
    attempts = invalid = 0
    truncated = False
    for candidate in core.candidates:
        remaining_attempts = max_enumeration_attempts - attempts
        if remaining_attempts < 1:
            raise ValueError("proposal-pool enumeration exceeds the shared attempt budget; no partial candidates returned")
        relay_set = enumerate_relays(
            state,
            candidate.event,
            coordinates,
            donors=roles.donors,
            acceptors=roles.acceptors,
            water_oxygens=roles.water_oxygens,
            max_participating_waters=max_participating_waters,
            max_candidates=max_relay_candidates,
            max_enumeration_attempts=remaining_attempts,
        )
        attempts += relay_set.n_enumerated
        invalid += relay_set.n_invalid_bookkeeping
        truncated = truncated or relay_set.truncated
        for proposal in relay_set.candidates:
            endpoint = proposal.event.apply(state).be
            old = relays_by_endpoint.get(endpoint)
            if old is None or (proposal.heuristic_cost, proposal.event.provenance) < (old.heuristic_cost, old.event.provenance):
                relays_by_endpoint[endpoint] = proposal
    relays = tuple(sorted(relays_by_endpoint.values(), key=lambda item: (item.heuristic_cost, item.event.provenance)))
    return ProposalPool(core, relays, attempts, invalid, truncated)
