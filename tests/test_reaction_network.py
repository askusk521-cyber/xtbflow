from __future__ import annotations

import hashlib
import json

import pytest

from xtbflow.validation import EdgeCost, NetworkNode, ReactionEdge, ReactionNetwork


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def node(node_id: str, *, kind: str = "intermediate", geometry: str = "geom", conformer: str | None = None) -> NetworkNode:
    return NetworkNode(
        node_id=node_id,
        composition=(("C", 1), ("H", 2), ("O", 1)),
        charge=0,
        multiplicity=1,
        connectivity_key="c=1;o=1",
        stereo_key="achiral",
        conformer_key=conformer or node_id,
        geometry_hash=digest(geometry),
        geometry_locator=f"n2://network/{node_id}.xyz",
        protocol_id="gfn2-v1",
        node_kind=kind,
        source_record_id=f"record-{node_id}",
        claim_limit="development-only",
    )


def cost() -> EdgeCost:
    return EdgeCost(("ledger-1", "ledger-2"), calculator_calls=7, wall_seconds=2.5, cpu_core_seconds=8.0)


def test_node_identity_keeps_conformers_and_connectivity_distinct():
    first = node("a", geometry="same", conformer="conf-1")
    same_identity = node("b", geometry="same", conformer="conf-1")
    different_conformer = node("c", geometry="other")
    assert first.identity_hash == same_identity.identity_hash
    assert first.identity_hash != different_conformer.identity_hash
    network = ReactionNetwork("campaign", "atom-pool-1")
    network.add_node(first)
    with pytest.raises(ValueError, match="duplicate node identity"):
        network.add_node(same_identity)
    network.add_node(different_conformer)


def test_unresolved_frontier_is_explicit_and_certified_promotion_is_recoverable(tmp_path):
    network = ReactionNetwork("campaign", "atom-pool-1")
    network.add_node(node("r0", kind="reactant"))
    network.add_node(node("i1"))
    frontier = ReactionEdge(
        edge_id="edge-1",
        source_node_id="r0",
        target_node_id=None,
        status="unresolved",
        proposal_id="proposal-1",
        protocol_id="gfn2-v1",
        observed_event=((0, 1, 1),),
        cost=cost(),
        failure_reason="irc_timeout",
        claim_limit="attempted_unresolved only",
    )
    network.add_edge(frontier)
    assert [edge.edge_id for edge in network.frontier()] == ["edge-1"]
    promoted = network.promote_edge(
        "edge-1",
        target_node_id="i1",
        observed_event=((1, 0, 1),),
        ts_reference="ts-1",
        mode_reference="mode-1",
        irc_reference="irc-1",
        endpoint_reference="endpoint-1",
        cost=cost(),
    )
    assert promoted.status == "certified"
    assert promoted.observed_event == ((0, 1, 1),)
    assert network.frontier() == ()
    assert network.certified_path("i1") == ("edge-1",)
    assert network.cost_summary()["calculator_calls"] == 7
    path = tmp_path / "network.json"
    network.save(path)
    restored = ReactionNetwork.load(path)
    assert restored.to_dict() == network.to_dict()
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == "xtbflow-reaction-network/v1"


def test_certified_edges_require_complete_evidence_and_missing_targets_fail():
    with pytest.raises(ValueError, match="certified edges require"):
        ReactionEdge(
            edge_id="edge",
            source_node_id="source",
            target_node_id="target",
            status="certified",
            proposal_id="proposal",
            protocol_id="protocol",
        )
    network = ReactionNetwork("campaign", "atom-pool-1")
    network.add_node(node("source", kind="reactant"))
    with pytest.raises(ValueError, match="target node is missing"):
        network.add_edge(
            ReactionEdge(
                edge_id="candidate",
                source_node_id="source",
                target_node_id="target",
                status="candidate",
                proposal_id="proposal",
                protocol_id="protocol",
            )
        )


def test_explicit_unknown_state_is_allowed_but_neutral_defaults_are_not_inferred():
    unresolved = node("unknown")
    unresolved = NetworkNode(
        **{**unresolved.to_dict(), "charge": "unknown", "multiplicity": "unverified"}
    )
    assert unresolved.charge == "unknown"
    assert unresolved.multiplicity == "unverified"
    with pytest.raises(ValueError, match="charge"):
        NetworkNode(**{**node("bad").to_dict(), "charge": None})


def test_certified_path_returns_none_for_unconnected_node():
    network = ReactionNetwork("campaign", "atom-pool-1")
    network.add_node(node("r0", kind="reactant"))
    network.add_node(node("isolated"))
    assert network.certified_path("isolated") is None

