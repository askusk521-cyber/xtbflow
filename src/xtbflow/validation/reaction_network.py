"""Auditable reaction-network contracts for bounded multi-step discovery.

The network layer is deliberately independent of proposal models and quantum
chemistry backends.  It stores identities and evidence references, while
certification services remain responsible for producing the referenced
artifacts.  A network can therefore be built from cached records, resumed
after a failure, and reviewed without treating an unresolved attempt as a
negative chemical result.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


NETWORK_SCHEMA = "xtbflow-reaction-network/v1"
NODE_KINDS = frozenset({"reactant", "intermediate", "product", "unknown"})
EDGE_STATUSES = frozenset({"candidate", "unresolved", "certified"})
_UNKNOWN_STATE = frozenset({"unknown", "unavailable", "ambiguous", "missing", "unverified"})


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _state(value: Any, name: str, *, multiplicity: bool = False) -> int | str:
    if type(value) is int:
        if multiplicity and value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return value
    if isinstance(value, str) and value.strip().lower() in _UNKNOWN_STATE:
        return value.strip().lower()
    raise ValueError(f"{name} must be an integer or an explicit unresolved state")


def _sha256(value: Any) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("network identity fields must be JSON-compatible") from exc
    return hashlib.sha256(encoded).hexdigest()


def _hash(value: Any, name: str) -> str:
    normalized = _text(value, name).lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return normalized


def _symbols(value: Sequence[Sequence[Any]] | Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    if isinstance(value, Mapping):
        value = tuple(value.items())
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError("composition must be a non-empty sequence of [element, count]")
    entries: list[tuple[str, int]] = []
    for item in value:
        if isinstance(item, (str, bytes)) or not isinstance(item, Sequence) or len(item) != 2:
            raise ValueError("composition entries must be [element, count]")
        symbol = _text(item[0], "composition element")
        count = item[1]
        if type(count) is not int or count < 1:
            raise ValueError("composition counts must be positive integers")
        entries.append((symbol, count))
    if len({symbol for symbol, _ in entries}) != len(entries):
        raise ValueError("composition must contain each element once")
    return tuple(sorted(entries))


def _event(value: Iterable[Sequence[Any]]) -> tuple[tuple[int, int, int], ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("observed_event must be an iterable of bond edits")
    edits: set[tuple[int, int, int]] = set()
    try:
        iterator = iter(value)
    except TypeError as exc:
        raise ValueError("observed_event must be an iterable of bond edits") from exc
    for item in iterator:
        if isinstance(item, (str, bytes)) or not isinstance(item, Sequence) or len(item) != 3:
            raise ValueError("bond edits must be [atom_i, atom_j, delta_order]")
        i, j, delta = item
        if type(i) is not int or type(j) is not int or type(delta) is not int:
            raise ValueError("bond edits must contain strict integers")
        if i < 0 or j < 0 or i == j or delta == 0:
            raise ValueError("bond edits require distinct nonnegative atoms and nonzero delta")
        if i > j:
            i, j = j, i
        edits.add((i, j, delta))
    return tuple(sorted(edits))


def _refs(value: Iterable[str], name: str, *, required: bool) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an iterable of non-empty strings")
    try:
        refs = tuple(sorted({_text(item, name) for item in value}))
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of non-empty strings") from exc
    if required and not refs:
        raise ValueError(f"{name} must contain at least one reference")
    return refs


def _nonnegative_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a nonnegative finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{name} must be a nonnegative finite number")
    return converted


@dataclass(frozen=True)
class NetworkNode:
    """A composition/state/conformer identity that can anchor network edges."""

    node_id: str
    composition: tuple[tuple[str, int], ...]
    charge: int | str
    multiplicity: int | str
    connectivity_key: str
    stereo_key: str
    conformer_key: str
    geometry_hash: str
    geometry_locator: str
    protocol_id: str
    node_kind: str = "unknown"
    source_record_id: str = ""
    claim_limit: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _text(self.node_id, "node_id"))
        object.__setattr__(self, "composition", _symbols(self.composition))
        object.__setattr__(self, "charge", _state(self.charge, "charge"))
        object.__setattr__(self, "multiplicity", _state(self.multiplicity, "multiplicity", multiplicity=True))
        for name in ("connectivity_key", "stereo_key", "conformer_key", "geometry_locator", "protocol_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "geometry_hash", _hash(self.geometry_hash, "geometry_hash"))
        if self.node_kind not in NODE_KINDS:
            raise ValueError(f"unsupported node_kind: {self.node_kind}")
        if self.source_record_id:
            object.__setattr__(self, "source_record_id", _text(self.source_record_id, "source_record_id"))
        if not isinstance(self.claim_limit, str):
            raise ValueError("claim_limit must be a string")

    @property
    def identity_payload(self) -> dict[str, Any]:
        """Return the frozen identity used for deduplication.

        ``node_id`` and the source locator are intentionally excluded: the
        same observed endpoint must not be duplicated under two local IDs.
        Geometry, connectivity, stereo and electronic state remain explicit so
        distinct conformers or constitutional/stereochemical species survive.
        """

        return {
            "composition": [list(item) for item in self.composition],
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "connectivity_key": self.connectivity_key,
            "stereo_key": self.stereo_key,
            "conformer_key": self.conformer_key,
            "geometry_hash": self.geometry_hash,
            "protocol_id": self.protocol_id,
        }

    @property
    def identity_hash(self) -> str:
        return _sha256(self.identity_payload)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["composition"] = [list(item) for item in self.composition]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NetworkNode":
        if not isinstance(payload, Mapping):
            raise ValueError("node payload must be a mapping")
        return cls(**dict(payload))


@dataclass(frozen=True)
class EdgeCost:
    """Measured resource cost linked to ledger records for one edge attempt."""

    ledger_event_ids: tuple[str, ...]
    calculator_calls: int
    wall_seconds: float = 0.0
    cpu_core_seconds: float = 0.0
    gpu_seconds: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "ledger_event_ids", _refs(self.ledger_event_ids, "ledger_event_ids", required=True))
        if type(self.calculator_calls) is not int or self.calculator_calls < 0:
            raise ValueError("calculator_calls must be a nonnegative integer")
        for name in ("wall_seconds", "cpu_core_seconds", "gpu_seconds"):
            object.__setattr__(self, name, _nonnegative_number(getattr(self, name), name))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"ledger_event_ids": list(self.ledger_event_ids)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EdgeCost":
        return cls(**dict(payload))


@dataclass(frozen=True)
class ReactionEdge:
    """A proposal, unresolved attempt, or certified transition."""

    edge_id: str
    source_node_id: str
    target_node_id: str | None
    status: str
    proposal_id: str
    protocol_id: str
    observed_event: tuple[tuple[int, int, int], ...] = ()
    ts_reference: str | None = None
    mode_reference: str | None = None
    irc_reference: str | None = None
    endpoint_reference: str | None = None
    cost: EdgeCost | None = None
    failure_reason: str | None = None
    claim_limit: str = ""

    def __post_init__(self) -> None:
        for name in ("edge_id", "source_node_id", "proposal_id", "protocol_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.target_node_id is not None:
            object.__setattr__(self, "target_node_id", _text(self.target_node_id, "target_node_id"))
        if self.status not in EDGE_STATUSES:
            raise ValueError(f"unsupported edge status: {self.status}")
        object.__setattr__(self, "observed_event", _event(self.observed_event))
        for name in ("ts_reference", "mode_reference", "irc_reference", "endpoint_reference"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _text(value, name))
        if self.cost is not None and not isinstance(self.cost, EdgeCost):
            raise ValueError("cost must be an EdgeCost")
        if self.failure_reason is not None:
            object.__setattr__(self, "failure_reason", _text(self.failure_reason, "failure_reason"))
        if not isinstance(self.claim_limit, str):
            raise ValueError("claim_limit must be a string")
        if self.status == "certified":
            required = {
                "target_node_id": self.target_node_id,
                "ts_reference": self.ts_reference,
                "mode_reference": self.mode_reference,
                "irc_reference": self.irc_reference,
                "endpoint_reference": self.endpoint_reference,
                "cost": self.cost,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(f"certified edges require evidence and cost references: {missing}")
            if self.failure_reason is not None:
                raise ValueError("certified edges cannot carry failure_reason")
        elif self.status == "unresolved" and self.failure_reason is None:
            raise ValueError("unresolved edges require an explicit failure_reason")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["observed_event"] = [list(item) for item in self.observed_event]
        payload["cost"] = self.cost.to_dict() if self.cost is not None else None
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReactionEdge":
        if not isinstance(payload, Mapping):
            raise ValueError("edge payload must be a mapping")
        values = dict(payload)
        if values.get("cost") is not None:
            values["cost"] = EdgeCost.from_dict(values["cost"])
        return cls(**values)


@dataclass
class ReactionNetwork:
    """A durable graph with explicit frontier and certification transitions."""

    network_id: str
    atom_pool_id: str
    nodes: dict[str, NetworkNode] = field(default_factory=dict)
    edges: dict[str, ReactionEdge] = field(default_factory=dict)
    schema_version: str = NETWORK_SCHEMA

    def __post_init__(self) -> None:
        self.network_id = _text(self.network_id, "network_id")
        self.atom_pool_id = _text(self.atom_pool_id, "atom_pool_id")
        if self.schema_version != NETWORK_SCHEMA:
            raise ValueError(f"schema_version must be {NETWORK_SCHEMA!r}")
        if not isinstance(self.nodes, dict) or not isinstance(self.edges, dict):
            raise ValueError("nodes and edges must be dictionaries")
        existing_nodes = dict(self.nodes)
        existing_edges = dict(self.edges)
        self.nodes = {}
        self.edges = {}
        for node in existing_nodes.values():
            if not isinstance(node, NetworkNode):
                raise ValueError("nodes must contain NetworkNode values")
            self.add_node(node)
        for edge in existing_edges.values():
            if not isinstance(edge, ReactionEdge):
                raise ValueError("edges must contain ReactionEdge values")
            self.add_edge(edge)

    def add_node(self, node: NetworkNode) -> str:
        """Add a node idempotently while rejecting identity collisions."""

        if not isinstance(node, NetworkNode):
            raise TypeError("node must be a NetworkNode")
        prior = self.nodes.get(node.node_id)
        if prior is not None:
            if prior != node:
                raise ValueError(f"node ID already exists with different contents: {node.node_id}")
            return node.node_id
        for existing in self.nodes.values():
            if existing.identity_hash == node.identity_hash:
                raise ValueError(f"duplicate node identity under different IDs: {existing.node_id} and {node.node_id}")
        self.nodes[node.node_id] = node
        return node.node_id

    def add_edge(self, edge: ReactionEdge) -> str:
        if not isinstance(edge, ReactionEdge):
            raise TypeError("edge must be a ReactionEdge")
        if edge.source_node_id not in self.nodes:
            raise ValueError(f"edge source node is missing: {edge.source_node_id}")
        if edge.target_node_id is not None and edge.target_node_id not in self.nodes:
            raise ValueError(f"edge target node is missing: {edge.target_node_id}")
        prior = self.edges.get(edge.edge_id)
        if prior is not None:
            if prior != edge:
                raise ValueError(f"edge ID already exists with different contents: {edge.edge_id}")
            return edge.edge_id
        self.edges[edge.edge_id] = edge
        return edge.edge_id

    def promote_edge(
        self,
        edge_id: str,
        *,
        target_node_id: str,
        observed_event: Iterable[Sequence[Any]],
        ts_reference: str,
        mode_reference: str,
        irc_reference: str,
        endpoint_reference: str,
        cost: EdgeCost,
        claim_limit: str = "",
    ) -> ReactionEdge:
        """Replace a candidate/unresolved edge with a certified transition."""

        edge_id = _text(edge_id, "edge_id")
        prior = self.edges.get(edge_id)
        if prior is None:
            raise KeyError(f"unknown edge: {edge_id}")
        if prior.status == "certified":
            return prior
        promoted = ReactionEdge(
            edge_id=edge_id,
            source_node_id=prior.source_node_id,
            target_node_id=target_node_id,
            status="certified",
            proposal_id=prior.proposal_id,
            protocol_id=prior.protocol_id,
            observed_event=_event(observed_event),
            ts_reference=ts_reference,
            mode_reference=mode_reference,
            irc_reference=irc_reference,
            endpoint_reference=endpoint_reference,
            cost=cost,
            claim_limit=claim_limit or prior.claim_limit,
        )
        # ``add_edge`` intentionally rejects changing an existing ID.  A
        # promotion is the one explicit state transition allowed by the
        # network contract, so validate references first and then replace the
        # immutable record atomically.
        if promoted.target_node_id not in self.nodes:
            raise ValueError(f"edge target node is missing: {promoted.target_node_id}")
        self.edges[edge_id] = promoted
        return promoted

    def frontier(self) -> tuple[ReactionEdge, ...]:
        """Return candidate and unresolved edges in deterministic order."""

        return tuple(sorted((edge for edge in self.edges.values() if edge.status in {"candidate", "unresolved"}), key=lambda edge: edge.edge_id))

    def certified_edges(self) -> tuple[ReactionEdge, ...]:
        return tuple(sorted((edge for edge in self.edges.values() if edge.status == "certified"), key=lambda edge: edge.edge_id))

    def certified_path(self, target_node_id: str) -> tuple[str, ...] | None:
        """Return one shortest certified path from any root to a target."""

        target_node_id = _text(target_node_id, "target_node_id")
        if target_node_id not in self.nodes:
            raise KeyError(f"unknown target node: {target_node_id}")
        roots = {node.node_id for node in self.nodes.values() if node.node_kind == "reactant"}
        if target_node_id in roots:
            return ()
        adjacency: dict[str, list[ReactionEdge]] = {}
        for edge in self.certified_edges():
            assert edge.target_node_id is not None
            adjacency.setdefault(edge.source_node_id, []).append(edge)
        queue: list[tuple[str, tuple[str, ...]]] = [(root, ()) for root in sorted(roots)]
        seen = set(roots)
        while queue:
            node_id, path = queue.pop(0)
            for edge in adjacency.get(node_id, []):
                assert edge.target_node_id is not None
                next_path = path + (edge.edge_id,)
                if edge.target_node_id == target_node_id:
                    return next_path
                if edge.target_node_id not in seen:
                    seen.add(edge.target_node_id)
                    queue.append((edge.target_node_id, next_path))
        return None

    def cost_summary(self) -> dict[str, int | float]:
        certified = self.certified_edges()
        return {
            "certified_edges": len(certified),
            "calculator_calls": sum(edge.cost.calculator_calls for edge in certified if edge.cost is not None),
            "wall_seconds": sum(edge.cost.wall_seconds for edge in certified if edge.cost is not None),
            "cpu_core_seconds": sum(edge.cost.cpu_core_seconds for edge in certified if edge.cost is not None),
            "gpu_seconds": sum(edge.cost.gpu_seconds for edge in certified if edge.cost is not None),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "network_id": self.network_id,
            "atom_pool_id": self.atom_pool_id,
            "nodes": [self.nodes[key].to_dict() for key in sorted(self.nodes)],
            "edges": [self.edges[key].to_dict() for key in sorted(self.edges)],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReactionNetwork":
        if not isinstance(payload, Mapping):
            raise ValueError("network payload must be a mapping")
        if payload.get("schema_version") != NETWORK_SCHEMA:
            raise ValueError(f"network payload must use {NETWORK_SCHEMA!r}")
        raw_nodes = payload.get("nodes")
        raw_edges = payload.get("edges")
        if isinstance(raw_nodes, (str, bytes)) or not isinstance(raw_nodes, Sequence):
            raise ValueError("network nodes must be a sequence")
        if isinstance(raw_edges, (str, bytes)) or not isinstance(raw_edges, Sequence):
            raise ValueError("network edges must be a sequence")
        network = cls(_text(payload.get("network_id"), "network_id"), _text(payload.get("atom_pool_id"), "atom_pool_id"))
        for raw_node in raw_nodes:
            network.add_node(NetworkNode.from_dict(raw_node))
        for raw_edge in raw_edges:
            network.add_edge(ReactionEdge.from_dict(raw_edge))
        return network

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        temporary.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> "ReactionNetwork":
        target = Path(path)
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid reaction network file: {exc}") from exc
        return cls.from_dict(payload)

