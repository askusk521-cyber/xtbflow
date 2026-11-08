# Reaction-network contract

`xtbflow.validation.reaction_network` is the storage boundary for bounded,
multi-step exploration. It does not run a calculator and it does not infer
missing chemistry metadata. A network records what was proposed, what was
attempted, and what was certified so a later scheduler can resume from the
same frontier without turning a failure into a negative label.

## Identity

Every `NetworkNode` carries a composition, total charge, multiplicity,
connectivity key, stereo key, conformer key, geometry SHA-256, geometry
locator, and protocol ID. Charge and multiplicity may be an explicit
`unknown`-style state for a quarantined or unresolved endpoint; the contract
never defaults them to neutral or singlet. The identity hash includes all
state, connectivity, stereo, conformer, geometry, and protocol fields. A
second local ID with the same identity is rejected, while different
conformers or connectivities remain separate nodes.

`atom_pool_id` belongs to the whole `ReactionNetwork`. It is the stable handle
for the frozen composition/fragment pool used by a campaign. The network layer
does not claim that two atom pools are chemically comparable.

## Edge states and evidence

`ReactionEdge.status` is one of:

- `candidate`: a proposal is recorded and may be scheduled;
- `unresolved`: work was attempted but remains incomplete; `failure_reason` is
  mandatory and does not mean that the proposed chemistry is false;
- `certified`: a target endpoint and references to TS, mode, IRC, endpoint,
  protocol, and measured cost are all present.

`observed_event` stores endpoint bond edits as canonical `(atom_i, atom_j,
delta_order)` tuples. Certification requires `EdgeCost`, which points to one
or more ledger event IDs and stores calculator calls and elapsed resource
seconds. These are accounting references; the network does not validate the
contents of the referenced artifacts.

`ReactionNetwork.promote_edge` is the only state transition that replaces an
existing candidate or unresolved edge. It requires all certification evidence
and a target node. `frontier()` returns candidate and unresolved edges in a
stable order, while `certified_path()` traverses only certified edges and
returns `None` when no connected path has been established.

## Persistence and limits

`save` writes a deterministic JSON document matching
`schemas/reaction_network.schema.json`; `load` validates the schema version,
identities, references, and state transitions before returning a network.
Node and edge IDs, evidence references, and ledger IDs are opaque strings and
must be resolved by their owning data or validation services. A passing
contract test or a serialized fixture is software evidence only. It is not a
TS/IRC result, a newly discovered reaction, a coverage estimate, or a claim of
method superiority.

The contract supports later fixed or feedback schedulers, but it deliberately
does not choose a frontier, compare strategies, or perform quantum-chemistry
work. Those activities need a separately frozen protocol, finite budget, and
independent evidence record.
