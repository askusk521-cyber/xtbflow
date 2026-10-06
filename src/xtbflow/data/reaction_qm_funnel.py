"""Task-level admission funnel for Reaction-QM (``zenodo:18551029@v2``).

The funnel answers one question per record: *which training tasks can this
source record safely support?*  It deliberately does not ask whether the whole
record is "complete", because Reaction-QM carries reliable single labels (a
mapped reaction graph, a transition-state geometry, species-level energies)
that a whole-record gate would throw away together with the unreliable ones.

Evidence boundaries kept explicit here
--------------------------------------
* The TS reaction SMILES carries *global* atom-map numbers; the HDF5 TS rows
  follow those numbers in ascending order (verified separately by
  ``scripts/audit_reaction_qm_coordinate_mapping.py``).  The event label is
  derived from the graph difference of the two sides of that one string, so it
  needs no endpoint mapping.
* Endpoint species (``R*``/``P*``) carry *local* map numbers.  Putting their
  coordinates into the global atom order needs a graph correspondence.  A
  correspondence is only reported as certified when it is unique, or when every
  residual symmetry permutes atoms that are not part of the reaction centre.
  Element-sequence agreement is never treated as a mapping proof.
* No force field exists in the main HDF5 (forces live in the separate IRC
  asset), so ``energy_force`` is reported as unavailable instead of being
  silently conflated with the species-level ``EHG`` energies.
* Parent grouping is a *project-derived* rule (``GROUPING_RULE``), not an
  official Reaction-QM family annotation; the source ships no such table.

Everything here is a pure function of the supplied species data so it can be
unit-tested without the 2 GB HDF5 file.  Reading the file lives in
``scripts/build_reaction_qm_task_funnel.py``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

from rdkit import Chem

SCHEMA = "xtbflow-reaction-qm-task-funnel/v1"
SOURCE_DATASET = "Reaction-QM"
SOURCE_REVISION = "zenodo:18551029@v2"
GROUPING_RULE = "reaction_qm_reactant_product_set_union_v1"

# CHNOS with explicit hydrogens is the declared model scope (configs/task.yaml).
SUPPORTED_ATOMIC_NUMBERS = frozenset({1, 6, 7, 8, 16})

TASKS = ("event_only", "geometry_only", "energy_only", "energy_force", "paired_joint")

# Endpoint-to-global mapping statuses, strongest evidence first.
MAP_UNIQUE = "unique_graph"
MAP_SYMMETRIC_NONCENTER = "symmetric_noncenter"
MAP_AMBIGUOUS_CENTER = "ambiguous_center"
MAP_NO_MATCH = "no_graph_match"
CERTIFIED_MAPPINGS = frozenset({MAP_UNIQUE, MAP_SYMMETRIC_NONCENTER})

# Why each exclusion exists and what would unblock it.  Surfaced verbatim in the
# report so every dropped record carries a next action, as #94 requires.
NEXT_ACTIONS: dict[str, str] = {
    "missing_species_or_fields": "source record is structurally incomplete; keep quarantined and report to the data provider",
    "smiles_parse_failed": "inspect the specific SMILES; no automatic repair",
    "map_ids_invalid": "atom-map numbers missing, duplicated or not contiguous; cannot define a global atom order",
    "coordinate_shape_invalid": "coordinate array shape or finiteness is wrong for the declared atoms",
    "out_of_scope_element": "outside the declared CHNOS pool (configs/task.yaml); deferred_out_of_scope, not a chemistry label",
    "non_singlet_state": "open-shell or non-singlet species are out of scope for v0.1",
    "electronic_state_inconsistent": "electron-count parity disagrees with multiplicity; keep quarantined",
    "charge_not_conserved": "reactant, product and TS charges disagree; keep quarantined",
    "ts_coordinate_order_unverified": "TS coordinate rows do not follow global map order",
    "species_coordinate_order_unverified": "an endpoint species' rows do not follow its own map order",
    "side_atom_set_mismatch": "reactant and product sides do not describe the same mapped atoms",
    "species_inventory_mismatch": "endpoint species do not add up to the TS atom inventory",
    "aromatic_representation_pending": "aromatic bond orders need an explicit Kekule/representation rule before a bond-electron label is trusted (#71)",
    "non_integral_bond_edit": "a bond order changes by a non-integer amount; no integer event label can be derived",
    "no_bond_change": "reactant and product graphs are identical; there is no event to supervise",
    "energy_not_finite": "a species energy is missing or non-finite",
    "forces_absent_in_source": "the main HDF5 has no forces; they exist only in the separate IRC asset (not downloaded)",
    "endpoint_mapping_not_certified": "endpoint local->global correspondence is ambiguous or absent; needs an independent mapping table or geometry-independent evidence",
}


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SpeciesInput:
    """One HDF5 species group (``R0``, ``P1``, ``TS`` ...), source-agnostic."""

    name: str
    smiles: str
    atomic_numbers: tuple[int, ...]
    charge: int | None
    multiplicity: int | None
    energies: tuple[float, ...]  # source ``EHG`` triple; semantics not asserted here
    coordinate_shape: tuple[int, ...]
    coordinates_finite: bool


@dataclass(frozen=True)
class ReactionInput:
    record_id: str
    species: Mapping[str, SpeciesInput]


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------


@dataclass
class RecordFunnel:
    """Outcome for one record.  ``reasons`` keeps *every* failed gate, while
    ``first_block`` records the first one in funnel order so stage counts
    partition cleanly."""

    record_id: str
    readable: bool = False
    in_scope: bool = False
    identity_verified: bool = False
    tasks: dict[str, bool] = field(default_factory=lambda: {name: False for name in TASKS})
    reasons: list[str] = field(default_factory=list)
    first_block: str | None = None
    n_atoms: int | None = None
    charge: int | None = None
    multiplicity: int | None = None
    edits: tuple[tuple[int, int, int], ...] = ()
    aromatic: bool = False
    aromatic_integral_edits: bool = False
    reactant_key: str | None = None
    product_key: str | None = None
    mapping: dict[str, str] = field(default_factory=dict)
    assignment: dict[str, str] = field(default_factory=dict)
    mapping_sha256: dict[str, str] = field(default_factory=dict)
    ts_identity_evidence: str = "source_declared_ts_for_reaction_id_no_irc_in_this_asset"

    def block(self, reason: str) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)
        if self.first_block is None:
            self.first_block = reason


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------


def parse_mapped(smiles: str) -> Chem.Mol:
    """Parse one mapped fragment without sanitisation.

    Reaction-QM contains zwitterions and hypervalent heteroatoms that RDKit's
    valence sanitiser would reject; we only need the explicit graph, so the
    property cache is built non-strictly and ring info is perceived cheaply.
    """

    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.sanitize = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        raise ValueError("smiles_parse_failed")
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol


def split_side(side: str) -> list[Chem.Mol]:
    if not side:
        raise ValueError("smiles_parse_failed")
    return [parse_mapped(fragment) for fragment in side.split(".")]


def _map_ids(mols: Iterable[Chem.Mol]) -> list[int]:
    ids: list[int] = []
    for mol in mols:
        for atom in mol.GetAtoms():
            ids.append(int(atom.GetAtomMapNum()))
    return ids


def _contiguous_unique(ids: Sequence[int]) -> bool:
    return bool(ids) and all(value >= 1 for value in ids) and sorted(ids) == list(range(1, len(ids) + 1))


def bond_table(mols: Iterable[Chem.Mol]) -> dict[tuple[int, int], float]:
    """Map-id keyed bond orders (aromatic bonds are 1.5)."""

    table: dict[tuple[int, int], float] = {}
    for mol in mols:
        for bond in mol.GetBonds():
            a = int(bond.GetBeginAtom().GetAtomMapNum())
            b = int(bond.GetEndAtom().GetAtomMapNum())
            table[(min(a, b), max(a, b))] = float(bond.GetBondTypeAsDouble())
    return table


def atom_table(mols: Iterable[Chem.Mol]) -> dict[int, tuple[int, int]]:
    return {
        int(atom.GetAtomMapNum()): (int(atom.GetAtomicNum()), int(atom.GetFormalCharge()))
        for mol in mols
        for atom in mol.GetAtoms()
    }


def event_edits(
    reactant: Mapping[tuple[int, int], float], product: Mapping[tuple[int, int], float]
) -> tuple[tuple[tuple[int, int, int], ...], bool]:
    """Bond-order differences ``product - reactant`` keyed by global map ids.

    Returns ``(edits, integral)``.  The sign convention matches
    ``xtbflow.data.dft_da._event_edits`` so downstream event tensors agree.
    """

    edits: list[tuple[int, int, int]] = []
    integral = True
    for key in sorted(set(reactant) | set(product)):
        delta = product.get(key, 0.0) - reactant.get(key, 0.0)
        if abs(delta) < 1e-9:
            continue
        rounded = int(round(delta))
        if abs(delta - rounded) > 1e-9 or rounded == 0:
            integral = False
            continue
        edits.append((key[0], key[1], rounded))
    return tuple(edits), integral


def _stripped(mol: Chem.Mol) -> Chem.Mol:
    copy = Chem.Mol(mol)
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(0)
    copy.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(copy)
    return copy


def component_key(mol: Chem.Mol) -> str:
    """Map-independent, stereo-independent identity of one component.

    Stereo is dropped on purpose: for *grouping* we want stereoisomers of one
    parent system to stay together (conservative against leakage).
    """

    return Chem.MolToSmiles(_stripped(mol), isomericSmiles=False, canonical=True)


def set_key(mols: Iterable[Chem.Mol]) -> str:
    return ".".join(sorted(component_key(mol) for mol in mols))


# --------------------------------------------------------------------------
# Endpoint local -> global correspondence
# --------------------------------------------------------------------------


def _verify_mapping(species: Chem.Mol, component: Chem.Mol, mapping: Sequence[int]) -> bool:
    """Round-trip check: elements, formal charges and the *complete* bond set.

    Substructure matching alone is not an isomorphism test (the target may carry
    extra bonds), so every claim of correspondence is re-verified here.
    """

    if species.GetNumAtoms() != component.GetNumAtoms() or species.GetNumBonds() != component.GetNumBonds():
        return False
    if len(set(mapping)) != len(mapping):
        return False
    for idx, target in enumerate(mapping):
        a, b = species.GetAtomWithIdx(idx), component.GetAtomWithIdx(target)
        if a.GetAtomicNum() != b.GetAtomicNum() or a.GetFormalCharge() != b.GetFormalCharge():
            return False
    for bond in species.GetBonds():
        other = component.GetBondBetweenAtoms(
            mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()]
        )
        if other is None or other.GetBondType() != bond.GetBondType():
            return False
    return True


@dataclass(frozen=True)
class EndpointMapping:
    status: str
    sha256: str | None = None
    n_isomorphisms_found: int = 0
    monotone_consistent: bool = False


def map_endpoint(species: Chem.Mol, component: Chem.Mol, centre_ids: frozenset[int]) -> EndpointMapping:
    """Correspond one endpoint species to one global component.

    ``centre_ids`` are global map ids touched by the event.  A symmetric
    species is only certified when none of those atoms has a symmetry-equivalent
    partner, because otherwise *which* equivalent atom reacted is not recoverable
    from the graphs.  Symmetry classes can only be coarser than true orbits, so
    this test errs towards "ambiguous", never towards false certainty.
    """

    s, c = _stripped(species), _stripped(component)
    matches = c.GetSubstructMatches(s, uniquify=False, maxMatches=2, useChirality=False)
    # ``matches[k][i]`` is the component atom hit by species atom ``i``.
    valid = [m for m in matches if _verify_mapping(s, c, m)]
    if not valid:
        return EndpointMapping(MAP_NO_MATCH)
    chosen = valid[0]
    species_local = [int(a.GetAtomMapNum()) for a in species.GetAtoms()]
    component_global = [int(a.GetAtomMapNum()) for a in component.GetAtoms()]
    pairs = sorted((species_local[i], component_global[chosen[i]]) for i in range(len(chosen)))
    digest = hashlib.sha256(",".join(f"{l}:{g}" for l, g in pairs).encode()).hexdigest()
    # Is the source's local numbering simply "k-th smallest global id"?  Recorded
    # as evidence about the source convention, not used to certify anything.
    gl_sorted = sorted(g for _, g in pairs)
    monotone = all(g == gl_sorted[k] for k, (_, g) in enumerate(pairs))
    if len(valid) == 1 and len(matches) == 1:
        return EndpointMapping(MAP_UNIQUE, digest, 1, monotone)
    # Symmetric species: check that the reaction centre sits on singleton classes.
    try:
        # Chirality is excluded on purpose: ignoring it merges classes, which can
        # only make the centre test stricter (more "ambiguous"), never looser.
        ranks = list(
            Chem.CanonicalRankAtoms(s, breakTies=False, includeChirality=False, includeAtomMaps=False)
        )
    except Exception:  # noqa: BLE001 - unresolved symmetry must fail closed
        return EndpointMapping(MAP_AMBIGUOUS_CENTER, digest, len(valid), monotone)
    class_size = Counter(ranks)
    centre_local = [i for i in range(len(chosen)) if component_global[chosen[i]] in centre_ids]
    if any(class_size[ranks[i]] > 1 for i in centre_local):
        return EndpointMapping(MAP_AMBIGUOUS_CENTER, digest, len(valid), monotone)
    return EndpointMapping(MAP_SYMMETRIC_NONCENTER, digest, len(valid), monotone)


def _assign_components(
    species_keys: Sequence[tuple[str, str]], component_keys: Sequence[str], component_min_id: Sequence[int]
) -> dict[str, tuple[int, str]] | None:
    """Assign named species to component indices.

    Species whose component key is unique are assigned unambiguously.  When
    several species share one identity (identical reactant molecules), the
    source's convention is applied: species order ``R0, R1, ...`` follows the
    ascending smallest global map id of the component.  That convention held for
    every distinguishable case in the 2026-10-06 exploratory sample, but it is
    unverifiable for identical species, so those assignments are labelled
    ``rule_min_id_order`` and never counted as fully certified.
    """

    by_key: dict[str, list[int]] = defaultdict(list)
    for idx, key in enumerate(component_keys):
        by_key[key].append(idx)
    species_by_key: dict[str, list[str]] = defaultdict(list)
    for name, key in species_keys:
        species_by_key[key].append(name)
    out: dict[str, tuple[int, str]] = {}
    for key, names in species_by_key.items():
        indices = sorted(by_key.get(key, []), key=lambda i: component_min_id[i])
        if len(indices) != len(names):
            return None
        for rank, name in enumerate(sorted(names)):
            out[name] = (indices[rank], "unique" if len(names) == 1 else "rule_min_id_order")
    return out


# --------------------------------------------------------------------------
# Per-record analysis
# --------------------------------------------------------------------------


def _local_order_ok(species: SpeciesInput, mols: Sequence[Chem.Mol]) -> bool:
    """Species coordinate rows must follow the species' own map order."""

    ids = _map_ids(mols)
    if not _contiguous_unique(ids):
        return False
    z_by_id = {int(a.GetAtomMapNum()): int(a.GetAtomicNum()) for m in mols for a in m.GetAtoms()}
    return tuple(z_by_id[i] for i in range(1, len(ids) + 1)) == tuple(species.atomic_numbers)


def analyse_reaction(rec: ReactionInput) -> RecordFunnel:
    out = RecordFunnel(record_id=rec.record_id)
    ts = rec.species.get("TS")
    reactant_names = sorted(n for n in rec.species if n.startswith("R"))
    product_names = sorted(n for n in rec.species if n.startswith("P"))
    if ts is None or not reactant_names or not product_names:
        out.block("missing_species_or_fields")
        return out

    # ---- stage 1: readable -------------------------------------------------
    try:
        sides = ts.smiles.split(">>")
        if len(sides) != 2:
            raise ValueError("smiles_parse_failed")
        r_mols, p_mols = split_side(sides[0]), split_side(sides[1])
        species_mols = {n: split_side(rec.species[n].smiles) for n in reactant_names + product_names}
    except ValueError as exc:
        out.block(str(exc))
        return out
    n_atoms = len(ts.atomic_numbers)
    out.n_atoms = n_atoms
    for sp in rec.species.values():
        if (
            len(sp.coordinate_shape) != 2
            or sp.coordinate_shape != (len(sp.atomic_numbers), 3)
            or not sp.coordinates_finite
        ):
            out.block("coordinate_shape_invalid")
    r_ids, p_ids = _map_ids(r_mols), _map_ids(p_mols)
    if not (_contiguous_unique(r_ids) and _contiguous_unique(p_ids)):
        out.block("map_ids_invalid")
    if out.first_block is not None:
        return out
    out.readable = True

    # ---- stage 2: in scope -------------------------------------------------
    states_known = all(sp.charge is not None and sp.multiplicity is not None for sp in rec.species.values())
    if any(z not in SUPPORTED_ATOMIC_NUMBERS for z in ts.atomic_numbers):
        out.block("out_of_scope_element")
    if not states_known:
        out.block("electronic_state_inconsistent")
    else:
        if any(sp.multiplicity != 1 for sp in rec.species.values()):
            out.block("non_singlet_state")
        for sp in rec.species.values():
            electrons = sum(sp.atomic_numbers) - int(sp.charge)
            # Multiplicity 2S+1 and electron count must have opposite parity.
            if (electrons - (int(sp.multiplicity) - 1)) % 2 != 0:
                out.block("electronic_state_inconsistent")
        reactant_charge = sum(int(rec.species[n].charge) for n in reactant_names)
        product_charge = sum(int(rec.species[n].charge) for n in product_names)
        if not (reactant_charge == product_charge == int(ts.charge)):
            out.block("charge_not_conserved")
        out.charge, out.multiplicity = int(ts.charge), int(ts.multiplicity)
    if out.first_block is None:
        out.in_scope = True

    # ---- stage 3: identity verifiable -------------------------------------
    identity_reasons: list[str] = []
    z_r = {int(a.GetAtomMapNum()): int(a.GetAtomicNum()) for m in r_mols for a in m.GetAtoms()}
    z_p = {int(a.GetAtomMapNum()): int(a.GetAtomicNum()) for m in p_mols for a in m.GetAtoms()}
    if tuple(z_r[i] for i in range(1, n_atoms + 1)) != tuple(ts.atomic_numbers):
        identity_reasons.append("ts_coordinate_order_unverified")
    if z_r != z_p:
        identity_reasons.append("side_atom_set_mismatch")
    for name in reactant_names + product_names:
        if not _local_order_ok(rec.species[name], species_mols[name]):
            identity_reasons.append("species_coordinate_order_unverified")
            break
    if sum(len(rec.species[n].atomic_numbers) for n in reactant_names) != n_atoms or sum(
        len(rec.species[n].atomic_numbers) for n in product_names
    ) != n_atoms:
        identity_reasons.append("species_inventory_mismatch")
    for reason in identity_reasons:
        out.block(reason)
    if out.in_scope and not identity_reasons:
        out.identity_verified = True

    # Graph-derived labels do not depend on scope, only on the parsed sides, so
    # grouping keys are computed for every readable record (the group rule must
    # cover the whole source to avoid leaking across scope boundaries).
    try:
        out.reactant_key, out.product_key = set_key(r_mols), set_key(p_mols)
    except Exception:  # noqa: BLE001
        out.reactant_key = out.product_key = None

    if not out.identity_verified:
        return out

    # ---- tasks -------------------------------------------------------------
    r_bonds, p_bonds = bond_table(r_mols), bond_table(p_mols)
    edits, integral = event_edits(r_bonds, p_bonds)
    out.edits = edits
    out.aromatic = any(abs(v - 1.5) < 1e-9 for v in (*r_bonds.values(), *p_bonds.values()))
    out.aromatic_integral_edits = out.aromatic and integral and bool(edits)

    event_ok = True
    if out.aromatic:
        out.reasons.append("aromatic_representation_pending")
        event_ok = False
    if not integral:
        out.reasons.append("non_integral_bond_edit")
        event_ok = False
    elif not edits:
        out.reasons.append("no_bond_change")
        event_ok = False
    out.tasks["event_only"] = event_ok
    out.tasks["geometry_only"] = True

    finite_energies = all(
        len(sp.energies) == 3 and all(math.isfinite(e) for e in sp.energies) for sp in rec.species.values()
    )
    out.tasks["energy_only"] = finite_energies
    if not finite_energies:
        out.reasons.append("energy_not_finite")
    # Structural fact about this asset, not a per-record guess.
    out.reasons.append("forces_absent_in_source")

    # Endpoint correspondence is only needed for placing endpoint coordinates in
    # the global atom order, i.e. for the paired/joint view.
    centre = frozenset(i for edit in edits for i in edit[:2])
    mapping_ok = True
    for side_names, side_mols, tag in ((reactant_names, r_mols, "R"), (product_names, p_mols, "P")):
        comp_keys = [component_key(m) for m in side_mols]
        comp_min = [min(int(a.GetAtomMapNum()) for a in m.GetAtoms()) for m in side_mols]
        assigned = _assign_components(
            [(n, component_key(species_mols[n][0])) if len(species_mols[n]) == 1 else (n, "multi") for n in side_names],
            comp_keys,
            comp_min,
        )
        if assigned is None:
            for n in side_names:
                out.mapping[n] = MAP_NO_MATCH
            mapping_ok = False
            continue
        for n in side_names:
            comp_idx, how = assigned[n]
            result = map_endpoint(species_mols[n][0], side_mols[comp_idx], centre)
            out.mapping[n] = result.status
            out.assignment[n] = how
            if result.sha256:
                out.mapping_sha256[n] = result.sha256
            if result.status not in CERTIFIED_MAPPINGS:
                mapping_ok = False
    if not mapping_ok:
        out.reasons.append("endpoint_mapping_not_certified")
    out.tasks["paired_joint"] = bool(event_ok and mapping_ok)
    return out


# --------------------------------------------------------------------------
# Grouping and aggregation
# --------------------------------------------------------------------------


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministic root choice keeps group ids reproducible.
            self.parent[max(ra, rb)] = min(ra, rb)


def assign_parent_groups(rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Project-derived parent groups.

    A reaction links its reactant-set identity and its product-set identity, so
    a forward reaction, its reverse and chains that share a species end up in one
    group.  Records with the same reactant set therefore always share a group,
    which also keeps identical reactant inputs from straddling a split.
    The id is a hash of the smallest member key, independent of record order.
    """

    uf = _UnionFind()
    keyed: list[tuple[str, str]] = []
    for row in rows:
        r, p = row.get("reactant_key"), row.get("product_key")
        if not r or not p:
            continue
        uf.union(r, p)
        keyed.append((row["record_id"], r))
    return {rid: "rqm_g_" + hashlib.sha256(uf.find(r).encode()).hexdigest()[:16] for rid, r in keyed}


def record_row(result: RecordFunnel) -> dict[str, Any]:
    """Compact, JSON-serialisable manifest row (no coordinates, no labels)."""

    return {
        "record_id": result.record_id,
        "readable": result.readable,
        "in_scope": result.in_scope,
        "identity_verified": result.identity_verified,
        "tasks": dict(result.tasks),
        "first_block": result.first_block,
        "reasons": list(result.reasons),
        "n_atoms": result.n_atoms,
        "charge": result.charge,
        "multiplicity": result.multiplicity,
        "n_bond_edits": len(result.edits),
        "aromatic": result.aromatic,
        "aromatic_integral_edits": result.aromatic_integral_edits,
        "reactant_key": result.reactant_key,
        "product_key": result.product_key,
        "endpoint_mapping": dict(result.mapping),
        "endpoint_assignment": dict(result.assignment),
        "endpoint_mapping_sha256": dict(result.mapping_sha256),
        "ts_identity_evidence": result.ts_identity_evidence,
    }


def _stage_counts(rows: Sequence[Mapping[str, Any]], groups: Mapping[str, str], pick) -> dict[str, int]:
    chosen = [row for row in rows if pick(row)]
    return {
        "records": len(chosen),
        "parent_groups": len({groups[row["record_id"]] for row in chosen if row["record_id"] in groups}),
    }


def build_report(
    rows: Sequence[Mapping[str, Any]], groups: Mapping[str, str], *, source_records: int
) -> dict[str, Any]:
    """Aggregate per-record rows into the funnel report.

    Counts always carry both records and parent groups because #94 asks whether
    new data adds independent groups rather than just more frames.
    """

    funnel = [
        {"stage": "source_records", "records": source_records,
         "parent_groups": len(set(groups.values())),
         "semantics": "reaction records in the pinned HDF5"},
        {"stage": "readable", **_stage_counts(rows, groups, lambda r: r["readable"]),
         "semantics": "TS and endpoint species present, SMILES parse, contiguous unique map ids, coordinate shapes valid"},
        {"stage": "in_scope", **_stage_counts(rows, groups, lambda r: r["in_scope"]),
         "semantics": "CHNOS only, singlet, parity-consistent, charge conserved"},
        {"stage": "identity_verifiable", **_stage_counts(rows, groups, lambda r: r["identity_verified"]),
         "semantics": "TS and endpoint coordinate rows follow map order; both sides describe the same mapped atoms; species add up to the TS inventory"},
    ]
    tasks = {name: _stage_counts(rows, groups, lambda r, n=name: r["tasks"].get(n, False)) for name in TASKS}
    tasks["energy_force"]["unavailable_reason"] = "forces_absent_in_source"

    first_block = Counter(r["first_block"] for r in rows if r["first_block"])
    all_reasons = Counter(reason for r in rows for reason in r["reasons"])
    reasons = {}
    for name in sorted(set(first_block) | set(all_reasons)):
        reasons[name] = {
            "first_blocking_records": first_block.get(name, 0),
            "records_with_reason": all_reasons.get(name, 0),
            "next_action": NEXT_ACTIONS.get(name, "unclassified; inspect"),
        }

    verified = [r for r in rows if r["identity_verified"]]
    status_counts: Counter[str] = Counter()
    assignment_counts: Counter[str] = Counter()
    for r in verified:
        for name, status in r["endpoint_mapping"].items():
            status_counts[status] += 1
            assignment_counts[r["endpoint_assignment"].get(name, "none")] += 1
    paired = [r for r in verified if r["tasks"]["paired_joint"]]
    rule_assisted = sum(1 for r in paired if "rule_min_id_order" in r["endpoint_assignment"].values())

    sizes = Counter(groups.values())
    size_hist = Counter(min(n, 1000) for n in sizes.values())
    largest = max(sizes.values()) if sizes else 0
    return {
        "schema": SCHEMA,
        "source_dataset": SOURCE_DATASET,
        "source_revision": SOURCE_REVISION,
        "funnel": funnel,
        "tasks": tasks,
        "paired_joint_detail": {
            "records": len(paired),
            "records_relying_on_identical_species_order_rule": rule_assisted,
            "records_fully_certified_without_order_rule": len(paired) - rule_assisted,
        },
        "exclusion_reasons": reasons,
        "endpoint_mapping_status_counts": dict(status_counts),
        "endpoint_component_assignment_counts": dict(assignment_counts),
        "aromatic": {
            "records_with_aromatic_bonds_in_verified_scope": sum(1 for r in verified if r["aromatic"]),
            "of_which_integral_edits_only": sum(1 for r in verified if r["aromatic_integral_edits"]),
        },
        "grouping": {
            "rule": GROUPING_RULE,
            "official_family_annotation": False,
            "groups": len(sizes),
            "largest_group_records": largest,
            "group_size_histogram_capped_at_1000": {str(k): v for k, v in sorted(size_hist.items())},
        },
        "claim_limits": [
            "Task admission only: no record is promoted to a development_* or confirmatory Track-B split here.",
            "TS identity is source-declared for the reaction ID; the IRC asset that could verify it is not part of this audit.",
            "Parent groups are project-derived, not an official Reaction-QM family annotation.",
            "Energies are species-level EHG triples; no forces exist in the main HDF5.",
            "Redistribution licence for the Zenodo record is unspecified; this funnel records identities and hashes only.",
        ],
    }
