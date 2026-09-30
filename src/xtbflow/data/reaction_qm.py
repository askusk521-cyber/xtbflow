"""Reaction-QM/RGD1 source identities and read-only reaction contracts.

The public Reaction-QM and RGD1 releases are large archives and are never
checked into this repository.  This module contains the source-level identity
and a strict, dependency-light contract for rows loaded from an n2 cache.  A
row is useful for development only after its reactant/product atom mapping,
electronic state, coordinates and reference protocol have all been verified.

Endpoint graph differences are bookkeeping labels.  They are not electron
density trajectories, arrow-pushing mechanisms or an IRC certificate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .records import _ELEMENT_ATOMIC_NUMBERS, canonical_hash


REACTION_QM_RECORD = "Reaction-QM"
REACTION_QM_REVISION = "zenodo:18551029@v2"
REACTION_QM_URL = "https://doi.org/10.5281/zenodo.18551029"
RGD1_RECORD = "RGD1"
RGD1_REVISION = "figshare:21066901@v6"
RGD1_URL = "https://doi.org/10.6084/m9.figshare.21066901.v6"
EVENT_RULE_VERSION = "mapped-endpoint-difference-v1"
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_UNRESOLVED = frozenset({"", "unknown", "unavailable", "ambiguous", "missing", "not_provided", "unverified"})
_EXPLICIT_COORDINATE_MAP_EVIDENCE = frozenset({"explicit_source_map", "independent_mapping_table"})


@dataclass(frozen=True)
class SourceFile:
    """One immutable public source asset declaration.

    ``expected_sha256`` stays ``None`` until the exact bytes have been
    downloaded and hashed in the cache.  A missing digest is therefore an
    explicit audit state and cannot accidentally be treated as verified.
    """

    name: str
    url: str
    role: str
    declared_size_bytes: int | None = None
    official_md5: str | None = None
    expected_sha256: str | None = None
    download_status: str = "not_downloaded"

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.url.strip() or not self.role.strip():
            raise ValueError("source file identity fields must be nonempty")
        if self.declared_size_bytes is not None and (
            type(self.declared_size_bytes) is not int or self.declared_size_bytes <= 0
        ):
            raise ValueError("declared_size_bytes must be a positive integer")
        if self.official_md5 is not None and (
            len(self.official_md5) != 32 or any(c not in "0123456789abcdefABCDEF" for c in self.official_md5)
        ):
            raise ValueError("official_md5 must be a hexadecimal MD5 digest")
        if self.expected_sha256 is not None and (
            len(self.expected_sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in self.expected_sha256)
        ):
            raise ValueError("expected_sha256 must be a hexadecimal SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "role": self.role,
            "declared_size_bytes": self.declared_size_bytes,
            "official_md5": self.official_md5,
            "expected_sha256": self.expected_sha256,
            "download_status": self.download_status,
        }


@dataclass(frozen=True)
class PublicSourceConfig:
    """Pinned source identity and field mapping for one public dataset."""

    dataset: str
    revision: str
    landing_url: str
    license_status: str
    field_mapping: Mapping[str, Any]
    files: tuple[SourceFile, ...]
    claim_limit: str

    def __post_init__(self) -> None:
        if not self.dataset.strip() or not self.revision.strip() or not self.landing_url.strip():
            raise ValueError("source identity fields must be nonempty")
        if not self.license_status.strip() or not self.claim_limit.strip():
            raise ValueError("license_status and claim_limit must be explicit")
        if not isinstance(self.field_mapping, Mapping) or not self.field_mapping:
            raise ValueError("field_mapping must be a nonempty mapping")
        if not self.files:
            raise ValueError("source config must declare at least one file")
        object.__setattr__(self, "field_mapping", json.loads(json.dumps(self.field_mapping, sort_keys=True)))

    @property
    def digest(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "revision": self.revision,
            "landing_url": self.landing_url,
            "license_status": self.license_status,
            "field_mapping": dict(self.field_mapping),
            "files": [item.to_dict() for item in self.files],
            "claim_limit": self.claim_limit,
        }


REACTION_QM_SOURCE = PublicSourceConfig(
    dataset=REACTION_QM_RECORD,
    revision=REACTION_QM_REVISION,
    landing_url=REACTION_QM_URL,
    license_status="record_license_field_not_exposed_in_landing_page; verify_before_redistribution",
    field_mapping={
        "reaction_info": {
            "reaction_id": ["reaction_id"],
            "reaction_smiles": ["reaction_smiles"],
            "reference_observables": ["dE_dagger", "dE", "dH_dagger", "dH", "dG_dagger", "dG"],
        },
        "hdf5": {
            "reaction_container": "recursive group containing R*, P* and TS entries",
            "species_fields": ["smiles", "EHG", "charge", "multiplicity", "atomic_numbers", "coordinates"],
            "coordinate_units": "angstrom",
            "energy_units": "hartree",
        },
        "irc": {"fields": ["atomic_numbers", "coordinates", "energies", "forces"]},
    },
    files=(
        SourceFile(
            "B3LYPD3_TZVP_reaction_info.csv",
            "https://zenodo.org/records/18551029/files/B3LYPD3_TZVP_reaction_info.csv?download=1",
            "reaction_info",
            declared_size_bytes=71468002,
            official_md5="6bbf509808a5193770a2093874fb7ad8",
            expected_sha256="2facf37090a4cba872394ec6ab0c360b011d41acff9658268ff2eb61e6fd5ae1",
            download_status="verified_n2_cache",
        ),
        SourceFile(
            "B3LYPD3_TZVP.h5",
            "https://zenodo.org/records/18551029/files/B3LYPD3_TZVP.h5?download=1",
            "geometry_and_ts",
            declared_size_bytes=2054981016,
            official_md5="2c572a68849e805bc0dbb257a53a1fd2",
            expected_sha256="3d0fc655819a9a2747f554a9025cd36cbdffd1175c4a1d40934b6fe5530af82a",
            download_status="verified_n2_cache",
        ),
        SourceFile(
            "B3LYPD3_TZVP_IRC.h5",
            "https://zenodo.org/records/18551029/files/B3LYPD3_TZVP_IRC.h5?download=1",
            "irc_path",
            official_md5="782a4e5e8099de8f2b8e0e90128028cb",
        ),
    ),
    claim_limit=(
        "B3LYP-D3/TZVP endpoint and TS records with source mapping can support "
        "development labels after audit; endpoint graph differences are not "
        "electron-density paths or complete mechanisms."
    ),
)


RGD1_SOURCE = PublicSourceConfig(
    dataset=RGD1_RECORD,
    revision=RGD1_REVISION,
    landing_url=RGD1_URL,
    license_status="repository_gpl3; Figshare_data_redistribution_terms_require_record_check",
    field_mapping={
        "mapped_smiles": {
            "reaction_id": "reaction",
            "reactant_smiles": "reactant",
            "product_smiles": "product",
            "barrier_and_energy": ["DE_F", "DE_B", "DG_F", "DG_B", "DH"],
        },
        "dft_info": {
            "reaction_id": "channel",
            "reactant_smiles": "reactant",
            "product_smiles": "product",
            "reaction_family": "type",
            "independent_reactant_system": "R_ind",
        },
        "geometry": {
            "reaction_file": "RGD1_CHNO.h5",
            "endpoint_file": "RGD1_RPs.h5",
            "mapping_file": "RandP_smiles.txt",
        },
    },
    files=(
        SourceFile("RGD1CHNO_AMsmiles.csv", "https://ndownloader.figshare.com/files/40272727", "mapped_index", declared_size_bytes=60724906, expected_sha256="7b0ecae0a7f7b2439a03bed4bfbf80fbc9b647b0823e5a8e7b1e3be96d0ad05e", download_status="verified_n2_cache"),
        SourceFile("DFT_reaction_info.csv", "https://ndownloader.figshare.com/files/40273231", "dft_reaction_info", declared_size_bytes=33353396, expected_sha256="afb3bfb9ca5e44fb1a6ea44664da56e0ef2a2ee95c82b22d651a778fad5e0b00", download_status="verified_n2_cache"),
        SourceFile("RandP_smiles.txt", "https://ndownloader.figshare.com/files/43291989", "endpoint_mapping", declared_size_bytes=4174042, expected_sha256="6ca74bafe9286ac1c6cba48e700e5f5100e0bb3d6a8f05592efd214ffdc624ff", download_status="verified_n2_cache"),
        SourceFile("RGD1_CHNO.h5", "https://ndownloader.figshare.com/files/38170323", "geometry_and_ts", declared_size_bytes=1340003744, expected_sha256="ed125b4cb1eac9af670a7cae8b9d29c88a9f8be24d0206a027f0f2c035f8f268", download_status="verified_n2_cache"),
        SourceFile("RGD1_RPs.h5", "https://ndownloader.figshare.com/files/43293162", "endpoint_geometry", declared_size_bytes=467575944),
    ),
    claim_limit=(
        "RGD1 is a cross-source validation source only. The aggregate reaction "
        "type and bond counts do not replace mapped graphs, explicit electronic "
        "state or a physically validated TS."
    ),
)


def source_configs() -> tuple[PublicSourceConfig, PublicSourceConfig]:
    return REACTION_QM_SOURCE, RGD1_SOURCE


def _digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_source_files(config: PublicSourceConfig, cache_dir: str | Path) -> list[dict[str, Any]]:
    """Audit cache bytes without downloading or mutating any source asset."""

    root = Path(cache_dir)
    result: list[dict[str, Any]] = []
    for spec in config.files:
        path = root / spec.name
        row = {
            "name": spec.name,
            "url": spec.url,
            "role": spec.role,
            "relative_path": spec.name,
            "declared_size_bytes": spec.declared_size_bytes,
        }
        if not path.is_file():
            row.update({"status": "unavailable", "size_bytes": None, "sha256": None, "md5": None})
            result.append(row)
            continue
        sha256 = _digest(path)
        md5 = _digest(path, "md5")
        status = "verified" if spec.expected_sha256 is not None and sha256 == spec.expected_sha256 else "observed_pending_sha256"
        if spec.expected_sha256 is not None and sha256 != spec.expected_sha256:
            status = "hash_mismatch"
        if spec.official_md5 is not None and md5 != spec.official_md5:
            status = "official_md5_mismatch"
        row.update({"status": status, "size_bytes": path.stat().st_size, "sha256": sha256, "md5": md5})
        result.append(row)
    return result


@dataclass(frozen=True)
class Bond:
    """A mapped undirected bond with an explicit order."""

    atom_i: int
    atom_j: int
    order: float
    aromatic: bool = False

    def __post_init__(self) -> None:
        if type(self.atom_i) is not int or type(self.atom_j) is not int or self.atom_i <= 0 or self.atom_j <= 0 or self.atom_i == self.atom_j:
            raise ValueError("bond atom map numbers must be distinct positive integers")
        if not math.isfinite(float(self.order)) or self.order <= 0:
            raise ValueError("bond order must be a positive finite number")
        if type(self.aromatic) is not bool:
            raise ValueError("aromatic must be boolean")

    @property
    def key(self) -> tuple[int, int]:
        return tuple(sorted((self.atom_i, self.atom_j)))

    def to_dict(self) -> dict[str, Any]:
        return {"atom_i": self.key[0], "atom_j": self.key[1], "order": self.order, "aromatic": self.aromatic}


@dataclass(frozen=True)
class MappedGraph:
    """An atom-mapped endpoint graph, independent of coordinate storage order."""

    atoms: Mapping[int, str]
    bonds: tuple[Bond, ...] = ()

    def __post_init__(self) -> None:
        atoms = dict(self.atoms)
        if not atoms or any(type(k) is not int or k <= 0 or not isinstance(v, str) or not v.strip() for k, v in atoms.items()):
            raise ValueError("mapped graph atoms require positive integer IDs and symbols")
        seen: set[tuple[int, int]] = set()
        normalized: list[Bond] = []
        for bond in self.bonds:
            if not isinstance(bond, Bond) or bond.key in seen:
                raise ValueError("mapped graph bonds must be unique Bond objects")
            if any(atom not in atoms for atom in bond.key):
                raise ValueError("bond references an atom missing from mapped graph")
            seen.add(bond.key)
            normalized.append(bond)
        # Preserve source SMILES atom order for coordinate-order checks.  The
        # canonical hash sorts mapping keys, so serialization remains stable.
        object.__setattr__(self, "atoms", dict(atoms))
        object.__setattr__(self, "bonds", tuple(sorted(normalized, key=lambda x: x.key)))

    @property
    def atom_maps(self) -> frozenset[int]:
        return frozenset(self.atoms)

    @property
    def bond_map(self) -> dict[tuple[int, int], Bond]:
        return {bond.key: bond for bond in self.bonds}

    def to_dict(self) -> dict[str, Any]:
        return {"atoms": {str(k): v for k, v in self.atoms.items()}, "bonds": [bond.to_dict() for bond in self.bonds]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MappedGraph":
        if not isinstance(payload, Mapping):
            raise ValueError("mapped graph payload must be a mapping")
        raw_atoms = payload.get("atoms")
        if not isinstance(raw_atoms, Mapping):
            raise ValueError("mapped graph atoms must be a mapping")
        atoms: dict[int, str] = {}
        for key, value in raw_atoms.items():
            try:
                atom_id = int(key)
            except (TypeError, ValueError) as exc:
                raise ValueError("mapped atom IDs must be integers") from exc
            if atom_id in atoms:
                raise ValueError("duplicate atom mapping")
            atoms[atom_id] = value
        bonds: list[Bond] = []
        for raw in payload.get("bonds", ()):
            if isinstance(raw, Mapping):
                i, j, order, aromatic = raw.get("atom_i"), raw.get("atom_j"), raw.get("order"), raw.get("aromatic", False)
            elif isinstance(raw, Sequence) and len(raw) in {3, 4}:
                i, j, order = raw[:3]
                aromatic = raw[3] if len(raw) == 4 else False
            else:
                raise ValueError("bond entries must be mappings or 3/4-tuples")
            bonds.append(Bond(int(i), int(j), float(order), bool(aromatic)))
        return cls(atoms, tuple(bonds))

    @classmethod
    def from_smiles(cls, smiles: str) -> "MappedGraph":
        """Parse mapped SMILES through RDKit when the optional dependency exists."""

        try:
            from rdkit import Chem  # type: ignore
        except ImportError as exc:
            raise RuntimeError("RDKit is required to parse mapped SMILES") from exc
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError("mapped SMILES must be a nonempty string")
        atoms: dict[int, str] = {}
        bonds: list[Bond] = []
        for component in smiles.split("."):
            # ``sanitize=False`` preserves explicitly mapped hydrogens.  The
            # public reaction files map H atoms too; default RDKit sanitizing
            # folds those atoms into implicit hydrogen counts and silently
            # destroys the atom-map contract.
            mol = Chem.MolFromSmiles(component, sanitize=False)
            if mol is None:
                raise ValueError("mapped SMILES failed RDKit parsing")
            for atom in mol.GetAtoms():
                atom_id = atom.GetAtomMapNum()
                if atom_id <= 0 or atom_id in atoms:
                    raise ValueError("mapped SMILES has missing or duplicate atom mappings")
                atoms[atom_id] = atom.GetSymbol()
            for raw_bond in mol.GetBonds():
                i = raw_bond.GetBeginAtom().GetAtomMapNum()
                j = raw_bond.GetEndAtom().GetAtomMapNum()
                bonds.append(Bond(i, j, float(raw_bond.GetBondTypeAsDouble()), bool(raw_bond.GetIsAromatic())))
        return cls(atoms, tuple(bonds))


def _graph_bond_map(graph: MappedGraph, *, side: str) -> dict[tuple[int, int], Bond]:
    if any(bond.aromatic for bond in graph.bonds):
        raise ValueError(f"ambiguous_aromatic_bond:{side}")
    return graph.bond_map


def derive_event_label(
    reactant_graph: MappedGraph,
    product_graph: MappedGraph,
    *,
    source_record_id: str,
    generation_rule_version: str = EVENT_RULE_VERSION,
) -> dict[str, Any]:
    """Derive ``B_product - B_reactant`` with auditable provenance."""

    if reactant_graph.atom_maps != product_graph.atom_maps:
        raise ValueError("atom_mapping_incomplete: reactant/product atom map sets differ")
    if any(reactant_graph.atoms[k] != product_graph.atoms[k] for k in reactant_graph.atom_maps):
        raise ValueError("atom_mapping_element_mismatch")
    reactant_bonds = _graph_bond_map(reactant_graph, side="reactant")
    product_bonds = _graph_bond_map(product_graph, side="product")
    edits: list[dict[str, Any]] = []
    for key in sorted(set(reactant_bonds) | set(product_bonds)):
        before = reactant_bonds.get(key)
        after = product_bonds.get(key)
        delta = (after.order if after else 0.0) - (before.order if before else 0.0)
        if delta:
            edits.append({"atom_i": key[0], "atom_j": key[1], "delta": delta})
    if not edits:
        raise ValueError("ambiguous_no_bond_change")
    core = {"bond_edits": edits, "label_type": "endpoint_bond_difference"}
    output_hash = canonical_hash(core)
    return {
        **core,
        "evidence": "derived_under_contract",
        "source_record_id": source_record_id,
        "generation_rule_version": generation_rule_version,
        "input_graph_hash": canonical_hash(reactant_graph.to_dict()),
        "output_event_label_hash": output_hash,
        "claim_limit": "Endpoint graph difference only; not an electron-density path or complete mechanism.",
    }


def reaction_qm_record_hash(
    *,
    record_id: str,
    parent_reaction_id: str,
    reaction_family_id: str,
    independent_reactant_system_id: str,
    repeated_ts_group: str,
    reactant_graph: MappedGraph,
    product_graph: MappedGraph,
    reactant_coordinates: Sequence[Sequence[float]],
    ts_coordinates: Sequence[Sequence[float]],
    charge: int,
    multiplicity: int,
    reference_protocol: Mapping[str, str],
    coordinate_map_evidence: Any,
    source_revision: str = REACTION_QM_REVISION,
) -> str:
    """Hash the complete source payload, including coordinates and state.

    The hash is deliberately computed from the values that enter the training
    contract.  A source identity hash over only an ID or SMILES is insufficient
    because it would remain unchanged if geometry or electronic state changed.
    """

    return canonical_hash({
        "record_id": record_id,
        "parent_reaction_id": parent_reaction_id,
        "reaction_family_id": reaction_family_id,
        "independent_reactant_system_id": independent_reactant_system_id,
        "repeated_ts_group": repeated_ts_group,
        "reactant_graph": reactant_graph.to_dict(),
        "product_graph": product_graph.to_dict(),
        "reactant_coordinates": [[float(value) for value in row] for row in reactant_coordinates],
        "ts_coordinates": [[float(value) for value in row] for row in ts_coordinates],
        "charge": charge,
        "multiplicity": multiplicity,
        "reference_protocol": dict(reference_protocol),
        "coordinate_map_evidence": coordinate_map_evidence,
        "source_revision": source_revision,
    })


def _coordinates(value: Sequence[Sequence[Any]], *, atoms: int, name: str) -> tuple[tuple[float, float, float], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != atoms:
        raise ValueError(f"{name} must contain one xyz row per atom")
    rows: list[tuple[float, float, float]] = []
    for row in value:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or len(row) != 3:
            raise ValueError(f"{name} must contain one xyz row per atom")
        values = tuple(float(component) for component in row)
        if not all(math.isfinite(component) for component in values):
            raise ValueError(f"{name} contains a nonfinite coordinate")
        rows.append(values)
    return tuple(rows)


@dataclass(frozen=True)
class ReactionQMRecord:
    """One Reaction-QM-style record after strict source-level validation."""

    record_id: str
    parent_reaction_id: str
    reaction_family_id: str
    independent_reactant_system_id: str
    repeated_ts_group: str
    reactant_graph: MappedGraph
    product_graph: MappedGraph
    reactant_coordinates: tuple[tuple[float, float, float], ...]
    ts_coordinates: tuple[tuple[float, float, float], ...]
    charge: int
    multiplicity: int
    reference_protocol: Mapping[str, str]
    source_record_hash: str
    event_label: Mapping[str, Any]
    admission: str = "quarantine"
    quarantine_reasons: tuple[str, ...] = field(default_factory=tuple)
    coordinate_map_evidence: Any = "unverified"
    source_revision: str = REACTION_QM_REVISION

    def __post_init__(self) -> None:
        for name in ("record_id", "parent_reaction_id", "reaction_family_id", "independent_reactant_system_id", "repeated_ts_group", "source_record_hash"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        if not isinstance(self.source_revision, str) or not self.source_revision.strip():
            raise ValueError("source_revision must be explicit")
        if type(self.charge) is not int or type(self.multiplicity) is not int or self.multiplicity < 1:
            raise ValueError("charge and multiplicity must be explicit strict integers")
        if self.reactant_graph.atom_maps != self.product_graph.atom_maps:
            raise ValueError("reactant/product atom mappings must match")
        object.__setattr__(self, "reactant_coordinates", _coordinates(self.reactant_coordinates, atoms=len(self.reactant_graph.atoms), name="reactant_coordinates"))
        object.__setattr__(self, "ts_coordinates", _coordinates(self.ts_coordinates, atoms=len(self.reactant_graph.atoms), name="ts_coordinates"))
        if not _SHA256.fullmatch(self.source_record_hash):
            raise ValueError("source_record_hash must be a SHA-256 digest")
        if not isinstance(self.reference_protocol, Mapping) or not self.reference_protocol:
            raise ValueError("reference_protocol must be explicit")
        map_evidence = self.coordinate_map_evidence
        if isinstance(map_evidence, Mapping):
            map_source = map_evidence.get("source")
            map_ids = map_evidence.get("map_ids")
            if not isinstance(map_source, str) or not map_source.strip():
                raise ValueError("coordinate_map_evidence source must be explicit")
            if isinstance(map_ids, (str, bytes)) or not isinstance(map_ids, Sequence):
                raise ValueError("coordinate_map_evidence map_ids must be a sequence")
            if tuple(map_ids) != tuple(self.reactant_graph.atoms):
                raise ValueError("coordinate_map_evidence map_ids must match coordinate row order")
            map_evidence = {"source": map_source, "map_ids": list(map_ids)}
            object.__setattr__(self, "coordinate_map_evidence", map_evidence)
        elif not isinstance(map_evidence, str) or not map_evidence.strip():
            raise ValueError("coordinate_map_evidence must be explicit")
        electronic_state_reason: str | None = None
        try:
            electron_count = sum(_ELEMENT_ATOMIC_NUMBERS[symbol] for symbol in self.reactant_graph.atoms.values()) - self.charge
        except KeyError as exc:
            if self.admission != "quarantine":
                raise ValueError(f"unsupported element for electronic-state validation: {exc.args[0]}") from exc
            electronic_state_reason = "electronic_state_unverifiable"
        else:
            unpaired = self.multiplicity - 1
            if electron_count < 0 or unpaired > electron_count or (electron_count - unpaired) % 2:
                if self.admission != "quarantine":
                    raise ValueError("charge and multiplicity are inconsistent with the electron count")
                electronic_state_reason = "electronic_state_inconsistent"
        if not isinstance(self.event_label, Mapping) or self.event_label.get("evidence") != "derived_under_contract":
            raise ValueError("event_label must carry derived_under_contract evidence")
        expected_event_label = derive_event_label(
            self.reactant_graph,
            self.product_graph,
            source_record_id=self.record_id,
        )
        if dict(self.event_label) != expected_event_label:
            raise ValueError("event_label does not match the mapped endpoint graphs")
        reasons = tuple(sorted(set(self.quarantine_reasons)))
        if electronic_state_reason is not None and self.admission == "quarantine":
            reasons = tuple(sorted(set((*reasons, electronic_state_reason))))
        if any(not isinstance(reason, str) or not reason.strip() for reason in reasons):
            raise ValueError("quarantine reasons must be nonempty strings")
        object.__setattr__(self, "quarantine_reasons", reasons)
        if self.admission not in {"quarantine", "development_train", "development_validation", "development_test", "confirmatory"}:
            raise ValueError("unsupported admission state")
        if self.admission != "quarantine" and reasons:
            raise ValueError("admitted rows cannot carry quarantine reasons")
        if self.admission != "quarantine":
            unresolved = [
                name for name, value in (
                    ("parent_reaction_id", self.parent_reaction_id),
                    ("reaction_family_id", self.reaction_family_id),
                    ("independent_reactant_system_id", self.independent_reactant_system_id),
                    ("repeated_ts_group", self.repeated_ts_group),
                    ("source_revision", self.source_revision),
                )
                if value.strip().lower() in _UNRESOLVED
            ]
            if unresolved:
                raise ValueError(f"admitted records cannot have unresolved grouping fields: {unresolved}")
            if not isinstance(self.coordinate_map_evidence, Mapping) or self.coordinate_map_evidence.get("source") not in _EXPLICIT_COORDINATE_MAP_EVIDENCE:
                raise ValueError("admitted records require explicit coordinate-to-map evidence")
        expected_hash = reaction_qm_record_hash(
            record_id=self.record_id,
            parent_reaction_id=self.parent_reaction_id,
            reaction_family_id=self.reaction_family_id,
            independent_reactant_system_id=self.independent_reactant_system_id,
            repeated_ts_group=self.repeated_ts_group,
            reactant_graph=self.reactant_graph,
            product_graph=self.product_graph,
            reactant_coordinates=self.reactant_coordinates,
            ts_coordinates=self.ts_coordinates,
            charge=self.charge,
            multiplicity=self.multiplicity,
            reference_protocol=self.reference_protocol,
            coordinate_map_evidence=self.coordinate_map_evidence,
            source_revision=self.source_revision,
        )
        if self.source_record_hash.lower() != expected_hash:
            raise ValueError("source_record_hash does not bind the complete record payload")

    def reactant_input(self) -> dict[str, Any]:
        """Deployment-visible input view; product, TS and event labels stay out."""

        return {
            "graph": self.reactant_graph.to_dict(),
            "coordinates": [list(row) for row in self.reactant_coordinates],
            "charge": self.charge,
            "multiplicity": self.multiplicity,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "parent_reaction_id": self.parent_reaction_id,
            "reaction_family_id": self.reaction_family_id,
            "independent_reactant_system_id": self.independent_reactant_system_id,
            "repeated_ts_group": self.repeated_ts_group,
            "reactant_graph": self.reactant_graph.to_dict(),
            "product_graph": self.product_graph.to_dict(),
            "reactant_coordinates": [list(row) for row in self.reactant_coordinates],
            "ts_coordinates": [list(row) for row in self.ts_coordinates],
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "reference_protocol": dict(self.reference_protocol),
            "source_record_hash": self.source_record_hash,
            "event_label": dict(self.event_label),
            "admission": self.admission,
            "quarantine_reasons": list(self.quarantine_reasons),
            "coordinate_map_evidence": self.coordinate_map_evidence,
            "source_revision": self.source_revision,
        }


class ReactionQMLoader:
    """Read-only loader for the two small Reaction-QM source assets.

    HDF5 access is lazy so source audits and unit tests can run without adding
    h5py to the core package.  No loader method writes to a cache or modifies
    source bytes.
    """

    def __init__(self, *, cache_dir: str | Path, config: PublicSourceConfig = REACTION_QM_SOURCE) -> None:
        self.cache_dir = Path(cache_dir)
        self.config = config

    def audit_files(self) -> list[dict[str, Any]]:
        return audit_source_files(self.config, self.cache_dir)

    def _verified_asset(self, name: str) -> Path:
        """Return a cache path only after its declared SHA-256 is verified."""

        specs = {spec.name: spec for spec in self.config.files}
        spec = specs.get(name)
        if spec is None:
            raise ValueError(f"asset is not declared by source config: {name}")
        path = self.cache_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if spec.expected_sha256 is None:
            raise ValueError(f"asset is not hash-verified: {name} (expected_sha256 is missing)")
        sha256 = _digest(path)
        if sha256 != spec.expected_sha256:
            raise ValueError(f"asset is not hash-verified: {name} (sha256 mismatch)")
        if spec.official_md5 is not None and _digest(path, "md5") != spec.official_md5:
            raise ValueError(f"asset is not hash-verified: {name} (official md5 mismatch)")
        return path

    def reaction_info(self) -> Iterator[dict[str, str]]:
        path = self._verified_asset("B3LYPD3_TZVP_reaction_info.csv")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            expected = {"reaction_id", "reaction_smiles"}
            if not expected.issubset(reader.fieldnames or ()):
                raise ValueError("Reaction-QM reaction_info schema is missing reaction_id/reaction_smiles")
            for row in reader:
                yield dict(row)

    def hdf5_reaction_count(self) -> int:
        path = self._verified_asset("B3LYPD3_TZVP.h5")
        try:
            import h5py  # type: ignore
        except ImportError as exc:
            raise RuntimeError("h5py is required to inspect Reaction-QM HDF5") from exc
        count = 0
        with h5py.File(path, "r") as handle:
            for group in _reaction_groups(handle):
                count += 1
        return count

    def iter_records(self, *, limit: int | None = None) -> Iterator[ReactionQMRecord]:
        """Yield strictly validated endpoint/TS records from the HDF5 cache.

        The source does not expose mother/family/independent-system IDs in the
        reaction-info CSV.  Those fields therefore remain explicit ``unknown``
        values and the rows stay quarantined until an independent grouping
        table is supplied.  A malformed row raises with its source ID so the
        caller can retain a failure entry in an audit report.
        """

        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("limit must be a positive integer when supplied")
        path = self._verified_asset("B3LYPD3_TZVP.h5")
        try:
            import h5py  # type: ignore
        except ImportError as exc:
            raise RuntimeError("h5py is required to load Reaction-QM HDF5") from exc
        yielded = 0
        with h5py.File(path, "r") as handle:
            for group in _reaction_groups(handle):
                record_id = str(group.name).rsplit("/", 1)[-1]
                if not record_id.startswith("RXN_"):
                    continue
                row = self._record_from_hdf5_group(record_id, group)
                yielded += 1
                yield row
                if limit is not None and yielded >= limit:
                    return

    def _record_from_hdf5_group(self, record_id: str, group: Any) -> ReactionQMRecord:
        required = {"TS"}
        if not required.issubset(group.keys()):
            raise ValueError(f"{record_id}: TS group is missing")
        ts = group["TS"]
        raw_smiles = ts["smiles"][()]
        smiles = raw_smiles.decode() if isinstance(raw_smiles, bytes) else str(raw_smiles)
        sides = smiles.split(">>")
        if len(sides) != 2:
            raise ValueError(f"{record_id}: TS reaction SMILES must contain one reaction arrow")
        reactant_graph = _mapped_graph_from_side(sides[0])
        product_graph = _mapped_graph_from_side(sides[1])
        event_label = derive_event_label(reactant_graph, product_graph, source_record_id=record_id)

        reactant_coordinates, reactant_numbers = _concatenate_species_coordinates(group, prefix="R")
        ts_coordinates = tuple(tuple(float(value) for value in row) for row in ts["coordinates"][()].tolist())
        expected_numbers = tuple(_atomic_number(symbol) for symbol in reactant_graph.atoms.values())
        # The HDF5 coordinate arrays carry atomic numbers but no atom-map
        # number per coordinate.  A different order from mapped SMILES is an
        # unresolved correspondence, even when the element multiset agrees.
        if reactant_numbers != expected_numbers:
            raise ValueError(f"{record_id}: coordinate_atom_order_unverified: reactant coordinates lack certified map order")
        if tuple(int(value) for value in ts["atomic_numbers"][()].tolist()) != expected_numbers:
            raise ValueError(f"{record_id}: coordinate_atom_order_unverified: TS coordinates lack certified map order")
        protocol = {"method": "B3LYP-D3", "basis": "TZVP", "coordinates": "angstrom", "energy": "hartree"}
        coordinate_map_evidence = "source_hdf5_has_no_atom_map_ids"
        parent_reaction_id = "unknown"
        reaction_family_id = "unknown"
        independent_reactant_system_id = "unknown"
        repeated_ts_group = "unknown"
        source_record_hash = reaction_qm_record_hash(
            record_id=record_id,
            parent_reaction_id=parent_reaction_id,
            reaction_family_id=reaction_family_id,
            independent_reactant_system_id=independent_reactant_system_id,
            repeated_ts_group=repeated_ts_group,
            reactant_graph=reactant_graph,
            product_graph=product_graph,
            reactant_coordinates=reactant_coordinates,
            ts_coordinates=ts_coordinates,
            charge=int(ts["charge"][()]),
            multiplicity=int(ts["multiplicity"][()]),
            reference_protocol=protocol,
            coordinate_map_evidence=coordinate_map_evidence,
            source_revision=self.config.revision,
        )
        return ReactionQMRecord(
            record_id=record_id,
            parent_reaction_id=parent_reaction_id,
            reaction_family_id=reaction_family_id,
            independent_reactant_system_id=independent_reactant_system_id,
            repeated_ts_group=repeated_ts_group,
            reactant_graph=reactant_graph,
            product_graph=product_graph,
            reactant_coordinates=reactant_coordinates,
            ts_coordinates=ts_coordinates,
            charge=int(ts["charge"][()]),
            multiplicity=int(ts["multiplicity"][()]),
            reference_protocol=protocol,
            source_record_hash=source_record_hash,
            event_label=event_label,
            admission="quarantine",
            quarantine_reasons=(
                "coordinate_map_unverified",
                "reaction_family_unresolved",
                "parent_reaction_unresolved",
                "independent_reactant_system_unresolved",
                "repeated_ts_group_unresolved",
            ),
            coordinate_map_evidence=coordinate_map_evidence,
            source_revision=self.config.revision,
        )


def _reaction_groups(group: Any, prefix: str = "") -> Iterator[Any]:
    """Yield HDF5 groups that contain reaction species without assuming depth."""

    keys = list(group.keys())
    labels = {str(key).upper() for key in keys}
    if any(label.startswith("R") for label in labels) and any(label.startswith("P") for label in labels) and any(label.startswith("TS") for label in labels):
        yield group
        return
    for key in keys:
        child = group[key]
        if hasattr(child, "keys"):
            yield from _reaction_groups(child, f"{prefix}/{key}")


_ATOMIC_SYMBOLS = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 14: "Si", 15: "P", 16: "S", 17: "Cl",
}


def _atomic_number(symbol: str) -> int:
    for number, candidate in _ATOMIC_SYMBOLS.items():
        if candidate == symbol:
            return number
    raise ValueError(f"unsupported source element for atom-order check: {symbol}")


def _mapped_graph_from_side(side: str) -> MappedGraph:
    atoms: dict[int, str] = {}
    bonds: list[Bond] = []
    try:
        from rdkit import Chem  # type: ignore
    except ImportError as exc:
        raise RuntimeError("RDKit is required to load mapped Reaction-QM SMILES") from exc
    for component in side.split("."):
        mol = Chem.MolFromSmiles(component, sanitize=False)
        if mol is None:
            raise ValueError("mapped TS SMILES failed parsing")
        for atom in mol.GetAtoms():
            atom_id = atom.GetAtomMapNum()
            if atom_id <= 0 or atom_id in atoms:
                raise ValueError("mapped TS SMILES has missing or duplicate atom mappings")
            atoms[atom_id] = atom.GetSymbol()
        for raw_bond in mol.GetBonds():
            bonds.append(Bond(raw_bond.GetBeginAtom().GetAtomMapNum(), raw_bond.GetEndAtom().GetAtomMapNum(), float(raw_bond.GetBondTypeAsDouble()), bool(raw_bond.GetIsAromatic())))
    return MappedGraph(atoms, tuple(bonds))


def _concatenate_species_coordinates(group: Any, *, prefix: str) -> tuple[tuple[tuple[float, float, float], ...], tuple[int, ...]]:
    def sort_key(key: str) -> tuple[int, int | str, str]:
        suffix = key[len(prefix):]
        return (0, int(suffix), key) if suffix.isdigit() else (1, suffix, key)

    keys = sorted((str(key) for key in group.keys() if str(key).upper().startswith(prefix.upper())), key=sort_key)
    rows: list[tuple[float, float, float]] = []
    numbers: list[int] = []
    for key in keys:
        species = group[key]
        if "coordinates" not in species or "atomic_numbers" not in species:
            raise ValueError(f"{key}: species coordinates or atomic numbers are missing")
        coordinates = species["coordinates"][()].tolist()
        atomic_numbers = species["atomic_numbers"][()].tolist()
        if len(coordinates) != len(atomic_numbers):
            raise ValueError(f"{key}: coordinate and atomic-number counts differ")
        rows.extend(tuple(float(value) for value in row) for row in coordinates)
        numbers.extend(int(value) for value in atomic_numbers)
    return tuple(rows), tuple(numbers)


__all__ = [
    "Bond", "EVENT_RULE_VERSION", "MappedGraph", "PublicSourceConfig", "REACTION_QM_SOURCE", "RGD1_SOURCE",
    "ReactionQMLoader", "ReactionQMRecord", "SourceFile", "audit_source_files", "derive_event_label", "reaction_qm_record_hash", "source_configs",
]
