"""Frozen reactant-only seed contracts for independent search pilots.

The seed layer is deliberately independent from reference labels.  A seed is
an immutable reactant inventory, electronic state, and provenance record.  It
can be generated before a model or quantum-chemistry driver is qualified, and
it never accepts product, transition-state, or target-selected environment
fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .records import CHNOS_ELEMENTS, canonical_hash


SEED_SCHEMA = "xtbflow-independent-reactant-seed/v1"
MANIFEST_SCHEMA = "xtbflow-independent-seed-manifest/v1"
UNKNOWN = "unknown"
FORBIDDEN_SEED_KEYS = frozenset(
    {
        "product",
        "product_coordinates",
        "mapped_product",
        "ts",
        "ts_geometry",
        "reference_ts",
        "reference_mode",
        "irc",
        "active_water_from_ts",
        "target_active_water",
        "sealed_test_label",
        "barrier",
        "energy",
    }
)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _strict_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer (bool is not accepted)")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _canonical_symbols(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError("symbols must be a non-empty sequence")
    symbols = tuple(_text(item, "symbol") for item in value)
    if any(symbol not in CHNOS_ELEMENTS for symbol in symbols):
        raise ValueError("seed contains an element outside the CHNOS scope")
    if "H" not in symbols:
        raise ValueError("independent seeds require explicit hydrogen atoms")
    return symbols


def _coordinates(value: Any, n_atoms: int) -> tuple[tuple[float, float, float], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != n_atoms:
        raise ValueError("coordinates must have one xyz row per atom")
    rows: list[tuple[float, float, float]] = []
    for index, row in enumerate(value):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or len(row) != 3:
            raise ValueError(f"coordinates[{index}] must be an xyz triplet")
        rows.append(tuple(_finite_float(component, f"coordinates[{index}]") for component in row))
    return tuple(rows)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    try:
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON serializable") from exc
    return dict(value)


@dataclass(frozen=True)
class IndependentSeed:
    """One frozen reactant-side input with no target-derived information."""

    seed_id: str
    parent_id: str
    parent_group: str
    source_uri: str
    source_revision: str
    source_hash: str
    license_ref: str
    symbols: tuple[str, ...]
    coordinates: tuple[tuple[float, float, float], ...]
    charge: int
    multiplicity: int
    charge_spin_evidence: str
    microstate_id: str
    microstate_evidence: str
    conformer_id: str
    approach_id: str
    environment_id: str
    environment_provenance: str
    declared_intent: str
    generation_version: str = "independent-seeds-v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "seed_id",
            "parent_id",
            "parent_group",
            "source_uri",
            "source_revision",
            "source_hash",
            "license_ref",
            "charge_spin_evidence",
            "microstate_id",
            "microstate_evidence",
            "conformer_id",
            "approach_id",
            "environment_id",
            "environment_provenance",
            "declared_intent",
            "generation_version",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        symbols = _canonical_symbols(self.symbols)
        coordinates = _coordinates(self.coordinates, len(symbols))
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "coordinates", coordinates)
        charge = _strict_int(self.charge, "charge")
        multiplicity = _strict_int(self.multiplicity, "multiplicity")
        if multiplicity != 1:
            raise ValueError("independent pilot scope requires closed-shell singlet multiplicity=1")
        object.__setattr__(self, "charge", charge)
        object.__setattr__(self, "multiplicity", multiplicity)
        metadata = _mapping(self.metadata, "metadata")
        forbidden = FORBIDDEN_SEED_KEYS.intersection(metadata)
        if forbidden:
            raise ValueError(f"target-derived fields are forbidden in seed metadata: {sorted(forbidden)}")
        object.__setattr__(self, "metadata", metadata)

    @property
    def atom_inventory_hash(self) -> str:
        # Atom ordering is a coordinate-mapping concern; inventory equality is
        # composition/state equality and therefore uses a canonical multiset.
        return canonical_hash({"symbols": sorted(self.symbols), "charge": self.charge, "multiplicity": self.multiplicity})

    @property
    def input_fingerprint(self) -> str:
        """Hash only fields that the proposal/search side may read."""

        return canonical_hash(self.reactant_input())

    def reactant_input(self) -> dict[str, Any]:
        return {
            "symbols": list(self.symbols),
            "coordinates": [list(row) for row in self.coordinates],
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "microstate_id": self.microstate_id,
            "conformer_id": self.conformer_id,
            "approach_id": self.approach_id,
            "environment_id": self.environment_id,
            "declared_intent": self.declared_intent,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SEED_SCHEMA,
            "seed_id": self.seed_id,
            "parent_id": self.parent_id,
            "parent_group": self.parent_group,
            "source_uri": self.source_uri,
            "source_revision": self.source_revision,
            "source_hash": self.source_hash,
            "license_ref": self.license_ref,
            "symbols": list(self.symbols),
            "coordinates": [list(row) for row in self.coordinates],
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "charge_spin_evidence": self.charge_spin_evidence,
            "microstate_id": self.microstate_id,
            "microstate_evidence": self.microstate_evidence,
            "conformer_id": self.conformer_id,
            "approach_id": self.approach_id,
            "environment_id": self.environment_id,
            "environment_provenance": self.environment_provenance,
            "declared_intent": self.declared_intent,
            "generation_version": self.generation_version,
            "metadata": dict(self.metadata),
            "atom_inventory_hash": self.atom_inventory_hash,
            "input_fingerprint": self.input_fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IndependentSeed":
        if not isinstance(payload, Mapping):
            raise ValueError("seed payload must be an object")
        value = dict(payload)
        if value.get("schema", SEED_SCHEMA) != SEED_SCHEMA:
            raise ValueError(f"unsupported seed schema: {value.get('schema')!r}")
        allowed = {
            "schema",
            "seed_id",
            "parent_id",
            "parent_group",
            "source_uri",
            "source_revision",
            "source_hash",
            "license_ref",
            "symbols",
            "coordinates",
            "charge",
            "multiplicity",
            "charge_spin_evidence",
            "microstate_id",
            "microstate_evidence",
            "conformer_id",
            "approach_id",
            "environment_id",
            "environment_provenance",
            "declared_intent",
            "generation_version",
            "metadata",
            "atom_inventory_hash",
            "input_fingerprint",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unsupported independent seed fields: {sorted(unknown)}")
        value.pop("schema", None)
        claimed_inventory = value.pop("atom_inventory_hash", None)
        claimed_input = value.pop("input_fingerprint", None)
        seed = cls(**value)
        if claimed_inventory is not None and claimed_inventory != seed.atom_inventory_hash:
            raise ValueError("atom_inventory_hash does not match seed content")
        if claimed_input is not None and claimed_input != seed.input_fingerprint:
            raise ValueError("input_fingerprint does not match seed content")
        return seed


@dataclass(frozen=True)
class SeedManifest:
    """Deterministic, append-only freeze of independent seeds."""

    generation_version: str
    config_hash: str
    freeze_id: str
    status: str
    seeds: tuple[IndependentSeed, ...]
    max_seeds_per_parent: int = 32
    manifest_version: str = "independent-pilot-v1"
    blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.generation_version, "generation_version")
        _text(self.config_hash, "config_hash")
        _text(self.freeze_id, "freeze_id")
        if self.status not in {"ready", "blocked"}:
            raise ValueError("manifest status must be ready or blocked")
        if type(self.max_seeds_per_parent) is not int or self.max_seeds_per_parent < 1:
            raise ValueError("max_seeds_per_parent must be positive")
        seeds = tuple(sorted(self.seeds, key=lambda item: item.seed_id))
        if len({seed.seed_id for seed in seeds}) != len(seeds):
            raise ValueError("seed_id values must be unique")
        by_parent: dict[str, list[IndependentSeed]] = {}
        for seed in seeds:
            by_parent.setdefault(seed.parent_id, []).append(seed)
        for parent_id, rows in by_parent.items():
            if len(rows) > self.max_seeds_per_parent:
                raise ValueError(f"parent {parent_id} exceeds max_seeds_per_parent")
            inventories = {row.atom_inventory_hash for row in rows}
            if len(inventories) != 1:
                raise ValueError(f"parent {parent_id} changes atom inventory or electronic state")
        blockers = tuple(sorted({_text(item, "blocker") for item in self.blockers}))
        if self.status == "ready" and blockers:
            raise ValueError("ready manifests cannot carry blockers")
        if self.status == "blocked" and not blockers:
            raise ValueError("blocked manifests require explicit blockers")
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "blockers", blockers)

    @property
    def parent_count(self) -> int:
        return len({seed.parent_id for seed in self.seeds})

    @property
    def manifest_hash(self) -> str:
        return canonical_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": MANIFEST_SCHEMA,
            "manifest_version": self.manifest_version,
            "generation_version": self.generation_version,
            "config_hash": self.config_hash,
            "freeze_id": self.freeze_id,
            "status": self.status,
            "max_seeds_per_parent": self.max_seeds_per_parent,
            "seed_count": len(self.seeds),
            "parent_count": self.parent_count,
            "blockers": list(self.blockers),
            "seeds": [seed.to_dict() for seed in self.seeds],
        }
        if include_hash:
            payload["manifest_hash"] = self.manifest_hash
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SeedManifest":
        if not isinstance(payload, Mapping) or payload.get("schema") != MANIFEST_SCHEMA:
            raise ValueError("unsupported independent seed manifest schema")
        value = dict(payload)
        claimed = value.pop("manifest_hash", None)
        rows = tuple(IndependentSeed.from_dict(row) for row in value.pop("seeds", ()))
        for derived in ("seed_count", "parent_count"):
            value.pop(derived, None)
        manifest = cls(seeds=rows, **value)
        if claimed is not None and claimed != manifest.manifest_hash:
            raise ValueError("manifest_hash does not match manifest content")
        return manifest


def _parent_row_to_seeds(parent: Mapping[str, Any], *, generation_version: str) -> list[IndependentSeed]:
    if not isinstance(parent, Mapping):
        raise ValueError("each parent must be an object")
    value = dict(parent)
    parent_id = _text(value.pop("parent_id", None), "parent_id")
    parent_group = _text(value.pop("parent_group", parent_id), "parent_group")
    rows = value.pop("seeds", None)
    if rows is None:
        rows = [value]
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError(f"parent {parent_id} seeds must be a sequence")
    output: list[IndependentSeed] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"parent {parent_id} seed {index} must be an object")
        seed_data = dict(row)
        seed_data.pop("schema", None)
        seed_data.pop("atom_inventory_hash", None)
        seed_data.pop("input_fingerprint", None)
        seed_data.setdefault("seed_id", f"{parent_id}:seed-{index:03d}")
        seed_data.setdefault("parent_id", parent_id)
        seed_data.setdefault("parent_group", parent_group)
        seed_data.setdefault("generation_version", generation_version)
        output.append(IndependentSeed(**seed_data))
    return output


def build_seed_manifest(config: Mapping[str, Any], *, freeze_id: str) -> SeedManifest:
    """Build a manifest from a config without accessing any QC backend."""

    if not isinstance(config, Mapping):
        raise ValueError("seed config must be an object")
    if config.get("schema") != "xtbflow-independent-pilot/v1":
        raise ValueError("unsupported independent pilot config schema")
    generation_version = _text(config.get("generation_version", "independent-seeds-v1"), "generation_version")
    scope = config.get("scope", {})
    if not isinstance(scope, Mapping):
        raise ValueError("scope must be an object")
    allowed = tuple(scope.get("allowed_elements", sorted(CHNOS_ELEMENTS)))
    if set(allowed) != set(CHNOS_ELEMENTS):
        raise ValueError("independent seed scope must remain exactly CHNOS")
    multiplicities = tuple(scope.get("allowed_multiplicities", [1]))
    if multiplicities != (1,):
        raise ValueError("independent seed scope only supports multiplicity=1")
    max_per_parent = scope.get("max_seeds_per_parent", 32)
    if type(max_per_parent) is not int or max_per_parent < 1:
        raise ValueError("scope.max_seeds_per_parent must be positive")
    parents = config.get("parents", [])
    if isinstance(parents, (str, bytes)) or not isinstance(parents, Sequence):
        raise ValueError("parents must be a sequence")
    seeds: list[IndependentSeed] = []
    for parent in parents:
        seeds.extend(_parent_row_to_seeds(parent, generation_version=generation_version))
    blockers = []
    if not seeds:
        blockers.append("no_source_qualified_reactant_seeds")
    qualification = config.get("qualification", {})
    if not isinstance(qualification, Mapping):
        raise ValueError("qualification must be an object")
    if qualification.get("status", "blocked") != "qualified":
        blockers.append("source_or_electronic_state_qualification_incomplete")
    return SeedManifest(
        generation_version=generation_version,
        config_hash=canonical_hash(config),
        freeze_id=freeze_id,
        status="ready" if not blockers else "blocked",
        seeds=tuple(seeds),
        max_seeds_per_parent=max_per_parent,
        blockers=tuple(blockers),
    )


def load_seed_manifest(path: str | Path) -> SeedManifest:
    return SeedManifest.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def write_seed_manifest(path: str | Path, manifest: SeedManifest) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def seed_rows(manifest: SeedManifest) -> tuple[dict[str, Any], ...]:
    return tuple(seed.to_dict() for seed in manifest.seeds)


def write_seed_jsonl(path: str | Path, manifest: SeedManifest) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for seed in manifest.seeds:
            handle.write(json.dumps(seed.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
