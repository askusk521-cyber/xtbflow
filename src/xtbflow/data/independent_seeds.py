"""Frozen independent-reactant seed contracts for product-free discovery."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import json
import re
from typing import Any, Iterable, Mapping

from .records import canonical_hash


INDEPENDENT_SEED_SCHEMA = "xtbflow-independent-reactant-seed/v1"
SEED_SPLIT_ROLES = frozenset(
    {
        "development_train",
        "development_validation",
        "development_test",
        "confirmatory",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ATOM_MAP = re.compile(r":(?:0|[1-9][0-9]*)\]")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _strict_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be a strict integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class IndependentReactantSeed:
    """A reactant microstate frozen before downstream search."""

    seed_id: str
    parent_system_id: str
    family_id: str
    mapped_explicit_h_smiles: str
    geometry_path: str
    geometry_sha256: str
    charge: int
    multiplicity: int
    charge_spin_evidence: str
    microstate_description: str
    stereochemistry_description: str
    solvent_snapshot_id: str
    solvent_selection_origin: str
    reactant_generation_protocol: str
    reactant_generation_protocol_locator: str
    reactant_generation_protocol_sha256: str
    provenance: str
    approved_split_group: str
    approved_split_role: str
    frozen_at: str
    license_record: str
    coordinate_unit: str = "angstrom"
    input_origin: str = "independent_reactant"
    allowed_intent: str | None = None
    buffer_species_explicit: tuple[str, ...] = field(default_factory=tuple)
    temperature_k: float | None = None
    ph: float | None = None
    schema_version: str = INDEPENDENT_SEED_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "seed_id",
            "parent_system_id",
            "family_id",
            "mapped_explicit_h_smiles",
            "geometry_path",
            "charge_spin_evidence",
            "microstate_description",
            "stereochemistry_description",
            "solvent_snapshot_id",
            "solvent_selection_origin",
            "reactant_generation_protocol",
            "reactant_generation_protocol_locator",
            "provenance",
            "approved_split_group",
            "approved_split_role",
            "frozen_at",
            "license_record",
            "coordinate_unit",
            "input_origin",
            "schema_version",
        ):
            _text(getattr(self, name), name)
        if self.schema_version != INDEPENDENT_SEED_SCHEMA:
            raise ValueError(f"unsupported seed schema: {self.schema_version}")
        if self.approved_split_role not in SEED_SPLIT_ROLES:
            raise ValueError(
                f"unsupported seed split role: {self.approved_split_role}"
            )
        _hash(self.geometry_sha256, "geometry_sha256")
        _hash(
            self.reactant_generation_protocol_sha256,
            "reactant_generation_protocol_sha256",
        )
        if not _ATOM_MAP.search(self.mapped_explicit_h_smiles):
            raise ValueError(
                "mapped_explicit_h_smiles must contain explicit atom-map indices"
            )
        _strict_int(self.charge, "charge")
        _strict_int(self.multiplicity, "multiplicity", minimum=1)
        if self.coordinate_unit != "angstrom":
            raise ValueError("independent seed coordinates must be in angstrom")
        try:
            frozen = datetime.fromisoformat(self.frozen_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("frozen_at must be an ISO-8601 timestamp") from exc
        if frozen.tzinfo is None:
            raise ValueError("frozen_at must include an explicit timezone")
        if self.input_origin != "independent_reactant":
            raise ValueError("independent seed input_origin must be independently generated")
        if self.solvent_selection_origin not in {
            "none",
            "independent_reactant",
            "declared_environment",
        }:
            raise ValueError("solvent selection origin is not deployment-safe")
        if self.allowed_intent is not None and (
            not isinstance(self.allowed_intent, str) or not self.allowed_intent.strip()
        ):
            raise ValueError("allowed_intent must be null or a non-empty string")
        if isinstance(self.buffer_species_explicit, (str, bytes)):
            raise ValueError("buffer_species_explicit must be a sequence of strings")
        buffers = tuple(self.buffer_species_explicit)
        if any(not isinstance(value, str) or not value.strip() for value in buffers):
            raise ValueError("buffer species must be non-empty strings")
        object.__setattr__(self, "buffer_species_explicit", buffers)
        if self.temperature_k is not None:
            if (
                isinstance(self.temperature_k, bool)
                or not isinstance(self.temperature_k, (int, float))
                or self.temperature_k <= 0
            ):
                raise ValueError("temperature_k must be positive")
            object.__setattr__(self, "temperature_k", float(self.temperature_k))
        if self.ph is not None:
            if isinstance(self.ph, bool) or not isinstance(self.ph, (int, float)):
                raise ValueError("ph must be numeric or null")
            object.__setattr__(self, "ph", float(self.ph))

    def input_fingerprint(self) -> str:
        """Hash the exact reactant input, excluding IDs and generation metadata."""

        return canonical_hash(
            {
                "mapped_explicit_h_smiles": self.mapped_explicit_h_smiles,
                "geometry_sha256": self.geometry_sha256,
                "coordinate_unit": self.coordinate_unit,
                "charge": self.charge,
                "multiplicity": self.multiplicity,
                "allowed_intent": self.allowed_intent,
                "buffer_species_explicit": self.buffer_species_explicit,
                "temperature_k": self.temperature_k,
                "ph": self.ph,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed_id": self.seed_id,
            "parent_system_id": self.parent_system_id,
            "family_id": self.family_id,
            "mapped_explicit_h_smiles": self.mapped_explicit_h_smiles,
            "geometry_path": self.geometry_path,
            "geometry_sha256": self.geometry_sha256,
            "coordinate_unit": self.coordinate_unit,
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "charge_spin_evidence": self.charge_spin_evidence,
            "microstate_description": self.microstate_description,
            "stereochemistry_description": self.stereochemistry_description,
            "solvent_snapshot_id": self.solvent_snapshot_id,
            "solvent_selection_origin": self.solvent_selection_origin,
            "reactant_generation_protocol": self.reactant_generation_protocol,
            "reactant_generation_protocol_locator": (
                self.reactant_generation_protocol_locator
            ),
            "reactant_generation_protocol_sha256": (
                self.reactant_generation_protocol_sha256
            ),
            "provenance": self.provenance,
            "approved_split_group": self.approved_split_group,
            "approved_split_role": self.approved_split_role,
            "frozen_at": self.frozen_at,
            "license_record": self.license_record,
            "input_origin": self.input_origin,
            "allowed_intent": self.allowed_intent,
            "buffer_species_explicit": list(self.buffer_species_explicit),
            "temperature_k": self.temperature_k,
            "ph": self.ph,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IndependentReactantSeed":
        if not isinstance(payload, Mapping):
            raise ValueError("seed payload must be a mapping")
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unsupported seed fields: {sorted(unknown)}")
        return cls(**dict(payload))


def load_seed_jsonl(path: str) -> list[IndependentReactantSeed]:
    rows: list[IndependentReactantSeed] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = IndependentReactantSeed.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid independent seed at line {line_number}: {exc}"
                ) from exc
            if row.seed_id in seen:
                raise ValueError(
                    f"duplicate seed_id at line {line_number}: {row.seed_id}"
                )
            seen.add(row.seed_id)
            rows.append(row)
    return sorted(rows, key=lambda row: row.seed_id)


def write_seed_jsonl(
    path: str, records: Iterable[IndependentReactantSeed]
) -> None:
    rows = list(records)
    if len({row.seed_id for row in rows}) != len(rows):
        raise ValueError("seed_id values must be unique")
    rows.sort(key=lambda row: row.seed_id)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            payload = json.dumps(
                row.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write(payload + "\n")


class SeedLeakageError(ValueError):
    """Raised when frozen reactant seeds cross predeclared split roles."""


def audit_seed_leakage(
    records: Iterable[IndependentReactantSeed],
    *,
    strict_family_holdout: bool = True,
) -> dict[str, Any]:
    rows = list(records)
    dimensions: dict[str, dict[str, set[str]]] = {
        "parent_system_id": defaultdict(set),
        "approved_split_group": defaultdict(set),
        "input_fingerprint": defaultdict(set),
    }
    if strict_family_holdout:
        dimensions["family_id"] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    for row in rows:
        split = row.approved_split_role
        split_counts[split] += 1
        dimensions["parent_system_id"][row.parent_system_id].add(split)
        dimensions["approved_split_group"][row.approved_split_group].add(split)
        dimensions["input_fingerprint"][row.input_fingerprint()].add(split)
        if strict_family_holdout:
            dimensions["family_id"][row.family_id].add(split)
    leaking = {
        dimension: {
            identity: sorted(splits)
            for identity, splits in values.items()
            if len(splits) > 1
        }
        for dimension, values in dimensions.items()
    }
    leaking = {key: value for key, value in leaking.items() if value}
    if leaking:
        raise SeedLeakageError(
            json.dumps(leaking, ensure_ascii=False, sort_keys=True)
        )
    return {
        "record_count": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "strict_family_holdout": strict_family_holdout,
        "leakage_dimensions_checked": sorted(dimensions),
    }
