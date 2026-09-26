"""Public-record contracts and the product/TS input firewall.

A record may be a quarantined source audit entry or an admitted scientific
sample.  Missing metadata is represented explicitly as ``unknown`` and never
silently converted to a neutral charge, singlet multiplicity, solvent label or
negative chemistry label.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping


UNKNOWN = "unknown"
QUARANTINED = "quarantine"
ADMITTED = frozenset({"train", "validation", "test", "diagnostic"})
FORBIDDEN_INPUT_KEYS = frozenset({
    "product", "mapped_product", "product_coordinates", "ts_geometry", "reference_ts",
    "reference_mode", "active_water_from_ts", "target_derived_solvent", "sealed_test_label",
})
ALLOWED_INPUT_KEYS = frozenset({
    "atomic_numbers", "symbols", "reactant_coordinates", "charge", "multiplicity",
    "microstate", "stereo", "solvent_membership", "declared_intent", "temperature",
    "buffer", "reactant_geometry_locator",
})


def canonical_hash(value: Any) -> str:
    """Hash JSON-compatible values with stable key and numeric serialization."""

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _unknown_or_string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, including explicit '{UNKNOWN}'")


@dataclass(frozen=True)
class PublicRecord:
    """One provenance-bearing public record or explicitly quarantined audit row."""

    source_dataset: str
    source_revision: str
    record_id: str
    parent_reaction_id: str
    split_group: str
    geometry_locator: str
    geometry_hash: str
    geometry_origin: str
    reference_protocol_id: str
    units: Mapping[str, str]
    available_label_mask: Mapping[str, bool]
    charge_spin_evidence: str
    license_record: str
    pretraining_overlap_audit: str
    record_status: str
    admission: str = QUARANTINED
    input_data: Mapping[str, Any] = field(default_factory=dict)
    label_data: Mapping[str, Any] = field(default_factory=dict)
    claim_limit: str = ""

    def __post_init__(self) -> None:
        for name in ("source_dataset", "source_revision", "record_id", "parent_reaction_id", "split_group", "geometry_locator", "geometry_hash", "geometry_origin", "reference_protocol_id", "charge_spin_evidence", "license_record", "pretraining_overlap_audit", "record_status", "admission"):
            _unknown_or_string(getattr(self, name), name)
        units = _mapping(self.units, "units")
        if not units or any(not isinstance(k, str) or not isinstance(v, str) for k, v in units.items()):
            raise ValueError("units must map names to explicit string units")
        mask = _mapping(self.available_label_mask, "available_label_mask")
        if any(not isinstance(k, str) or type(v) is not bool for k, v in mask.items()):
            raise ValueError("available_label_mask must map names to booleans")
        inputs = _mapping(self.input_data, "input_data")
        forbidden = FORBIDDEN_INPUT_KEYS.intersection(inputs)
        if forbidden:
            raise ValueError(f"target-derived fields cannot enter reactant input: {sorted(forbidden)}")
        unknown_keys = set(inputs) - ALLOWED_INPUT_KEYS
        if unknown_keys:
            raise ValueError(f"unsupported input fields require an explicit contract: {sorted(unknown_keys)}")
        _mapping(self.label_data, "label_data")
        if self.admission in ADMITTED:
            self.validate_admission()
        elif self.admission != QUARANTINED:
            raise ValueError(f"unsupported admission state: {self.admission}")

    def validate_admission(self) -> None:
        """Require enough real identity and state evidence before training use."""

        required = {
            "parent_reaction_id": self.parent_reaction_id,
            "split_group": self.split_group,
            "geometry_locator": self.geometry_locator,
            "geometry_hash": self.geometry_hash,
            "charge_spin_evidence": self.charge_spin_evidence,
        }
        missing = [name for name, value in required.items() if value in {UNKNOWN, "unavailable", "ambiguous"}]
        if missing:
            raise ValueError(f"admitted records cannot have unresolved fields: {missing}")
        if not self.input_data:
            raise ValueError("admitted records require reactant input data")
        if self.input_data.get("charge", UNKNOWN) == UNKNOWN or self.input_data.get("multiplicity", UNKNOWN) == UNKNOWN:
            raise ValueError("admitted records cannot guess charge or multiplicity")
        if self.pretraining_overlap_audit not in {"pass", "not_applicable"}:
            raise ValueError("admitted records require a passing overlap audit")

    def reactant_input(self) -> dict[str, Any]:
        """Return only deployment-visible fields; labels never affect this view."""

        return json.loads(json.dumps(self.input_data, ensure_ascii=False, sort_keys=True))

    def input_fingerprint(self) -> str:
        return canonical_hash(self.reactant_input())

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_dataset": self.source_dataset,
            "source_revision": self.source_revision,
            "record_id": self.record_id,
            "parent_reaction_id": self.parent_reaction_id,
            "split_group": self.split_group,
            "geometry_locator": self.geometry_locator,
            "geometry_hash": self.geometry_hash,
            "geometry_origin": self.geometry_origin,
            "reference_protocol_id": self.reference_protocol_id,
            "units": dict(self.units),
            "available_label_mask": dict(self.available_label_mask),
            "charge_spin_evidence": self.charge_spin_evidence,
            "license_record": self.license_record,
            "pretraining_overlap_audit": self.pretraining_overlap_audit,
            "record_status": self.record_status,
            "admission": self.admission,
            "input_data": dict(self.input_data),
            "label_data": dict(self.label_data),
            "claim_limit": self.claim_limit,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PublicRecord":
        return cls(**dict(payload))


def load_jsonl(path: str) -> list[PublicRecord]:
    """Load a JSONL manifest; blank lines are ignored, malformed rows fail closed."""

    records: list[PublicRecord] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(PublicRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid public record at line {line_number}: {exc}") from exc
    return records


def write_jsonl(path: str, records: list[PublicRecord]) -> None:
    """Write canonical machine-readable records without introducing defaults."""

    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
