"""Track-B reaction/event/TS admission and leakage contracts.

Track B is deliberately separate from ordinary energy/force calibration.  A
record is useful for joint event--geometry work only when the reactant-visible
view, product supervision, event label, TS geometry, electronic state, atom
correspondence, grouping and provenance can all be reconstructed without
target-derived inputs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
import json
import re
from typing import Any, Iterable, Mapping, Sequence

from .records import canonical_hash


TRACK_B_SCHEMA = "xtbflow-track-b-reaction-ts/v1"
TRACK_B_ADMISSIONS = frozenset(
    {"quarantine", "development_train", "development_validation", "development_test", "confirmatory"}
)
TRACK_B_SPLITS = {
    "development_train": "train",
    "development_validation": "validation",
    "development_test": "test",
    "confirmatory": "confirmatory",
}
SAFE_REACTANT_ORIGINS = frozenset({"independent_reactant", "declared_reactant_endpoint"})
SAFE_SOLVENT_ORIGINS = frozenset({"none", "independent_reactant", "declared_environment"})
VALID_EVIDENCE = frozenset({"observed", "derived_under_contract"})
UNKNOWN_VALUES = frozenset({"", "unknown", "unavailable", "ambiguous", "missing", "not_provided", "unverified"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_REACTANT_KEYS = frozenset(
    {
        "graph_locator",
        "graph_sha256",
        "coordinates_locator",
        "coordinates_sha256",
        "coordinate_unit",
        "input_origin",
        "solvent_selection_origin",
        "charge",
        "multiplicity",
        "charge_spin_evidence",
        "microstate_id",
        "geometry_generation_protocol",
        "geometry_generation_protocol_locator",
        "geometry_generation_protocol_sha256",
        "declared_intent",
    }
)
_EVENT_KEYS = frozenset({"locator", "sha256", "representation", "evidence"})
_PRODUCT_KEYS = frozenset(
    {
        "graph_locator",
        "graph_sha256",
        "graph_evidence",
        "coordinates_locator",
        "coordinates_sha256",
        "coordinate_unit",
        "coordinate_evidence",
        "atom_order_matches_input",
    }
)
_TS_KEYS = frozenset({"locator", "sha256", "coordinate_unit", "evidence", "atom_order_matches_input"})
_PROTOCOL_KEYS = frozenset(
    {
        "protocol_id",
        "protocol_locator",
        "protocol_sha256",
        "method",
        "software",
        "version",
        "environment",
    }
)
_INTENT_KEYS = frozenset(
    {
        "permission_mode",
        "reactive_map_ids",
        "net_bond_edits",
        "provenance",
        "uses_reference_labels",
    }
)
_INTENT_MODES = frozenset({"none", "declared_center", "declared_net_intent"})


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    result = dict(value)
    try:
        canonical_hash(result)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON-compatible") from exc
    return result


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value.strip().lower() not in UNKNOWN_VALUES


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value))


def _normalise_reasons(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("quarantine_reasons must be an iterable of strings")
    reasons: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("quarantine reasons must be non-empty strings")
        reasons.add(value.strip())
    return tuple(sorted(reasons))


def _sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


@dataclass(frozen=True)
class TrackBRecord:
    """One reaction/event/TS record with a deployment-safe reactant view."""

    record_id: str
    source_dataset: str
    source_revision: str
    source_record_id: str
    source_asset_sha256: str
    source_record_sha256: str
    license_record: str
    parent_reaction_id: str
    family_id: str
    split_group: str
    atoms: Mapping[str, Any]
    reactant: Mapping[str, Any]
    event_label: Mapping[str, Any]
    product_label: Mapping[str, Any]
    ts_geometry: Mapping[str, Any]
    reference_protocol: Mapping[str, Any]
    pretraining_overlap_audit: str
    admission: str = "quarantine"
    intended_use: str = "event_geometry_supervision"
    claim_limit: str = "development evidence only"
    quarantine_reasons: tuple[str, ...] = field(default_factory=tuple)
    schema_version: str = TRACK_B_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "record_id",
            "source_dataset",
            "source_revision",
            "source_record_id",
            "source_asset_sha256",
            "source_record_sha256",
            "license_record",
            "parent_reaction_id",
            "family_id",
            "split_group",
            "pretraining_overlap_audit",
            "admission",
            "intended_use",
            "schema_version",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.schema_version != TRACK_B_SCHEMA:
            raise ValueError(f"unsupported Track-B schema: {self.schema_version}")
        if self.admission not in TRACK_B_ADMISSIONS:
            raise ValueError(f"unsupported Track-B admission: {self.admission}")
        if not isinstance(self.claim_limit, str):
            raise ValueError("claim_limit must be a string")
        for name in (
            "atoms",
            "reactant",
            "event_label",
            "product_label",
            "ts_geometry",
            "reference_protocol",
        ):
            object.__setattr__(self, name, _mapping(getattr(self, name), name))
        reasons = _normalise_reasons(self.quarantine_reasons)
        object.__setattr__(self, "quarantine_reasons", reasons)
        if self.admission != "quarantine":
            policy_reasons = self.policy_reasons()
            if policy_reasons:
                raise ValueError(f"admitted Track-B record violates contract: {list(policy_reasons)}")
            if reasons:
                raise ValueError("admitted Track-B records cannot carry quarantine reasons")

    def policy_reasons(self) -> tuple[str, ...]:
        """Return deterministic admission failures without guessing metadata."""

        reasons: set[str] = set()
        for name in (
            "source_dataset",
            "source_revision",
            "source_record_id",
            "license_record",
            "parent_reaction_id",
            "family_id",
            "split_group",
        ):
            if not _nonempty(getattr(self, name)):
                reasons.add(f"{name}_unresolved")
        if not _sha256(self.source_asset_sha256):
            reasons.add("source_asset_sha256_invalid")
        if not _sha256(self.source_record_sha256):
            reasons.add("source_record_sha256_invalid")
        if self.intended_use != "event_geometry_supervision":
            reasons.add("intended_use_not_event_geometry_supervision")
        if self.pretraining_overlap_audit not in {"pass", "not_applicable"}:
            reasons.add("pretraining_overlap_audit_unresolved")

        atomic_numbers = self.atoms.get("atomic_numbers")
        map_ids = self.atoms.get("map_ids")
        atom_count = 0
        if not _sequence(atomic_numbers) or not atomic_numbers:
            reasons.add("atomic_numbers_missing_or_malformed")
        elif any(type(value) is not int or value < 1 for value in atomic_numbers):
            reasons.add("atomic_numbers_invalid")
        else:
            atom_count = len(atomic_numbers)
        if not _sequence(map_ids) or not map_ids:
            reasons.add("atom_mapping_missing_or_malformed")
        elif any(type(value) is not int or value < 0 for value in map_ids):
            reasons.add("atom_mapping_invalid")
        elif len(set(map_ids)) != len(map_ids):
            reasons.add("atom_mapping_not_unique")
        elif atom_count and len(map_ids) != atom_count:
            reasons.add("atom_mapping_length_mismatch")

        extra_reactant = set(self.reactant) - _REACTANT_KEYS
        if extra_reactant:
            reasons.add("reactant_contract_has_unsupported_fields")
        for key in (
            "graph_locator",
            "coordinates_locator",
            "charge_spin_evidence",
            "microstate_id",
            "geometry_generation_protocol",
            "geometry_generation_protocol_locator",
        ):
            if not _nonempty(self.reactant.get(key)):
                reasons.add(f"reactant_{key}_unresolved")
        for key in (
            "graph_sha256",
            "coordinates_sha256",
            "geometry_generation_protocol_sha256",
        ):
            if not _sha256(self.reactant.get(key)):
                reasons.add(f"reactant_{key}_invalid")
        if self.reactant.get("coordinate_unit") != "angstrom":
            reasons.add("reactant_coordinate_unit_not_angstrom")
        input_origin = self.reactant.get("input_origin")
        if input_origin not in SAFE_REACTANT_ORIGINS:
            reasons.add("reactant_input_target_derived_or_unresolved")
        if self.admission == "confirmatory" and input_origin != "independent_reactant":
            reasons.add("confirmatory_input_not_independent")
        if self.reactant.get("solvent_selection_origin") not in SAFE_SOLVENT_ORIGINS:
            reasons.add("solvent_selection_target_derived_or_unresolved")

        declared_intent = self.reactant.get("declared_intent")
        if declared_intent is not None:
            if not isinstance(declared_intent, Mapping):
                reasons.add("declared_intent_malformed")
            else:
                intent = dict(declared_intent)
                try:
                    canonical_hash(intent)
                except (TypeError, ValueError):
                    reasons.add("declared_intent_not_json_compatible")
                if set(intent) - _INTENT_KEYS:
                    reasons.add("declared_intent_has_unsupported_fields")
                mode = intent.get("permission_mode")
                if mode not in _INTENT_MODES:
                    reasons.add("declared_intent_permission_mode_invalid")
                if intent.get("uses_reference_labels") is not False:
                    reasons.add("declared_intent_uses_reference_labels")
                if not _nonempty(intent.get("provenance")):
                    reasons.add("declared_intent_provenance_unresolved")
                allowed_maps = (
                    set(map_ids)
                    if _sequence(map_ids)
                    and all(type(value) is int and value >= 0 for value in map_ids)
                    else set()
                )
                reactive_map_ids = intent.get("reactive_map_ids", [])
                if not _sequence(reactive_map_ids) or any(
                    type(value) is not int for value in reactive_map_ids
                ):
                    reasons.add("declared_intent_reactive_map_ids_invalid")
                    reactive_map_ids = []
                elif len(set(reactive_map_ids)) != len(reactive_map_ids):
                    reasons.add("declared_intent_reactive_map_ids_not_unique")
                elif allowed_maps and any(
                    value not in allowed_maps for value in reactive_map_ids
                ):
                    reasons.add("declared_intent_reactive_map_id_out_of_range")
                net_bond_edits = intent.get("net_bond_edits", [])
                edits_valid = _sequence(net_bond_edits)
                if edits_valid:
                    for edit in net_bond_edits:
                        if (
                            not _sequence(edit)
                            or len(edit) != 3
                            or any(type(value) is not int for value in edit)
                            or edit[0] == edit[1]
                            or edit[2] == 0
                            or (allowed_maps and (edit[0] not in allowed_maps or edit[1] not in allowed_maps))
                        ):
                            edits_valid = False
                            break
                if not edits_valid:
                    reasons.add("declared_intent_net_bond_edits_invalid")
                    net_bond_edits = []
                if mode == "none" and (reactive_map_ids or net_bond_edits):
                    reasons.add("declared_intent_none_contains_targets")
                if mode == "declared_center" and (
                    not reactive_map_ids or net_bond_edits
                ):
                    reasons.add("declared_center_contract_invalid")
                if mode == "declared_net_intent" and not net_bond_edits:
                    reasons.add("declared_net_intent_missing_edits")

        charge = self.reactant.get("charge")
        multiplicity = self.reactant.get("multiplicity")
        if type(charge) is not int:
            reasons.add("charge_missing_or_noninteger")
        if type(multiplicity) is not int or multiplicity < 1:
            reasons.add("multiplicity_missing_or_invalid")
        if atom_count and type(charge) is int and type(multiplicity) is int and multiplicity > 0:
            electron_count = sum(atomic_numbers) - charge
            if electron_count < 1:
                reasons.add("electron_count_invalid")
            elif (electron_count - (multiplicity - 1)) % 2:
                reasons.add("electron_spin_parity_mismatch")

        extra_event = set(self.event_label) - _EVENT_KEYS
        if extra_event:
            reasons.add("event_label_has_unsupported_fields")
        for key in ("locator", "representation"):
            if not _nonempty(self.event_label.get(key)):
                reasons.add(f"event_label_{key}_unresolved")
        if not _sha256(self.event_label.get("sha256")):
            reasons.add("event_label_sha256_invalid")
        if self.event_label.get("evidence") not in VALID_EVIDENCE:
            reasons.add("event_label_evidence_insufficient")

        extra_product = set(self.product_label) - _PRODUCT_KEYS
        if extra_product:
            reasons.add("product_label_has_unsupported_fields")
        if not _nonempty(self.product_label.get("graph_locator")):
            reasons.add("product_graph_locator_unresolved")
        if not _sha256(self.product_label.get("graph_sha256")):
            reasons.add("product_graph_sha256_invalid")
        if self.product_label.get("graph_evidence") not in VALID_EVIDENCE:
            reasons.add("product_graph_evidence_insufficient")

        coordinate_evidence = self.product_label.get("coordinate_evidence")
        coordinate_values = (
            self.product_label.get("coordinates_locator"),
            self.product_label.get("coordinates_sha256"),
            self.product_label.get("coordinate_unit"),
        )
        atom_order_matches = self.product_label.get("atom_order_matches_input")
        if coordinate_evidence == "unavailable":
            if any(value is not None for value in coordinate_values):
                reasons.add("product_coordinates_unavailable_but_populated")
            if atom_order_matches is not None:
                reasons.add("product_atom_order_unavailable_but_populated")
        elif coordinate_evidence in VALID_EVIDENCE:
            if not _nonempty(self.product_label.get("coordinates_locator")):
                reasons.add("product_coordinates_locator_unresolved")
            if not _sha256(self.product_label.get("coordinates_sha256")):
                reasons.add("product_coordinates_sha256_invalid")
            if self.product_label.get("coordinate_unit") != "angstrom":
                reasons.add("product_coordinate_unit_not_angstrom")
            if atom_order_matches is not True:
                reasons.add("product_atom_correspondence_unresolved")
        else:
            reasons.add("product_coordinate_evidence_unresolved")

        extra_ts = set(self.ts_geometry) - _TS_KEYS
        if extra_ts:
            reasons.add("ts_geometry_has_unsupported_fields")
        if not _nonempty(self.ts_geometry.get("locator")):
            reasons.add("ts_geometry_locator_unresolved")
        if not _sha256(self.ts_geometry.get("sha256")):
            reasons.add("ts_geometry_sha256_invalid")
        if self.ts_geometry.get("coordinate_unit") != "angstrom":
            reasons.add("ts_geometry_unit_not_angstrom")
        if self.ts_geometry.get("evidence") not in VALID_EVIDENCE:
            reasons.add("ts_geometry_evidence_insufficient")
        if self.ts_geometry.get("atom_order_matches_input") is not True:
            reasons.add("ts_atom_correspondence_unresolved")

        extra_protocol = set(self.reference_protocol) - _PROTOCOL_KEYS
        if extra_protocol:
            reasons.add("reference_protocol_has_unsupported_fields")
        for key in (
            "protocol_id",
            "protocol_locator",
            "method",
            "software",
            "version",
            "environment",
        ):
            if not _nonempty(self.reference_protocol.get(key)):
                reasons.add(f"reference_{key}_unresolved")
        if not _sha256(self.reference_protocol.get("protocol_sha256")):
            reasons.add("reference_protocol_sha256_invalid")
        return tuple(sorted(reasons))

    def reactant_view(self) -> dict[str, Any]:
        """Return the deployment-visible manifest view with no target metadata."""

        view = {
            "schema_version": self.schema_version,
            "atoms": {
                "atomic_numbers": self.atoms.get("atomic_numbers"),
                "map_ids": self.atoms.get("map_ids"),
            },
            "graph_locator": self.reactant.get("graph_locator"),
            "graph_sha256": self.reactant.get("graph_sha256"),
            "coordinates_locator": self.reactant.get("coordinates_locator"),
            "coordinates_sha256": self.reactant.get("coordinates_sha256"),
            "coordinate_unit": self.reactant.get("coordinate_unit"),
            "charge": self.reactant.get("charge"),
            "multiplicity": self.reactant.get("multiplicity"),
            "microstate_id": self.reactant.get("microstate_id"),
            "declared_intent": self.reactant.get("declared_intent"),
        }
        return json.loads(json.dumps(view, ensure_ascii=False, sort_keys=True))

    def input_fingerprint(self) -> str:
        """Hash model-visible input identity, excluding provenance metadata.

        ``declared_intent`` records a proposal/source-side explanation and may
        vary between records for the same reactant input. It must not split
        exact-input leakage detection. The microstate identity is part of the
        scientific input and is included explicitly.
        """

        return canonical_hash(
            {
                "schema_version": self.schema_version,
                "atomic_numbers": self.atoms.get("atomic_numbers"),
                "map_ids": self.atoms.get("map_ids"),
                "graph_sha256": self.reactant.get("graph_sha256"),
                "coordinates_sha256": self.reactant.get("coordinates_sha256"),
                "coordinate_unit": self.reactant.get("coordinate_unit"),
                "charge": self.reactant.get("charge"),
                "multiplicity": self.reactant.get("multiplicity"),
                "microstate_id": self.reactant.get("microstate_id"),
            }
        )

    def quarantined(self, reasons: Iterable[str]) -> "TrackBRecord":
        new_reasons = _normalise_reasons(reasons)
        merged = _normalise_reasons((*self.quarantine_reasons, *new_reasons))
        return replace(self, admission="quarantine", quarantine_reasons=merged)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "source_dataset": self.source_dataset,
            "source_revision": self.source_revision,
            "source_record_id": self.source_record_id,
            "source_asset_sha256": self.source_asset_sha256,
            "source_record_sha256": self.source_record_sha256,
            "license_record": self.license_record,
            "parent_reaction_id": self.parent_reaction_id,
            "family_id": self.family_id,
            "split_group": self.split_group,
            "atoms": dict(self.atoms),
            "reactant": dict(self.reactant),
            "event_label": dict(self.event_label),
            "product_label": dict(self.product_label),
            "ts_geometry": dict(self.ts_geometry),
            "reference_protocol": dict(self.reference_protocol),
            "pretraining_overlap_audit": self.pretraining_overlap_audit,
            "admission": self.admission,
            "intended_use": self.intended_use,
            "claim_limit": self.claim_limit,
            "quarantine_reasons": list(self.quarantine_reasons),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrackBRecord":
        if not isinstance(payload, Mapping):
            raise ValueError("Track-B payload must be a mapping")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unsupported Track-B fields: {sorted(unknown)}")
        return cls(**dict(payload))


@dataclass(frozen=True)
class TrackBFilterResult:
    accepted: tuple[TrackBRecord, ...]
    quarantined: tuple[TrackBRecord, ...]

    @property
    def records(self) -> tuple[TrackBRecord, ...]:
        return tuple(sorted((*self.accepted, *self.quarantined), key=lambda row: row.record_id))


def filter_track_b_records(records: Iterable[TrackBRecord]) -> TrackBFilterResult:
    """Keep every rejected row and attach stable machine-readable reasons."""

    rows = list(records)
    counts = Counter(row.record_id for row in rows)
    accepted: list[TrackBRecord] = []
    quarantined: list[TrackBRecord] = []
    for row in rows:
        reasons = set(row.policy_reasons())
        reasons.update(row.quarantine_reasons)
        if counts[row.record_id] > 1:
            reasons.add("duplicate_record_id")
        if row.admission == "quarantine" or reasons:
            quarantined.append(row.quarantined(reasons or {"not_admitted"}))
        else:
            accepted.append(row)
    accepted.sort(key=lambda row: row.record_id)
    quarantined.sort(key=lambda row: (row.record_id, canonical_hash(row.to_dict())))
    return TrackBFilterResult(tuple(accepted), tuple(quarantined))


class TrackBLeakageError(ValueError):
    """Raised when a parent, family, input, or source row crosses split roles."""


def audit_track_b_leakage(records: Iterable[TrackBRecord], *, strict_family_holdout: bool = True) -> dict[str, Any]:
    """Audit admitted records using parents, families and exact input identities."""

    rows = [row for row in records if row.admission != "quarantine"]
    dimensions: dict[str, dict[str, set[str]]] = {
        "parent_reaction_id": defaultdict(set),
        "split_group": defaultdict(set),
        "input_fingerprint": defaultdict(set),
        "source_record": defaultdict(set),
        "source_record_bytes": defaultdict(set),
    }
    if strict_family_holdout:
        dimensions["family_id"] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    for row in rows:
        split = TRACK_B_SPLITS[row.admission]
        split_counts[split] += 1
        dimensions["parent_reaction_id"][row.parent_reaction_id].add(split)
        dimensions["split_group"][row.split_group].add(split)
        dimensions["input_fingerprint"][row.input_fingerprint()].add(split)
        source_identity = (
            f"{row.source_dataset}:{row.source_revision}:{row.source_record_id}"
        )
        dimensions["source_record"][source_identity].add(split)
        dimensions["source_record_bytes"][row.source_record_sha256].add(split)
        if strict_family_holdout:
            dimensions["family_id"][row.family_id].add(split)
    leaking = {
        dimension: {identity: sorted(splits) for identity, splits in values.items() if len(splits) > 1}
        for dimension, values in dimensions.items()
    }
    leaking = {dimension: values for dimension, values in leaking.items() if values}
    if leaking:
        raise TrackBLeakageError(json.dumps(leaking, sort_keys=True, ensure_ascii=False))
    return {
        "record_count": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "strict_family_holdout": strict_family_holdout,
        "leakage_dimensions_checked": sorted(dimensions),
    }


def load_track_b_jsonl(path: str) -> list[TrackBRecord]:
    records: list[TrackBRecord] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = TrackBRecord.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid Track-B record at line {line_number}: {exc}"
                ) from exc
            if row.record_id in seen:
                raise ValueError(
                    f"duplicate Track-B record_id at line {line_number}: "
                    f"{row.record_id}"
                )
            seen.add(row.record_id)
            records.append(row)
    return sorted(records, key=lambda row: row.record_id)


def write_track_b_jsonl(path: str, records: Iterable[TrackBRecord]) -> None:
    rows = list(records)
    if len({row.record_id for row in rows}) != len(rows):
        raise ValueError("Track-B record_id values must be unique")
    rows.sort(key=lambda row: row.record_id)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
