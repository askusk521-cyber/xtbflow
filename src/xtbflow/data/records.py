"""Public-record contracts, quarantine policy, and deterministic JSONL I/O.

A record may be a quarantined source audit entry or an admitted scientific
sample. Missing metadata is represented explicitly as ``unknown`` and never
silently converted to a neutral charge, singlet multiplicity, solvent label or
negative chemistry label.  Policy filtering is deliberately separate from
record construction: a rejected row remains available as a quarantine row with
stable machine-readable reason codes.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence


UNKNOWN = "unknown"
QUARANTINED = "quarantine"
ADMITTED = frozenset({"train", "validation", "test", "diagnostic"})
PUBLIC_CANONICAL_UNITS = {
    "coordinates": "angstrom",
    "energy": "hartree",
    "forces": "hartree/angstrom",
}
CHNOS_ELEMENTS = frozenset({"C", "H", "N", "O", "S"})
FORBIDDEN_INPUT_KEYS = frozenset({
    "product", "mapped_product", "product_coordinates", "ts_geometry", "reference_ts",
    "reference_mode", "active_water_from_ts", "target_derived_solvent", "sealed_test_label",
})
ALLOWED_INPUT_KEYS = frozenset({
    "atomic_numbers", "symbols", "elements", "reactant_coordinates", "charge", "multiplicity",
    "microstate", "stereo", "solvent_membership", "declared_intent", "temperature",
    "buffer", "reactant_geometry_locator",
})

# Atomic numbers are used only for an electron-count parity check.  They do not
# fill in missing source metadata or infer a charge/multiplicity.
_ELEMENT_ATOMIC_NUMBERS = {
    symbol: number
    for number, symbol in enumerate(
        (
            "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
        )
    )
    if symbol
}


_MISSING = object()


def canonical_hash(value: Any) -> str:
    """Hash JSON-compatible values with stable key and numeric serialization."""

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    result = dict(value)
    try:
        canonical_hash(result)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON-compatible") from exc
    return result


def _unknown_or_string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, including explicit '{UNKNOWN}'")


def _strict_int(value: Any, name: str, *, positive: bool = False, allow_unknown: bool = False) -> None:
    """Validate integer chemistry state without accepting Python booleans.

    Quarantined source-audit rows may carry an explicit unresolved sentinel,
    but admitted rows and all numeric values must use strict Python integers.
    """

    if allow_unknown and isinstance(value, str) and value.strip().lower() in {
        UNKNOWN, "unavailable", "ambiguous", "missing", "not_provided", "unverified",
    }:
        return
    if type(value) is not int:
        raise ValueError(f"{name} must be a strict integer (bool is not accepted)")
    if positive and value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _unresolved(value: Any) -> bool:
    return not isinstance(value, str) or not value.strip() or value.strip().lower() in {
        UNKNOWN, "unavailable", "ambiguous", "missing", "not_provided", "unverified",
    }


def _normalise_reason_codes(reasons: Iterable[str]) -> tuple[str, ...]:
    if reasons is None or isinstance(reasons, (str, bytes)):
        raise ValueError("quarantine reasons must be an iterable of non-empty strings")
    try:
        iterator = iter(reasons)
    except TypeError as exc:
        raise ValueError("quarantine reasons must be an iterable of non-empty strings") from exc
    values: set[str] = set()
    for reason in iterator:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("quarantine reasons must be non-empty strings")
        values.add(reason.strip())
    return tuple(sorted(values))


def _symbols_from_input(input_data: Mapping[str, Any]) -> tuple[tuple[str, ...] | None, str | None]:
    """Return explicit element symbols, or a stable reason for their absence.

    Atomic numbers are accepted as an equivalent explicit representation for
    the parity check.  No symbol is guessed when either representation is
    malformed or unknown.
    """

    symbols_value = input_data.get("symbols", _MISSING)
    elements_value = input_data.get("elements", _MISSING)
    if symbols_value is not _MISSING and elements_value is not _MISSING and symbols_value != elements_value:
        return None, "conflicting_element_fields"
    raw = symbols_value if symbols_value is not _MISSING else elements_value
    if raw is _MISSING:
        raw_numbers = input_data.get("atomic_numbers", _MISSING)
        if raw_numbers is _MISSING:
            return None, "elements_missing"
        raw = raw_numbers
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            return None, "elements_malformed"
        symbols: list[str] = []
        for number in raw:
            if type(number) is not int or number < 1:
                return None, "elements_malformed"
            symbol = next((candidate for candidate, atomic_number in _ELEMENT_ATOMIC_NUMBERS.items() if atomic_number == number), None)
            if symbol is None:
                return None, "elements_malformed"
            symbols.append(symbol)
        return tuple(symbols), None
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or not raw:
        return None, "elements_malformed"
    symbols = tuple(raw)
    if any(not isinstance(symbol, str) or not symbol.strip() for symbol in symbols):
        return None, "elements_malformed"
    if any(symbol not in _ELEMENT_ATOMIC_NUMBERS for symbol in symbols):
        return None, "element_symbol_invalid"
    return symbols, None


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
    quarantine_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "source_dataset", "source_revision", "record_id", "parent_reaction_id", "split_group",
            "geometry_locator", "geometry_hash", "geometry_origin", "reference_protocol_id",
            "charge_spin_evidence", "license_record", "pretraining_overlap_audit", "record_status",
            "admission",
        ):
            _unknown_or_string(getattr(self, name), name)
        if not isinstance(self.claim_limit, str):
            raise ValueError("claim_limit must be a string")
        units = _mapping(self.units, "units")
        if not units or any(not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip() for k, v in units.items()):
            raise ValueError("units must map names to non-empty explicit string units")
        mask = _mapping(self.available_label_mask, "available_label_mask")
        if any(not isinstance(k, str) or not k.strip() or type(v) is not bool for k, v in mask.items()):
            raise ValueError("available_label_mask must map names to booleans")
        inputs = _mapping(self.input_data, "input_data")
        forbidden = FORBIDDEN_INPUT_KEYS.intersection(inputs)
        if forbidden:
            raise ValueError(f"target-derived fields cannot enter reactant input: {sorted(forbidden)}")
        unknown_keys = set(inputs) - ALLOWED_INPUT_KEYS
        if unknown_keys:
            raise ValueError(f"unsupported input fields require an explicit contract: {sorted(unknown_keys)}")
        # Validate any supplied state even for quarantine rows.  A quarantine
        # row can be incomplete, but it must not contain a misleading bool or
        # float in place of an explicit electronic state.
        if "charge" in inputs:
            _strict_int(inputs["charge"], "charge", allow_unknown=self.admission == QUARANTINED)
        if "multiplicity" in inputs:
            _strict_int(inputs["multiplicity"], "multiplicity", positive=True, allow_unknown=self.admission == QUARANTINED)
        _mapping(self.label_data, "label_data")
        reasons = _normalise_reason_codes(self.quarantine_reasons)
        object.__setattr__(self, "quarantine_reasons", reasons)
        if self.admission == QUARANTINED:
            return
        if self.admission not in ADMITTED:
            raise ValueError(f"unsupported admission state: {self.admission}")
        if reasons:
            raise ValueError("admitted records cannot carry quarantine reasons")
        self.validate_admission()

    def validate_admission(self) -> None:
        """Require enough identity, state, and unit evidence before training use."""

        required = {
            "parent_reaction_id": self.parent_reaction_id,
            "split_group": self.split_group,
            "geometry_locator": self.geometry_locator,
            "geometry_hash": self.geometry_hash,
            "charge_spin_evidence": self.charge_spin_evidence,
            "source_dataset": self.source_dataset,
            "source_revision": self.source_revision,
            "geometry_origin": self.geometry_origin,
            "reference_protocol_id": self.reference_protocol_id,
            "license_record": self.license_record,
            "record_status": self.record_status,
        }
        missing = [name for name, value in required.items() if _unresolved(value)]
        if missing:
            raise ValueError(f"admitted records cannot have unresolved fields: {missing}")
        if dict(self.units) != PUBLIC_CANONICAL_UNITS:
            raise ValueError("admitted records require canonical units")
        if not self.input_data:
            raise ValueError("admitted records require reactant input data")
        charge = self.input_data.get("charge", _MISSING)
        multiplicity = self.input_data.get("multiplicity", _MISSING)
        if charge is _MISSING or multiplicity is _MISSING:
            raise ValueError("admitted records require explicit charge and multiplicity")
        _strict_int(charge, "charge")
        _strict_int(multiplicity, "multiplicity", positive=True)
        if self.pretraining_overlap_audit not in {"pass", "not_applicable"}:
            raise ValueError("admitted records require a passing overlap audit")

    def reactant_input(self) -> dict[str, Any]:
        """Return only deployment-visible fields; labels never affect this view."""

        return json.loads(json.dumps(self.input_data, ensure_ascii=False, sort_keys=True))

    def input_fingerprint(self) -> str:
        return canonical_hash(self.reactant_input())

    def quarantined(self, reasons: Iterable[str]) -> "PublicRecord":
        """Return this row in quarantine without changing its historical fields."""

        merged = _normalise_reason_codes((*self.quarantine_reasons, *tuple(reasons)))
        return replace(self, admission=QUARANTINED, quarantine_reasons=merged)

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        """Compatibility alias for consumers that call them rejection reasons."""

        return self.quarantine_reasons

    @property
    def quarantine_reason_codes(self) -> tuple[str, ...]:
        return self.quarantine_reasons

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
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
        # Keep historical rows byte-compatible when no reason metadata exists;
        # newly quarantined rows carry an explicit machine-readable field.
        if self.quarantine_reasons:
            payload["quarantine_reasons"] = list(self.quarantine_reasons)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PublicRecord":
        if not isinstance(payload, Mapping):
            raise ValueError("public record payload must be a mapping")
        value = dict(payload)
        allowed = {
            "source_dataset", "source_revision", "record_id", "parent_reaction_id", "split_group",
            "geometry_locator", "geometry_hash", "geometry_origin", "reference_protocol_id", "units",
            "available_label_mask", "charge_spin_evidence", "license_record", "pretraining_overlap_audit",
            "record_status", "admission", "input_data", "label_data", "claim_limit", "quarantine_reasons",
            "rejection_reasons",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unsupported public record fields: {sorted(unknown)}")
        if "rejection_reasons" in value:
            if "quarantine_reasons" in value and value["quarantine_reasons"] != value["rejection_reasons"]:
                raise ValueError("conflicting quarantine and rejection reasons")
            value["quarantine_reasons"] = value.pop("rejection_reasons")
        return cls(**value)


@dataclass(frozen=True)
class RecordPolicy:
    """Reusable admission policy for public records.

    ``allowed_elements`` is optional so source-audit rows can be loaded without
    inventing chemistry.  Set it to ``CHNOS_ELEMENTS`` for the project's
    explicit CHNOS scope.  Every decision is represented by stable reason codes
    and the serialized policy itself has a deterministic SHA-256 digest.
    """

    allowed_elements: frozenset[str] | None = None
    canonical_units: Mapping[str, str] = field(default_factory=lambda: dict(PUBLIC_CANONICAL_UNITS))
    require_source_provenance: bool = True
    require_license: bool = True
    require_overlap_audit: bool = True
    require_charge_spin_evidence: bool = True
    require_explicit_state: bool = True
    require_spin_parity: bool = True
    policy_version: str = "record-policy-v1"

    def __post_init__(self) -> None:
        if not isinstance(self.policy_version, str) or not self.policy_version.strip():
            raise ValueError("policy_version must be a non-empty string")
        if self.allowed_elements is not None:
            if isinstance(self.allowed_elements, (str, bytes)):
                raise ValueError("allowed_elements must be a collection of element symbols")
            normalized = frozenset(self.allowed_elements)
            if any(not isinstance(symbol, str) or symbol not in _ELEMENT_ATOMIC_NUMBERS for symbol in normalized):
                raise ValueError("allowed_elements must contain canonical element symbols")
            object.__setattr__(self, "allowed_elements", normalized)
        units = _mapping(self.canonical_units, "canonical_units")
        if any(not isinstance(key, str) or not key.strip() or not isinstance(value, str) or not value.strip() for key, value in units.items()):
            raise ValueError("canonical_units must map names to non-empty strings")
        object.__setattr__(self, "canonical_units", units)
        for name in (
            "require_source_provenance", "require_license", "require_overlap_audit",
            "require_charge_spin_evidence", "require_explicit_state", "require_spin_parity",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")

    @property
    def element_scope(self) -> frozenset[str] | None:
        """Alias used by callers that describe the scope rather than a filter."""

        return self.allowed_elements

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "allowed_elements": None if self.allowed_elements is None else sorted(self.allowed_elements),
            "canonical_units": dict(self.canonical_units),
            "require_source_provenance": self.require_source_provenance,
            "require_license": self.require_license,
            "require_overlap_audit": self.require_overlap_audit,
            "require_charge_spin_evidence": self.require_charge_spin_evidence,
            "require_explicit_state": self.require_explicit_state,
            "require_spin_parity": self.require_spin_parity,
        }

    @property
    def digest(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def policy_digest(self) -> str:
        return self.digest

    def reasons_for(self, record: PublicRecord) -> tuple[str, ...]:
        """Return deterministic reason codes; an empty tuple means admitted."""

        reasons: set[str] = set()
        if record.admission == QUARANTINED:
            reasons.add("already_quarantined")
        elif record.admission not in ADMITTED:
            reasons.add("admission_invalid")
        if dict(record.units) != dict(self.canonical_units):
            reasons.add("units_noncanonical")

        if self.require_source_provenance:
            for name in (
                "source_dataset", "source_revision", "parent_reaction_id", "split_group",
                "geometry_locator", "geometry_hash", "geometry_origin", "reference_protocol_id",
                "record_status",
            ):
                if _unresolved(getattr(record, name)):
                    reasons.add(f"{name}_unresolved")
        if self.require_license and _unresolved(record.license_record):
            reasons.add("license_unresolved")
        if self.require_overlap_audit:
            if record.pretraining_overlap_audit in {UNKNOWN, "unavailable", "ambiguous", "missing", "not_provided", "unverified"}:
                reasons.add("overlap_audit_unresolved")
            elif record.pretraining_overlap_audit not in {"pass", "not_applicable"}:
                reasons.add("overlap_audit_failed")
        if self.require_charge_spin_evidence and _unresolved(record.charge_spin_evidence):
            reasons.add("charge_spin_evidence_unresolved")

        inputs = record.input_data
        charge = inputs.get("charge", _MISSING)
        multiplicity = inputs.get("multiplicity", _MISSING)
        if self.require_explicit_state:
            if charge is _MISSING:
                reasons.add("charge_missing")
            elif type(charge) is not int:
                reasons.add("charge_not_integer")
            if multiplicity is _MISSING:
                reasons.add("multiplicity_missing")
            elif type(multiplicity) is not int:
                reasons.add("multiplicity_not_integer")
            elif multiplicity < 1:
                reasons.add("multiplicity_not_positive")
        else:
            # Even an optional state must not silently accept bools.
            if charge is not _MISSING and type(charge) is not int:
                reasons.add("charge_not_integer")
            if multiplicity is not _MISSING and (type(multiplicity) is not int or multiplicity < 1):
                reasons.add("multiplicity_invalid")

        symbols, symbol_reason = _symbols_from_input(inputs)
        if self.allowed_elements is not None or self.require_spin_parity:
            if symbols is None:
                reasons.add(symbol_reason or "elements_missing")
            else:
                if self.allowed_elements is not None and any(symbol not in self.allowed_elements for symbol in symbols):
                    reasons.add("elements_out_of_scope")
                if self.require_spin_parity and type(charge) is int and type(multiplicity) is int and multiplicity >= 1:
                    electron_count = sum(_ELEMENT_ATOMIC_NUMBERS[symbol] for symbol in symbols) - charge
                    if electron_count < 1:
                        reasons.add("electron_count_invalid")
                    elif (electron_count - (multiplicity - 1)) % 2 != 0:
                        reasons.add("electron_spin_parity_mismatch")
        return tuple(sorted(reasons))


@dataclass(frozen=True)
class RecordFilterResult:
    """Deterministic accepted/quarantine partition and the policy digest."""

    accepted: tuple[PublicRecord, ...]
    quarantined: tuple[PublicRecord, ...]
    policy_digest: str

    @property
    def admitted(self) -> tuple[PublicRecord, ...]:
        return self.accepted

    @property
    def rejected(self) -> tuple[PublicRecord, ...]:
        return self.quarantined

    @property
    def records(self) -> tuple[PublicRecord, ...]:
        return tuple(sorted((*self.accepted, *self.quarantined), key=lambda row: (row.record_id, canonical_hash(row.to_dict()))))

    def __iter__(self):
        # Permit the convenient ``accepted, quarantined = filter_records(...)``
        # form without sacrificing named result fields.
        yield self.accepted
        yield self.quarantined


def filter_records(records: Iterable[PublicRecord], policy: RecordPolicy | None = None) -> RecordFilterResult:
    """Apply a policy while retaining every rejected row in quarantine.

    Duplicate IDs are rejected in the filter as well as by JSONL loading so a
    caller filtering an in-memory stream cannot accidentally admit an ambiguous
    identity.  Both partitions are sorted by record ID and canonical payload.
    """

    selected = policy or RecordPolicy()
    rows = list(records)
    counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, PublicRecord):
            raise TypeError("filter_records expects PublicRecord instances")
        counts[row.record_id] = counts.get(row.record_id, 0) + 1
    accepted: list[PublicRecord] = []
    quarantined: list[PublicRecord] = []
    for row in rows:
        reasons = set(selected.reasons_for(row))
        if counts[row.record_id] > 1:
            reasons.add("duplicate_record_id")
        if reasons:
            quarantined.append(row.quarantined(sorted(reasons)))
        else:
            accepted.append(row)
    key = lambda item: (item.record_id, canonical_hash(item.to_dict()))
    accepted.sort(key=key)
    quarantined.sort(key=key)
    return RecordFilterResult(tuple(accepted), tuple(quarantined), selected.digest)


def load_jsonl(path: str) -> list[PublicRecord]:
    """Load a JSONL manifest with strict rows and unique record IDs."""

    records: list[PublicRecord] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                record = PublicRecord.from_dict(payload)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid public record at line {line_number}: {exc}") from exc
            if record.record_id in seen:
                raise ValueError(f"duplicate public record_id at line {line_number}: {record.record_id}")
            seen.add(record.record_id)
            records.append(record)
    return sorted(records, key=lambda item: (item.record_id, canonical_hash(item.to_dict())))


def write_jsonl(path: str, records: Iterable[PublicRecord]) -> None:
    """Write unique canonical records in deterministic record-ID order."""

    rows = list(records)
    seen: set[str] = set()
    for record in rows:
        if not isinstance(record, PublicRecord):
            raise TypeError("write_jsonl expects PublicRecord instances")
        if record.record_id in seen:
            raise ValueError(f"duplicate public record_id: {record.record_id}")
        seen.add(record.record_id)
    rows.sort(key=lambda item: (item.record_id, canonical_hash(item.to_dict())))
    with open(path, "w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
