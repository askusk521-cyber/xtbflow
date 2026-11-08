"""Frozen W11 campaign input contracts.

The campaign manifest describes a common, reactant-side input pool for a
bounded aqueous demonstration.  It records explicit component identities,
water roles, electronic states, provenance and comparison arms without
containing products, transition states, observed events or reference labels.

This module is deliberately independent from the research PR stack.  A
manifest can be prepared and audited before any model, calculator or
certification implementation is available.  ``quarantine`` and
``development_screening`` states may keep unresolved fields explicit; a
``frozen`` manifest requires every input needed to reproduce the campaign.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CAMPAIGN_SCHEMA = "xtbflow-w11-campaign/v1"
CAMPAIGN_STATES = frozenset({"development_screening", "frozen", "quarantine"})
COMPONENT_ROLES = frozenset(
    {"substrate", "co_reactant", "water", "buffer", "counterion", "spectator"}
)
WATER_ROLES = frozenset({"participating", "spectator", "unresolved"})
CANDIDATE_STATES = frozenset({"development", "quarantine"})
PROPOSAL_MODES = frozenset({"rule", "serial", "joint", "joint_agent"})
PHYSICS_MODES = frozenset({"fixed", "adaptive"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UNRESOLVED = frozenset({"unknown", "unresolved", "not_provided", "unverified"})
_FORBIDDEN_TARGET_KEYS = frozenset(
    {
        "product",
        "mapped_product",
        "product_coordinates",
        "ts_geometry",
        "reference_ts",
        "reference_mode",
        "observed_event",
        "sealed_test_label",
        "target",
    }
)


class CampaignManifestError(ValueError):
    """Raised when a W11 campaign manifest is malformed or unsafe."""


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CampaignManifestError(f"{name} must be a mapping")
    result = dict(value)
    try:
        json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CampaignManifestError(f"{name} must be JSON-compatible") from exc
    return result


def _sequence(value: Any, name: str, *, allow_empty: bool = False) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CampaignManifestError(f"{name} must be a sequence")
    items = list(value)
    if not allow_empty and not items:
        raise CampaignManifestError(f"{name} must not be empty")
    return items


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CampaignManifestError(f"{name} must be a non-empty string")
    return value.strip()


def _strict_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise CampaignManifestError(f"{name} must be a boolean")
    return value


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CampaignManifestError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignManifestError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise CampaignManifestError(f"{name} must be finite and >= {minimum}")
    return result


def _sha256(value: Any, name: str) -> str:
    text = _text(value, name)
    if not _SHA256.fullmatch(text):
        raise CampaignManifestError(f"{name} must be a lowercase SHA-256")
    return text


def _sha256_or_unresolved(value: Any, name: str, *, allow_unresolved: bool) -> str:
    text = _text(value, name)
    if allow_unresolved and text.lower() in _UNRESOLVED:
        return text
    if not _SHA256.fullmatch(text):
        raise CampaignManifestError(f"{name} must be a lowercase SHA-256")
    return text


def _int_or_unresolved(value: Any, name: str, *, allow_unresolved: bool, minimum: int) -> int | str:
    if allow_unresolved and isinstance(value, str) and value.strip().lower() in _UNRESOLVED:
        return value.strip().lower()
    return _strict_int(value, name, minimum=minimum)


def _text_list(value: Any, name: str, *, allow_empty: bool = False) -> list[str]:
    items = _sequence(value, name, allow_empty=allow_empty)
    result = [_text(item, f"{name}[{index}]") for index, item in enumerate(items)]
    if len(set(result)) != len(result):
        raise CampaignManifestError(f"{name} values must be unique")
    return result


def _iso_timestamp(value: Any, name: str, *, allow_unknown: bool = False) -> str:
    text = _text(value, name)
    if allow_unknown and text.lower() in {"unknown", "unresolved", "not_frozen"}:
        return text
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CampaignManifestError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise CampaignManifestError(f"{name} must include an explicit timezone")
    return text


def canonical_hash(value: Any) -> str:
    """Return a deterministic hash for JSON-compatible campaign inputs."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CampaignManifestError("campaign value must be JSON-compatible") from exc
    return hashlib.sha256(encoded).hexdigest()


def _reject_target_fields(payload: Mapping[str, Any], name: str) -> None:
    forbidden = sorted(_FORBIDDEN_TARGET_KEYS.intersection(payload))
    if forbidden:
        raise CampaignManifestError(
            f"{name} contains target-derived fields: {forbidden}"
        )


def _validate_source(value: Any) -> dict[str, Any]:
    source = _mapping(value, "source")
    required = {"locator", "revision", "sha256", "license_record"}
    missing = sorted(required - set(source))
    if missing:
        raise CampaignManifestError(f"source missing required fields: {missing}")
    _text(source["locator"], "source.locator")
    _text(source["revision"], "source.revision")
    _sha256(source["sha256"], "source.sha256")
    _text(source["license_record"], "source.license_record")
    return source


def _validate_composition(value: Any, name: str) -> dict[str, int]:
    composition = _mapping(value, name)
    if not composition:
        raise CampaignManifestError(f"{name} must not be empty")
    result: dict[str, int] = {}
    for element, count in composition.items():
        symbol = _text(element, f"{name}.element")
        result[symbol] = _strict_int(count, f"{name}.{symbol}", minimum=1)
    return result


def _validate_component(value: Any, index: int) -> dict[str, Any]:
    name = f"atom_pool[{index}]"
    component = _mapping(value, name)
    _reject_target_fields(component, name)
    required = {
        "component_id",
        "role",
        "species_label",
        "explicit_hydrogens",
        "composition",
        "charge",
        "multiplicity",
        "input_locator",
        "input_sha256",
        "license_record",
    }
    missing = sorted(required - set(component))
    if missing:
        raise CampaignManifestError(f"{name} missing required fields: {missing}")
    result = dict(component)
    result["component_id"] = _text(component["component_id"], f"{name}.component_id")
    result["role"] = _text(component["role"], f"{name}.role")
    if result["role"] not in COMPONENT_ROLES:
        raise CampaignManifestError(f"unsupported {name}.role: {result['role']}")
    result["species_label"] = _text(component["species_label"], f"{name}.species_label")
    result["explicit_hydrogens"] = _strict_bool(
        component["explicit_hydrogens"], f"{name}.explicit_hydrogens"
    )
    result["composition"] = _validate_composition(component["composition"], f"{name}.composition")
    result["charge"] = _strict_int(component["charge"], f"{name}.charge", minimum=-1000)
    result["multiplicity"] = _strict_int(
        component["multiplicity"], f"{name}.multiplicity", minimum=1
    )
    result["input_locator"] = _text(component["input_locator"], f"{name}.input_locator")
    result["input_sha256"] = _sha256(component["input_sha256"], f"{name}.input_sha256")
    result["license_record"] = _text(component["license_record"], f"{name}.license_record")
    if result["role"] == "water" and not result["explicit_hydrogens"]:
        raise CampaignManifestError(f"{name} water components require explicit_hydrogens=true")
    return result


def _validate_water_roles(
    value: Any,
    name: str,
    *,
    water_ids: set[str],
    frozen: bool,
) -> dict[str, str]:
    roles = _mapping(value, name)
    if set(roles) != water_ids:
        raise CampaignManifestError(
            f"{name} must declare every water component exactly once"
        )
    result: dict[str, str] = {}
    for component_id, role in roles.items():
        _text(component_id, f"{name}.component_id")
        role_text = _text(role, f"{name}.{component_id}")
        if role_text not in WATER_ROLES:
            raise CampaignManifestError(f"unsupported water role: {role_text}")
        if frozen and role_text == "unresolved":
            raise CampaignManifestError(
                f"frozen campaigns cannot keep {name}.{component_id} unresolved"
            )
        result[component_id] = role_text
    if frozen and "participating" not in result.values():
        raise CampaignManifestError(
            "frozen campaigns require at least one participating water"
        )
    return result


def _validate_candidate(
    value: Any,
    index: int,
    *,
    component_ids: set[str],
    water_ids: set[str],
    frozen: bool,
) -> dict[str, Any]:
    name = f"candidates[{index}]"
    candidate = _mapping(value, name)
    _reject_target_fields(candidate, name)
    required = {
        "candidate_id",
        "seed_id",
        "parent_system_id",
        "family_id",
        "geometry_locator",
        "geometry_sha256",
        "charge",
        "multiplicity",
        "component_ids",
        "water_roles",
        "input_origin",
        "provenance",
        "license_record",
        "status",
    }
    missing = sorted(required - set(candidate))
    if missing:
        raise CampaignManifestError(f"{name} missing required fields: {missing}")
    result = dict(candidate)
    for field in (
        "candidate_id",
        "seed_id",
        "parent_system_id",
        "family_id",
        "geometry_locator",
        "input_origin",
        "provenance",
        "license_record",
        "status",
    ):
        result[field] = _text(candidate[field], f"{name}.{field}")
    if result["status"] not in CANDIDATE_STATES:
        raise CampaignManifestError(f"unsupported {name}.status: {result['status']}")
    if result["input_origin"] != "independent_reactant":
        raise CampaignManifestError(
            f"{name}.input_origin must be independently prepared reactant input"
        )
    unresolved = result["status"] == "quarantine"
    result["geometry_sha256"] = _sha256_or_unresolved(
        candidate["geometry_sha256"], f"{name}.geometry_sha256", allow_unresolved=unresolved
    )
    result["charge"] = _int_or_unresolved(
        candidate["charge"], f"{name}.charge", allow_unresolved=unresolved, minimum=-1000
    )
    result["multiplicity"] = _int_or_unresolved(
        candidate["multiplicity"], f"{name}.multiplicity", allow_unresolved=unresolved, minimum=1
    )
    candidate_components = _text_list(candidate["component_ids"], f"{name}.component_ids")
    candidate_component_set = set(candidate_components)
    if not candidate_component_set.issubset(component_ids):
        unknown = sorted(candidate_component_set - component_ids)
        raise CampaignManifestError(
            f"{name}.component_ids reference unknown components: {unknown}"
        )
    if candidate_component_set != component_ids:
        missing_components = sorted(component_ids - candidate_component_set)
        raise CampaignManifestError(
            f"{name}.component_ids must preserve the fixed atom pool; missing: {missing_components}"
        )
    result["component_ids"] = candidate_components
    result["water_roles"] = _validate_water_roles(
        candidate["water_roles"],
        f"{name}.water_roles",
        water_ids=water_ids.intersection(candidate_components),
        frozen=frozen,
    )
    if frozen and result["status"] != "development":
        raise CampaignManifestError("frozen campaigns cannot include quarantined candidates")
    if result["status"] == "quarantine":
        missing_fields = _text_list(
            candidate.get("missing", ["unresolved campaign input"]),
            f"{name}.missing",
            allow_empty=False,
        )
        result["missing"] = missing_fields
    elif "missing" in candidate:
        raise CampaignManifestError(f"{name}.development candidates cannot declare missing fields")
    return result


def _validate_protocol(value: Any, *, frozen: bool) -> dict[str, Any]:
    protocol = _mapping(value, "reference_protocol")
    required = {"protocol_id", "protocol_sha256", "method", "software", "version", "units"}
    missing = sorted(required - set(protocol))
    if missing:
        raise CampaignManifestError(f"reference_protocol missing required fields: {missing}")
    result = dict(protocol)
    for field in ("protocol_id", "method", "software", "version"):
        result[field] = _text(protocol[field], f"reference_protocol.{field}")
    result["protocol_sha256"] = _sha256(
        protocol["protocol_sha256"], "reference_protocol.protocol_sha256"
    )
    units = _mapping(protocol["units"], "reference_protocol.units")
    if not units:
        raise CampaignManifestError("reference_protocol.units must not be empty")
    result["units"] = {
        _text(key, "reference_protocol.units.key"): _text(value, "reference_protocol.units.value")
        for key, value in units.items()
    }
    if frozen and any(text.lower() in {"unknown", "unresolved"} for text in result.values() if isinstance(text, str)):
        raise CampaignManifestError("frozen campaigns cannot use unknown reference protocol fields")
    return result


def _validate_budget(value: Any) -> dict[str, Any]:
    budget = _mapping(value, "budget")
    required = {"calculator_calls", "wall_seconds", "cpu_core_hours", "gpu_hours"}
    missing = sorted(required - set(budget))
    if missing:
        raise CampaignManifestError(f"budget missing required fields: {missing}")
    result = dict(budget)
    result["calculator_calls"] = _strict_int(
        budget["calculator_calls"], "budget.calculator_calls", minimum=1
    )
    for field in ("wall_seconds", "cpu_core_hours", "gpu_hours"):
        result[field] = _number(budget[field], f"budget.{field}", minimum=0.0)
    return result


def _validate_deduplication(value: Any) -> dict[str, Any]:
    dedup = _mapping(value, "deduplication")
    required = {"keys", "within_parent_system", "cross_strategy"}
    missing = sorted(required - set(dedup))
    if missing:
        raise CampaignManifestError(f"deduplication missing required fields: {missing}")
    result = dict(dedup)
    result["keys"] = _text_list(dedup["keys"], "deduplication.keys")
    result["within_parent_system"] = _strict_bool(
        dedup["within_parent_system"], "deduplication.within_parent_system"
    )
    result["cross_strategy"] = _strict_bool(
        dedup["cross_strategy"], "deduplication.cross_strategy"
    )
    return result


def _validate_comparison(
    value: Any,
    index: int,
    *,
    candidate_ids: set[str],
    input_fingerprint: str,
) -> dict[str, Any]:
    name = f"comparisons[{index}]"
    comparison = _mapping(value, name)
    required = {
        "comparison_id",
        "proposal_mode",
        "physics_mode",
        "candidate_ids",
        "budget_id",
        "input_fingerprint",
    }
    missing = sorted(required - set(comparison))
    if missing:
        raise CampaignManifestError(f"{name} missing required fields: {missing}")
    result = dict(comparison)
    result["comparison_id"] = _text(comparison["comparison_id"], f"{name}.comparison_id")
    result["proposal_mode"] = _text(comparison["proposal_mode"], f"{name}.proposal_mode")
    if result["proposal_mode"] not in PROPOSAL_MODES:
        raise CampaignManifestError(f"unsupported proposal mode: {result['proposal_mode']}")
    result["physics_mode"] = _text(comparison["physics_mode"], f"{name}.physics_mode")
    if result["physics_mode"] not in PHYSICS_MODES:
        raise CampaignManifestError(f"unsupported physics mode: {result['physics_mode']}")
    listed = set(_text_list(comparison["candidate_ids"], f"{name}.candidate_ids"))
    if listed != candidate_ids:
        raise CampaignManifestError(
            f"{name}.candidate_ids must match the frozen common input pool"
        )
    result["candidate_ids"] = sorted(listed)
    result["budget_id"] = _text(comparison["budget_id"], f"{name}.budget_id")
    result["input_fingerprint"] = _sha256(
        comparison["input_fingerprint"], f"{name}.input_fingerprint"
    )
    if result["input_fingerprint"] != input_fingerprint:
        raise CampaignManifestError(
            f"{name}.input_fingerprint does not match campaign input pool"
        )
    if result["proposal_mode"] == "joint_agent":
        result["agent_policy"] = _text(comparison.get("agent_policy"), f"{name}.agent_policy")
    elif "agent_policy" in comparison:
        raise CampaignManifestError(f"{name}.agent_policy is only valid for joint_agent")
    return result


@dataclass(frozen=True)
class CampaignManifest:
    """A bounded W11 input pool and its fair comparison arms."""

    campaign_id: str
    status: str
    description: str
    source: Mapping[str, Any]
    atom_pool: tuple[Mapping[str, Any], ...]
    candidates: tuple[Mapping[str, Any], ...]
    reference_protocol: Mapping[str, Any]
    budget: Mapping[str, Any]
    deduplication: Mapping[str, Any]
    comparisons: tuple[Mapping[str, Any], ...]
    claim_limit: str
    frozen_at: str | None = None
    schema_version: str = CAMPAIGN_SCHEMA

    def __post_init__(self) -> None:
        payload = self.to_dict()
        validated = self._validate_payload(payload)
        object.__setattr__(self, "source", validated["source"])
        object.__setattr__(self, "atom_pool", tuple(validated["atom_pool"]))
        object.__setattr__(self, "candidates", tuple(validated["candidates"]))
        object.__setattr__(self, "reference_protocol", validated["reference_protocol"])
        object.__setattr__(self, "budget", validated["budget"])
        object.__setattr__(self, "deduplication", validated["deduplication"])
        object.__setattr__(self, "comparisons", tuple(validated["comparisons"]))
        object.__setattr__(self, "frozen_at", validated["frozen_at"])

    @classmethod
    def _validate_payload(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        data = _mapping(payload, "campaign manifest")
        allowed = {
            "schema_version",
            "campaign_id",
            "status",
            "description",
            "source",
            "atom_pool",
            "candidates",
            "reference_protocol",
            "budget",
            "deduplication",
            "comparisons",
            "claim_limit",
            "frozen_at",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise CampaignManifestError(f"unsupported campaign fields: {unknown}")
        if data.get("schema_version") != CAMPAIGN_SCHEMA:
            raise CampaignManifestError(
                f"unsupported campaign schema: {data.get('schema_version')}"
            )
        campaign_id = _text(data.get("campaign_id"), "campaign_id")
        status = _text(data.get("status"), "status")
        if status not in CAMPAIGN_STATES:
            raise CampaignManifestError(f"unsupported campaign status: {status}")
        description = _text(data.get("description"), "description")
        claim_limit = _text(data.get("claim_limit"), "claim_limit")
        source = _validate_source(data.get("source"))
        raw_components = _sequence(data.get("atom_pool"), "atom_pool")
        components = [_validate_component(row, i) for i, row in enumerate(raw_components)]
        component_ids = [row["component_id"] for row in components]
        if len(set(component_ids)) != len(component_ids):
            raise CampaignManifestError("atom_pool component_id values must be unique")
        water_ids = {row["component_id"] for row in components if row["role"] == "water"}
        if not water_ids:
            raise CampaignManifestError("atom_pool must include at least one water component")
        frozen = status == "frozen"
        raw_candidates = _sequence(data.get("candidates"), "candidates")
        candidates = [
            _validate_candidate(
                row,
                i,
                component_ids=set(component_ids),
                water_ids=water_ids,
                frozen=frozen,
            )
            for i, row in enumerate(raw_candidates)
        ]
        candidate_ids = [row["candidate_id"] for row in candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise CampaignManifestError("candidate_id values must be unique")
        candidate_input = {
            "source": source,
            "atom_pool": components,
            "candidates": candidates,
        }
        input_fingerprint = canonical_hash(candidate_input)
        protocol = _validate_protocol(data.get("reference_protocol"), frozen=frozen)
        budget = _validate_budget(data.get("budget"))
        deduplication = _validate_deduplication(data.get("deduplication"))
        raw_comparisons = _sequence(data.get("comparisons"), "comparisons")
        comparisons = [
            _validate_comparison(
                row,
                i,
                candidate_ids=set(candidate_ids),
                input_fingerprint=input_fingerprint,
            )
            for i, row in enumerate(raw_comparisons)
        ]
        comparison_ids = [row["comparison_id"] for row in comparisons]
        if len(set(comparison_ids)) != len(comparison_ids):
            raise CampaignManifestError("comparison_id values must be unique")
        if status == "frozen":
            frozen_at = _iso_timestamp(data.get("frozen_at"), "frozen_at")
        else:
            raw_frozen_at = data.get("frozen_at")
            frozen_at = None if raw_frozen_at is None else _iso_timestamp(
                raw_frozen_at, "frozen_at", allow_unknown=True
            )
        return {
            "schema_version": CAMPAIGN_SCHEMA,
            "campaign_id": campaign_id,
            "status": status,
            "description": description,
            "source": source,
            "atom_pool": components,
            "candidates": candidates,
            "reference_protocol": protocol,
            "budget": budget,
            "deduplication": deduplication,
            "comparisons": comparisons,
            "claim_limit": claim_limit,
            "frozen_at": frozen_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CampaignManifest":
        validated = cls._validate_payload(payload)
        return cls(
            campaign_id=validated["campaign_id"],
            status=validated["status"],
            description=validated["description"],
            source=validated["source"],
            atom_pool=tuple(validated["atom_pool"]),
            candidates=tuple(validated["candidates"]),
            reference_protocol=validated["reference_protocol"],
            budget=validated["budget"],
            deduplication=validated["deduplication"],
            comparisons=tuple(validated["comparisons"]),
            claim_limit=validated["claim_limit"],
            frozen_at=validated["frozen_at"],
            schema_version=validated["schema_version"],
        )

    def input_fingerprint(self) -> str:
        return canonical_hash(
            {
                "source": self.source,
                "atom_pool": list(self.atom_pool),
                "candidates": list(self.candidates),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "status": self.status,
            "description": self.description,
            "source": dict(self.source),
            "atom_pool": [dict(row) for row in self.atom_pool],
            "candidates": [dict(row) for row in self.candidates],
            "reference_protocol": dict(self.reference_protocol),
            "budget": dict(self.budget),
            "deduplication": dict(self.deduplication),
            "comparisons": [dict(row) for row in self.comparisons],
            "claim_limit": self.claim_limit,
            "frozen_at": self.frozen_at,
        }


def load_campaign_manifest(path: str | Path) -> CampaignManifest:
    """Load and validate one UTF-8 JSON campaign manifest."""

    source = Path(path)
    try:
        with source.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignManifestError(f"unable to read campaign manifest {source}: {exc}") from exc
    return CampaignManifest.from_dict(payload)


def write_campaign_manifest(path: str | Path, manifest: CampaignManifest) -> None:
    """Write a validated campaign manifest with deterministic formatting."""

    if not isinstance(manifest, CampaignManifest):
        raise CampaignManifestError("manifest must be a CampaignManifest")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(
            manifest.to_dict(),
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        handle.write("\n")


def validate_campaign_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a mapping and return its normalized JSON-ready form."""

    return CampaignManifest._validate_payload(payload)
