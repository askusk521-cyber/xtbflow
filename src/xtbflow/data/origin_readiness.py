"""Bridge literature candidate registries to independent-reactant seed gates.

The audit is intentionally read-only.  A literature anchor may be complete
without being a runnable seed, and provisional charge/spin values are never
promoted.  Only explicit, contract-valid ``IndependentReactantSeed`` payloads
can make a candidate seed-ready.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .independent_seeds import IndependentReactantSeed, audit_seed_leakage


ORIGIN_CANDIDATE_SCHEMA = "xtbflow-origin-candidates/v0.1"
ORIGIN_SEED_READINESS_SCHEMA = "xtbflow-origin-seed-readiness-audit/v1"
_ALLOWED_ADMISSIONS = frozenset({"quarantine", "admitted"})
_ALLOWED_CHARGE_SPIN_STATES = frozenset({"unresolved", "provisional", "confirmed"})


class OriginReadinessError(ValueError):
    """Raised when the candidate registry itself is malformed."""


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise OriginReadinessError(f"{name} must be a mapping")
    return dict(value)


def _sequence(value: Any, name: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise OriginReadinessError(f"{name} must be a sequence")
    return list(value)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OriginReadinessError(f"{name} must be a non-empty string")
    return value


def _text_list(value: Any, name: str, *, allow_empty: bool = False) -> list[str]:
    items = _sequence(value, name)
    if not allow_empty and not items:
        raise OriginReadinessError(f"{name} must not be empty")
    for index, item in enumerate(items):
        _text(item, f"{name}[{index}]")
    return [str(item) for item in items]


def _strict_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise OriginReadinessError(f"{name} must be a strict integer")
    if minimum is not None and value < minimum:
        raise OriginReadinessError(f"{name} must be >= {minimum}")
    return value


def _validate_anchor(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    prefix = f"candidates[{index}]"
    candidate_id = _text(row.get("candidate_id"), f"{prefix}.candidate_id")
    parent_id = _text(
        row.get("parent_reaction_id"), f"{prefix}.parent_reaction_id"
    )
    family_id = _text(
        row.get("mechanism_family_id"), f"{prefix}.mechanism_family_id"
    )
    _text(row.get("title"), f"{prefix}.title")
    source = _mapping(row.get("source"), f"{prefix}.source")
    for field in ("doi", "locator", "verified_on"):
        _text(source.get(field), f"{prefix}.source.{field}")
    _text_list(row.get("reactant_identities"), f"{prefix}.reactant_identities")
    _text(row.get("event_intent"), f"{prefix}.event_intent")
    _text_list(row.get("competing_events"), f"{prefix}.competing_events")
    source_conditions = _mapping(
        row.get("source_conditions"), f"{prefix}.source_conditions"
    )
    if not source_conditions:
        raise OriginReadinessError(f"{prefix}.source_conditions must not be empty")
    admission = _text(row.get("admission"), f"{prefix}.admission")
    if admission not in _ALLOWED_ADMISSIONS:
        raise OriginReadinessError(f"unsupported admission for {candidate_id}: {admission}")
    structure_status = _text(
        row.get("structure_status"), f"{prefix}.structure_status"
    )
    charge_spin = _mapping(row.get("charge_spin"), f"{prefix}.charge_spin")
    charge_spin_status = _text(
        charge_spin.get("status"), f"{prefix}.charge_spin.status"
    )
    if charge_spin_status not in _ALLOWED_CHARGE_SPIN_STATES:
        raise OriginReadinessError(
            f"unsupported charge/spin status for {candidate_id}: {charge_spin_status}"
        )
    declared_missing = _text_list(
        row.get("missing", []), f"{prefix}.missing", allow_empty=True
    )
    return {
        "candidate_id": candidate_id,
        "parent_reaction_id": parent_id,
        "mechanism_family_id": family_id,
        "source_doi": source["doi"],
        "admission": admission,
        "structure_status": structure_status,
        "charge_spin": charge_spin,
        "charge_spin_status": charge_spin_status,
        "declared_missing": declared_missing,
    }


def _validate_seed_records(
    row: Mapping[str, Any],
    *,
    candidate_id: str,
    parent_reaction_id: str,
    mechanism_family_id: str,
    expected_charge: int | None,
    expected_multiplicity: int | None,
) -> tuple[list[IndependentReactantSeed], list[str]]:
    raw_records = row.get("independent_seed_records")
    if raw_records is None:
        return [], ["independent_seed_records_absent"]
    records = _sequence(raw_records, f"{candidate_id}.independent_seed_records")
    if not records:
        return [], ["independent_seed_records_absent"]

    validated: list[IndependentReactantSeed] = []
    errors: list[str] = []
    for index, raw_record in enumerate(records):
        try:
            seed = IndependentReactantSeed.from_dict(
                _mapping(raw_record, f"{candidate_id}.independent_seed_records[{index}]")
            )
        except (OriginReadinessError, TypeError, ValueError) as exc:
            errors.append(f"invalid_seed_record[{index}]: {exc}")
            continue
        if seed.parent_system_id != parent_reaction_id:
            errors.append(f"seed_parent_mismatch[{index}]")
            continue
        if seed.family_id != mechanism_family_id:
            errors.append(f"seed_family_mismatch[{index}]")
            continue
        if expected_charge is not None and seed.charge != expected_charge:
            errors.append(f"seed_charge_mismatch[{index}]")
            continue
        if (
            expected_multiplicity is not None
            and seed.multiplicity != expected_multiplicity
        ):
            errors.append(f"seed_multiplicity_mismatch[{index}]")
            continue
        validated.append(seed)
    return validated, errors


def audit_origin_candidate_registry(
    payload: Mapping[str, Any],
    *,
    registry_locator: str,
    registry_source_commit: str,
    registry_sha256: str,
    repository_base_commit: str,
) -> dict[str, Any]:
    """Audit an exact candidate registry against the independent-seed contract."""

    registry = _mapping(payload, "candidate registry")
    if registry.get("schema") != ORIGIN_CANDIDATE_SCHEMA:
        raise OriginReadinessError(
            f"unsupported candidate registry schema: {registry.get('schema')}"
        )
    _text(registry_locator, "registry_locator")
    _text(registry_source_commit, "registry_source_commit")
    _text(registry_sha256, "registry_sha256")
    _text(repository_base_commit, "repository_base_commit")

    rows = _sequence(registry.get("candidates"), "candidates")
    if not rows:
        raise OriginReadinessError("candidate registry must contain candidates")
    declared_count = _strict_int(registry.get("candidate_count"), "candidate_count")
    if declared_count != len(rows):
        raise OriginReadinessError("candidate_count does not match candidates")

    anchor_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    candidate_ids: list[str] = []
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"candidates[{index}]")
        anchor = _validate_anchor(row, index)
        candidate_ids.append(anchor["candidate_id"])
        anchor_rows.append((row, anchor))
    if len(set(candidate_ids)) != len(candidate_ids):
        raise OriginReadinessError("candidate_id values must be unique")

    observed_family_counts = Counter(
        anchor["mechanism_family_id"] for _, anchor in anchor_rows
    )
    declared_family_counts = _mapping(
        registry.get("family_counts"), "family_counts"
    )
    if dict(observed_family_counts) != declared_family_counts:
        raise OriginReadinessError("family_counts does not match candidates")

    candidate_reports: list[dict[str, Any]] = []
    all_validated_seeds: list[IndependentReactantSeed] = []
    for row, anchor in anchor_rows:
        blockers: list[str] = []
        if anchor["admission"] != "admitted":
            blockers.append("candidate_not_admitted")
        if anchor["structure_status"] != "frozen_hashed":
            blockers.append("structure_not_frozen_hashed")

        charge_spin = anchor["charge_spin"]
        status = anchor["charge_spin_status"]
        expected_charge: int | None = None
        expected_multiplicity: int | None = None
        if status != "confirmed":
            blockers.append("charge_spin_not_confirmed")
        else:
            try:
                expected_charge = _strict_int(
                    charge_spin.get("charge"), "charge_spin.charge"
                )
                expected_multiplicity = _strict_int(
                    charge_spin.get("multiplicity"),
                    "charge_spin.multiplicity",
                    minimum=1,
                )
            except OriginReadinessError:
                blockers.append("confirmed_charge_spin_values_invalid")

        if anchor["declared_missing"]:
            blockers.append("declared_requirements_missing")
        if not isinstance(row.get("license_record"), str) or not row[
            "license_record"
        ].strip():
            blockers.append("license_record_absent")

        seeds, seed_errors = _validate_seed_records(
            row,
            candidate_id=anchor["candidate_id"],
            parent_reaction_id=anchor["parent_reaction_id"],
            mechanism_family_id=anchor["mechanism_family_id"],
            expected_charge=expected_charge,
            expected_multiplicity=expected_multiplicity,
        )
        if seed_errors:
            blockers.extend(error.split(":", 1)[0] for error in seed_errors)
        identity_state_ready = not any(
            code
            in {
                "candidate_not_admitted",
                "structure_not_frozen_hashed",
                "charge_spin_not_confirmed",
                "confirmed_charge_spin_values_invalid",
                "declared_requirements_missing",
                "license_record_absent",
            }
            for code in blockers
        )
        independent_seed_ready = identity_state_ready and bool(seeds) and not seed_errors
        if independent_seed_ready:
            all_validated_seeds.extend(seeds)

        provisional_values_ignored = status != "confirmed" and (
            "charge" in charge_spin or "multiplicity" in charge_spin
        )
        candidate_reports.append(
            {
                "candidate_id": anchor["candidate_id"],
                "parent_reaction_id": anchor["parent_reaction_id"],
                "mechanism_family_id": anchor["mechanism_family_id"],
                "source_doi": anchor["source_doi"],
                "literature_anchor_ready": True,
                "identity_state_ready": identity_state_ready,
                "independent_seed_ready": independent_seed_ready,
                "validated_seed_record_count": len(seeds),
                "provisional_charge_spin_values_ignored": (
                    provisional_values_ignored
                ),
                "declared_missing": anchor["declared_missing"],
                "blocking_codes": sorted(set(blockers)),
                "seed_validation_errors": seed_errors,
            }
        )

    leakage_report: dict[str, Any] | None = None
    leakage_error: str | None = None
    if all_validated_seeds:
        try:
            leakage_report = audit_seed_leakage(
                all_validated_seeds,
                strict_family_holdout=True,
            )
        except ValueError as exc:
            leakage_error = str(exc)

    ready_candidates = sum(
        1 for row in candidate_reports if row["independent_seed_ready"]
    )
    declared_runnable = _strict_int(
        registry.get("runnable_input_count"), "runnable_input_count"
    )
    if declared_runnable != ready_candidates:
        raise OriginReadinessError(
            "runnable_input_count does not match seed-ready candidates"
        )

    return {
        "schema_version": ORIGIN_SEED_READINESS_SCHEMA,
        "registry": {
            "schema": ORIGIN_CANDIDATE_SCHEMA,
            "locator": registry_locator,
            "source_commit": registry_source_commit,
            "sha256": registry_sha256,
            "candidate_count": len(candidate_reports),
        },
        "repository_base_commit": repository_base_commit,
        "literature_anchor_ready_count": sum(
            1 for row in candidate_reports if row["literature_anchor_ready"]
        ),
        "identity_state_ready_count": sum(
            1 for row in candidate_reports if row["identity_state_ready"]
        ),
        "independent_seed_ready_count": ready_candidates,
        "validated_seed_record_count": len(all_validated_seeds),
        "strict_family_holdout": True,
        "leakage_report": leakage_report,
        "leakage_error": leakage_error,
        "candidate_reports": candidate_reports,
        "seed_manifest_population_allowed": bool(all_validated_seeds)
        and leakage_error is None,
        "search_authorized": False,
        "scientific_claim_allowed": False,
        "claim_boundary": (
            "Candidate-to-seed governance evidence only. This audit never creates "
            "reactant geometries, fills missing electronic states, authorizes "
            "search, or provides event/TS supervision."
        ),
    }
