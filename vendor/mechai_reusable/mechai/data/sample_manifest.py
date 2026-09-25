"""Validation for the unified event--geometry sample manifest.

The manifest is a provenance and task-definition contract.  It does not join
Transition1x, Kingfisher, FlowER, or any other data source, and it does not
decide whether a chemistry label is physically correct.  Missing values stay
explicitly ``unknown``/``ambiguous`` instead of being converted to neutral or
negative labels.
"""
from __future__ import annotations

from typing import Any, Mapping


SCHEMA_VERSION = "event-geometry-sample-v1"
TASK_MODES = {
    "endpoint_geometry",
    "intent_event_geometry",
    "endpoint_microsequence_geometry",
}
EVIDENCE_STATUSES = {
    "observed",
    "declared",
    "derived_under_contract",
    "unknown",
    "ambiguous",
    "unavailable",
}
GATE_STATUSES = {"pending", "pass", "blocked", "unknown"}
COORDINATE_ROLES = {
    "unavailable",
    "independent_metadata",
    "declared_endpoint",
    "generated_candidate",
    "reference_derived",
}
GEOMETRY_SOURCES = {
    "generated_candidate",
    "independent_initial_guess",
    "declared_endpoint",
    "reference_ts_oracle",
}
PERMISSION_MODES = {
    "none",
    "declared_net_intent",
    "declared_center",
    "declared_endpoints",
    "full_mapped_product",
}
VARIABLES = {
    "current_graph",
    "formal_charge",
    "multiplicity",
    "microstate",
    "stereo",
    "reactant_coordinates",
    "product_coordinates",
    "endpoint_coordinates",
    "net_heavy_atom_edits",
    "reactive_centers",
    "full_mapped_product",
    "solvent_membership",
    "active_solvent_ids",
    "event_identity",
    "proton_transfer_assignment",
    "event_order",
    "intermediate_states",
    "ts_geometry",
    "competing_channel_identity",
    "family_identity",
}


class ManifestValidationError(ValueError):
    """Raised when a sample manifest violates the task/provenance contract."""


def _error(path: str, message: str) -> str:
    return f"{path}: {message}"


def _require_mapping(value: Any, path: str, errors: list[str]) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(_error(path, "must be an object"))
        return None
    return value


def _require_keys(value: Mapping[str, Any], keys: tuple[str, ...], path: str, errors: list[str]) -> None:
    for key in keys:
        if key not in value:
            errors.append(_error(path, f"missing required field {key!r}"))


def _evidence_field(value: Any, path: str, errors: list[str]) -> None:
    field = _require_mapping(value, path, errors)
    if field is None:
        return
    _require_keys(field, ("value", "status", "source_ref"), path, errors)
    status = field.get("status")
    if status not in EVIDENCE_STATUSES:
        errors.append(_error(f"{path}.status", f"must be one of {sorted(EVIDENCE_STATUSES)}"))
    ref = field.get("source_ref")
    if ref is not None and not isinstance(ref, str):
        errors.append(_error(f"{path}.source_ref", "must be string or null"))


def _string_list(value: Any, path: str, errors: list[str]) -> list[str] | None:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        errors.append(_error(path, "must be a list of nonempty strings"))
        return None
    if len(set(value)) != len(value):
        errors.append(_error(path, "must not contain duplicate values"))
    return value


def validate_manifest(manifest: Mapping[str, Any], *, raise_on_error: bool = True) -> list[str]:
    """Validate one manifest and return deterministic error messages.

    The JSON schema provides the structural contract for external tooling; this
    function additionally checks equal atom metadata lengths and the
    task-mode-specific known/unknown variable rules that JSON Schema alone
    would make unnecessarily opaque.
    """
    errors: list[str] = []
    root = _require_mapping(manifest, "$", errors)
    if root is None:
        if raise_on_error:
            raise ManifestValidationError("; ".join(errors))
        return errors
    _require_keys(root, ("schema_version", "sample_id", "task_mode", "source", "atoms", "condition", "variables", "candidate_pool", "geometry", "labels", "gate"), "$", errors)
    if root.get("schema_version") != SCHEMA_VERSION:
        errors.append(_error("$.schema_version", f"must equal {SCHEMA_VERSION!r}"))
    if not isinstance(root.get("sample_id"), str) or not root.get("sample_id"):
        errors.append(_error("$.sample_id", "must be a nonempty string"))
    task_mode = root.get("task_mode")
    if task_mode not in TASK_MODES:
        errors.append(_error("$.task_mode", f"must be one of {sorted(TASK_MODES)}"))

    source = _require_mapping(root.get("source"), "$.source", errors)
    if source is not None:
        _require_keys(source, ("dataset", "revision", "raw_record_id", "family_evidence"), "$.source", errors)
        for key in ("dataset", "revision", "raw_record_id"):
            if not isinstance(source.get(key), str) or not source.get(key):
                errors.append(_error(f"$.source.{key}", "must be a nonempty string"))
        if source.get("family_evidence") not in EVIDENCE_STATUSES:
            errors.append(_error("$.source.family_evidence", "has invalid evidence status"))
        for key in ("parent_calculation_ids",):
            if key in source and _string_list(source[key], f"$.source.{key}", errors) is None:
                pass

    atoms = _require_mapping(root.get("atoms"), "$.atoms", errors)
    if atoms is not None:
        _require_keys(atoms, ("atomic_numbers", "coordinate_unit"), "$.atoms", errors)
        numbers = atoms.get("atomic_numbers")
        if not isinstance(numbers, list) or not numbers or not all(isinstance(value, int) and value > 0 for value in numbers):
            errors.append(_error("$.atoms.atomic_numbers", "must be a nonempty list of positive integers"))
            numbers = []
        if atoms.get("coordinate_unit") != "angstrom":
            errors.append(_error("$.atoms.coordinate_unit", "must be 'angstrom'"))
        for key in ("isotopes", "map_ids", "stereo"):
            if key in atoms and atoms[key] is not None:
                if not isinstance(atoms[key], list) or len(atoms[key]) != len(numbers):
                    errors.append(_error(f"$.atoms.{key}", "must have the same length as atomic_numbers"))

    condition = _require_mapping(root.get("condition"), "$.condition", errors)
    known_condition = None
    if condition is not None:
        _require_keys(condition, ("current_graph", "formal_charge", "multiplicity", "microstate", "input_coordinates", "coordinate_provenance", "solvent_membership", "known_condition"), "$.condition", errors)
        for key in ("current_graph", "formal_charge", "multiplicity", "microstate", "input_coordinates"):
            _evidence_field(condition.get(key), f"$.condition.{key}", errors)
        provenance = _require_mapping(condition.get("coordinate_provenance"), "$.condition.coordinate_provenance", errors)
        if provenance is not None:
            _require_keys(provenance, ("role", "selected_by_reference"), "$.condition.coordinate_provenance", errors)
            if provenance.get("role") not in COORDINATE_ROLES:
                errors.append(_error("$.condition.coordinate_provenance.role", "has invalid coordinate role"))
            if provenance.get("selected_by_reference") not in (True, False, None):
                errors.append(_error("$.condition.coordinate_provenance.selected_by_reference", "must be boolean or null"))
        solvent = condition.get("solvent_membership")
        if not isinstance(solvent, list):
            errors.append(_error("$.condition.solvent_membership", "must be a list"))
        else:
            for index, item in enumerate(solvent):
                field = _require_mapping(item, f"$.condition.solvent_membership[{index}]", errors)
                if field is not None:
                    _require_keys(field, ("solvent_id", "atom_indices", "status", "source_ref"), f"$.condition.solvent_membership[{index}]", errors)
                    if not isinstance(field.get("solvent_id"), str) or not field.get("solvent_id"):
                        errors.append(_error(f"$.condition.solvent_membership[{index}].solvent_id", "must be nonempty string"))
                    if not isinstance(field.get("atom_indices"), list) or not all(isinstance(value, int) and value >= 0 for value in field.get("atom_indices", [])):
                        errors.append(_error(f"$.condition.solvent_membership[{index}].atom_indices", "must be a list of nonnegative integers"))
                    if field.get("status") not in EVIDENCE_STATUSES:
                        errors.append(_error(f"$.condition.solvent_membership[{index}].status", "has invalid evidence status"))
        known_condition = _require_mapping(condition.get("known_condition"), "$.condition.known_condition", errors)
        if known_condition is not None:
            _require_keys(known_condition, ("permission_mode", "already_given_variables", "variables_to_infer"), "$.condition.known_condition", errors)
            if known_condition.get("permission_mode") not in PERMISSION_MODES:
                errors.append(_error("$.condition.known_condition.permission_mode", "has invalid permission mode"))

    variables = _require_mapping(root.get("variables"), "$.variables", errors)
    known: set[str] = set()
    unknown: set[str] = set()
    targets: set[str] = set()
    if variables is not None:
        for key, target in (("known", known), ("unknown", unknown), ("prediction_targets", targets)):
            _require_keys(variables, ("known", "unknown", "prediction_targets"), "$.variables", errors)
            values = _string_list(variables.get(key), f"$.variables.{key}", errors)
            if values is not None:
                invalid = set(values) - VARIABLES
                if invalid:
                    errors.append(_error(f"$.variables.{key}", f"unknown variables: {sorted(invalid)}"))
                target.update(values)
        if known & unknown:
            errors.append(_error("$.variables", f"known and unknown overlap: {sorted(known & unknown)}"))
        if not targets <= unknown:
            errors.append(_error("$.variables.prediction_targets", "every prediction target must be listed as unknown"))

    if known_condition is not None:
        given = set(_string_list(known_condition.get("already_given_variables", []), "$.condition.known_condition.already_given_variables", errors) or [])
        infer = set(_string_list(known_condition.get("variables_to_infer", []), "$.condition.known_condition.variables_to_infer", errors) or [])
        if given - known:
            errors.append(_error("$.condition.known_condition.already_given_variables", "contains variables absent from variables.known"))
        if infer - unknown:
            errors.append(_error("$.condition.known_condition.variables_to_infer", "contains variables absent from variables.unknown"))

    if task_mode == "endpoint_geometry":
        if "ts_geometry" not in targets:
            errors.append(_error("$.variables.prediction_targets", "endpoint_geometry requires ts_geometry"))
        if known_condition is not None and known_condition.get("permission_mode") not in {"declared_center", "declared_endpoints", "full_mapped_product"}:
            errors.append(_error("$.condition.known_condition.permission_mode", "endpoint_geometry requires declared center/endpoints or full mapped product"))
    elif task_mode == "intent_event_geometry":
        for variable in ("event_identity", "ts_geometry"):
            if variable not in targets:
                errors.append(_error("$.variables.prediction_targets", f"intent_event_geometry requires {variable}"))
        if known_condition is not None and known_condition.get("permission_mode") == "full_mapped_product":
            errors.append(_error("$.condition.known_condition.permission_mode", "intent_event_geometry cannot claim full mapped product as an unknown-event input"))
    elif task_mode == "endpoint_microsequence_geometry":
        for variable in ("event_order", "intermediate_states", "ts_geometry"):
            if variable not in targets:
                errors.append(_error("$.variables.prediction_targets", f"endpoint_microsequence_geometry requires {variable}"))
        if known_condition is not None and known_condition.get("permission_mode") not in {"declared_endpoints", "full_mapped_product"}:
            errors.append(_error("$.condition.known_condition.permission_mode", "endpoint_microsequence_geometry requires declared endpoints or full mapped product"))

    pool = _require_mapping(root.get("candidate_pool"), "$.candidate_pool", errors)
    if pool is not None:
        _require_keys(pool, ("generator_revision", "uses_reference_labels", "events"), "$.candidate_pool", errors)
        if not isinstance(pool.get("uses_reference_labels"), bool):
            errors.append(_error("$.candidate_pool.uses_reference_labels", "must be boolean"))
        if not isinstance(pool.get("events"), list):
            errors.append(_error("$.candidate_pool.events", "must be a list"))
        elif pool.get("uses_reference_labels"):
            for index, event in enumerate(pool["events"]):
                if isinstance(event, Mapping) and event.get("origin") == "reference_label":
                    pass

    geometry = _require_mapping(root.get("geometry"), "$.geometry", errors)
    if geometry is not None:
        _require_keys(geometry, ("candidate_geometry_source", "coordinates_ref", "geometry_validity"), "$.geometry", errors)
        if geometry.get("candidate_geometry_source") not in GEOMETRY_SOURCES:
            errors.append(_error("$.geometry.candidate_geometry_source", "has invalid geometry source"))
        _evidence_field(geometry.get("coordinates_ref"), "$.geometry.coordinates_ref", errors)
        _evidence_field(geometry.get("geometry_validity"), "$.geometry.geometry_validity", errors)

    labels = _require_mapping(root.get("labels"), "$.labels", errors)
    if labels is not None:
        _require_keys(labels, ("references", "competing_channel_coverage", "physical_negative_labels_available"), "$.labels", errors)
        if not isinstance(labels.get("references"), list):
            errors.append(_error("$.labels.references", "must be a list"))
        if labels.get("competing_channel_coverage") not in {"known", "partial", "unknown", "none"}:
            errors.append(_error("$.labels.competing_channel_coverage", "has invalid coverage status"))
        if not isinstance(labels.get("physical_negative_labels_available"), bool):
            errors.append(_error("$.labels.physical_negative_labels_available", "must be boolean"))

    gate = _require_mapping(root.get("gate"), "$.gate", errors)
    if gate is not None:
        _require_keys(gate, ("geometry_only", "event_geometry_matched", "independent_evaluation", "failure_reasons"), "$.gate", errors)
        for key in ("geometry_only", "event_geometry_matched", "independent_evaluation"):
            if gate.get(key) not in GATE_STATUSES:
                errors.append(_error(f"$.gate.{key}", "has invalid gate status"))
        if not isinstance(gate.get("failure_reasons"), list) or not all(isinstance(item, str) for item in gate.get("failure_reasons", [])):
            errors.append(_error("$.gate.failure_reasons", "must be a list of strings"))
        if gate.get("independent_evaluation") == "pass" and source is not None and source.get("family_evidence") in {"unknown", "ambiguous"}:
            errors.append(_error("$.gate.independent_evaluation", "cannot pass while source.family_evidence is unknown/ambiguous"))

    if raise_on_error and errors:
        raise ManifestValidationError("; ".join(errors))
    return errors
