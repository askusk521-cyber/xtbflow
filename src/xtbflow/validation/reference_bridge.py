"""Same-geometry reference/semiempirical bridge records and metrics."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from xtbflow.calculators import CalculationResult, MolecularSystem


def _force_rows(
    values: Sequence[Sequence[float]] | None, *, name: str
) -> tuple[tuple[float, float, float], ...]:
    if values is None:
        raise ValueError(f"{name} forces are required")
    rows: list[tuple[float, float, float]] = []
    for row in values:
        if len(row) != 3:
            raise ValueError(f"{name} force rows must have three components")
        converted = tuple(float(value) for value in row)
        if not all(math.isfinite(value) for value in converted):
            raise ValueError(f"{name} forces must be finite")
        rows.append(converted)
    if not rows:
        raise ValueError(f"{name} forces cannot be empty")
    return tuple(rows)


def force_delta_metrics(
    reference: Sequence[Sequence[float]],
    comparison: Sequence[Sequence[float]],
) -> dict[str, float]:
    """Compare force components after both backends reach the public unit contract."""

    left = _force_rows(reference, name="reference")
    right = _force_rows(comparison, name="comparison")
    if len(left) != len(right):
        raise ValueError("reference and comparison force inventories differ")
    deltas = [
        right_value - reference_value
        for left_row, right_row in zip(left, right)
        for reference_value, right_value in zip(left_row, right_row)
    ]
    absolute = [abs(value) for value in deltas]
    return {
        "component_mae_hartree_per_angstrom": sum(absolute) / len(absolute),
        "component_rmse_hartree_per_angstrom": math.sqrt(
            sum(value * value for value in deltas) / len(deltas)
        ),
        "component_max_abs_hartree_per_angstrom": max(absolute),
    }


def bridge_case_record(
    system: MolecularSystem,
    reference: CalculationResult,
    comparison: CalculationResult,
    *,
    selection_reason: str,
) -> dict[str, Any]:
    """Create one auditable comparison without pooling absolute energies."""

    if reference.status != "success" or comparison.status != "success":
        raise ValueError("bridge records require two successful calculator results")
    if reference.input_hash != system.input_hash:
        raise ValueError("reference result does not match the requested geometry/state")
    if comparison.input_hash != system.input_hash:
        raise ValueError("comparison result does not match the requested geometry/state")
    if reference.energy is None or comparison.energy is None:
        raise ValueError("bridge records require energies from both calculators")
    metrics = force_delta_metrics(reference.forces or (), comparison.forces or ())
    return {
        "system_id": system.system_id,
        "symbols": list(system.symbols),
        "coordinates_angstrom": [list(row) for row in system.coordinates],
        "charge": system.charge,
        "multiplicity": system.multiplicity,
        "input_hash": system.input_hash,
        "selection_reason": selection_reason,
        "same_geometry_energy_delta_hartree": (
            float(comparison.energy) - float(reference.energy)
        ),
        "force_delta": metrics,
    }


def summarize_bridge(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate force errors only; cross-stoichiometry energy pooling is forbidden."""

    successful = [record for record in records if record.get("status") == "success"]
    if not successful:
        return {
            "successful_cases": 0,
            "force_component_mae_mean_hartree_per_angstrom": None,
            "force_component_max_abs_hartree_per_angstrom": None,
            "energy_aggregation": "not_reported_across_stoichiometries",
        }
    maes = [
        float(record["bridge"]["force_delta"]["component_mae_hartree_per_angstrom"])
        for record in successful
    ]
    maxima = [
        float(record["bridge"]["force_delta"]["component_max_abs_hartree_per_angstrom"])
        for record in successful
    ]
    return {
        "successful_cases": len(successful),
        "force_component_mae_mean_hartree_per_angstrom": sum(maes) / len(maes),
        "force_component_max_abs_hartree_per_angstrom": max(maxima),
        "energy_aggregation": "not_reported_across_stoichiometries",
    }
