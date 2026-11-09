from __future__ import annotations

import json

from xtbflow.calculators import MolecularSystem
from xtbflow.sampling.reactant_candidates import build_reactant_approach


def _system() -> MolecularSystem:
    return MolecularSystem(
        ("C", "H", "H", "C", "H", "H"),
        ((-2.0, 0.0, 0.0), (-2.7, 0.7, 0.0), (-2.7, -0.7, 0.0), (2.0, 0.0, 0.0), (2.7, 0.7, 0.0), (2.7, -0.7, 0.0)),
        0,
        1,
        {"periodic": False},
        "reactant-approach-fixture",
    )


def test_reactant_approach_is_deterministic_and_target_free():
    system = _system()
    first = build_reactant_approach(system, ((0, 1, 2), (3, 4, 5)), target_centroid_distance_angstrom=2.5)
    second = build_reactant_approach(system, ((0, 1, 2), (3, 4, 5)), target_centroid_distance_angstrom=2.5)

    assert first.coordinates == second.coordinates
    assert first.geometry_hash == second.geometry_hash
    assert first.metadata["input_hash"] == system.input_hash
    assert first.metadata["generator"] == "reactant-principal-axis-approach-v1"
    assert first.metadata["minimum_cross_component_distance_angstrom"] >= 0.70
    assert "target" not in json.dumps(first.metadata, sort_keys=True).lower()


def test_reactant_approach_rejects_non_partitioned_components():
    try:
        build_reactant_approach(_system(), ((0, 1), (2, 3, 4)), target_centroid_distance_angstrom=2.5)
    except ValueError as exc:
        assert "partition" in str(exc)
    else:
        raise AssertionError("non-partitioned reactant components must fail closed")
