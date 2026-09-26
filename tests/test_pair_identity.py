from __future__ import annotations

import json

import pytest

from xtbflow.calculators import CalculationResult, CalculatorProtocol, MolecularSystem
from xtbflow.data import append_pair_cache, build_ef_pair, failure_pair, load_pair_cache


def system() -> MolecularSystem:
    return MolecularSystem(("H", "H"), ((0.0, 0.0, 0.0), (0.7, 0.0, 0.0)), 0, 1, {"solvent": "none"})


def result(protocol: str, item: MolecularSystem, energy: float, force: float) -> CalculationResult:
    return CalculationResult("fixture", protocol, item.input_hash, item.charge, item.multiplicity, "energy_forces", energy=energy, forces=((force, 0.0, 0.0), (-force, 0.0, 0.0)))


def test_pair_preserves_exact_geometry_and_delta_labels(tmp_path):
    item = system()
    pair = build_ef_pair(item, result("gfn2", item, -1.0, 0.1), result("ref", item, -1.2, 0.2), source_record_id="row-1")
    assert pair.status == "success"
    assert pair.delta_energy == pytest.approx(-0.2)
    assert pair.delta_forces[0][0] == pytest.approx(0.1)
    assert pair.geometry_hash
    path = tmp_path / "pairs.jsonl"
    assert append_pair_cache(path, [pair]) == (1, 0)
    assert append_pair_cache(path, [pair]) == (0, 1)
    loaded = load_pair_cache(path)
    assert loaded[pair.pair_id].to_dict() == pair.to_dict()


def test_pair_rejects_coordinate_or_state_mismatch():
    item = system()
    altered = item.with_coordinates(((0.0, 0.0, 0.0), (0.8, 0.0, 0.0)))
    with pytest.raises(ValueError, match="input hash"):
        build_ef_pair(item, result("gfn2", altered, -1.0, 0.1), result("ref", item, -1.2, 0.2), source_record_id="row-1")


def test_failure_rows_are_retained_and_cache_conflicts_fail(tmp_path):
    item = system()
    failure = failure_pair(item, source_record_id="row-fail", semi_empirical_protocol_id="gfn2", reference_protocol_id="ref", category="convergence", message="SCF failed")
    assert failure.status == "failure"
    path = tmp_path / "pairs.jsonl"
    append_pair_cache(path, [failure])
    payload = failure.to_dict()
    payload["error_message"] = "different"
    path.write_text(path.read_text() + json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting"):
        load_pair_cache(path)

