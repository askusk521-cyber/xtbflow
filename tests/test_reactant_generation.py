from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from xtbflow.evaluation.generation import (
    ReactantView,
    ReferenceLabels,
    candidate_fingerprints,
    checkpoint_atom_count,
    generate_candidates,
    label_replacement_invariant,
    load_reactant_inputs,
    score_candidates,
)
from xtbflow.models import JointFlowRuntimeConfig, total_electron_projector


ROOT = Path(__file__).parents[1]
RUNTIME = ROOT / "configs/models/joint_flow_runtime_v0.1.json"


def _row(**overrides):
    row = {
        "schema": "xtbflow-reactant-input/v1",
        "record_id": "r0",
        "symbols": ["C", "O"],
        "map_ids": [1, 2],
        "reactant_coordinates_angstrom": [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        "reactant_bonds": [[0.0, 1.0], [1.0, 0.0]],
        "charge": 0,
        "multiplicity": 1,
    }
    row.update(overrides)
    return row


def _zero_model():
    runtime = JointFlowRuntimeConfig.load(RUNTIME)
    model = runtime.build(total_electron_projector(("C", "O"), dtype=torch.float32))
    for parameter in model.parameters():
        parameter.data.zero_()
    return model


def _labels(product_bond: float, distance: float) -> ReferenceLabels:
    return ReferenceLabels(
        "r0",
        (1, 2),
        np.asarray([[0.0, product_bond], [product_bond, 0.0]], dtype=np.float64),
        np.asarray([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=np.float64),
    )


def test_reactant_input_rejects_reference_fields_recursively():
    with pytest.raises(ValueError, match="forbidden target field"):
        ReactantView.from_mapping({**_row(), "product_bonds": [[0.0, 2.0], [2.0, 0.0]]})
    with pytest.raises(ValueError, match="forbidden target field"):
        ReactantView.from_mapping({**_row(), "aux": {"reference": {"ts_coordinates": []}}})


def test_reactant_jsonl_loader_rejects_target_rows(tmp_path: Path):
    path = tmp_path / "inputs.jsonl"
    path.write_text(json.dumps({**_row(), "event": {"edits": []}}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_reactant_inputs(path)


def test_generation_starts_from_reactants_and_decodes_without_labels():
    model = _zero_model()
    view = ReactantView.from_mapping(_row())
    batch = generate_candidates(model, view, max_atoms=2, control_mode="joint_bidirectional", steps=4)
    assert batch.attempted == 1
    assert batch.accepted == 1
    assert batch.rejected == 0
    candidate = batch.candidates[0]
    assert candidate.record_id == "r0"
    assert len(candidate.coordinates) == 2
    assert len(candidate.packed_event) == 3
    assert all(value == value for row in candidate.coordinates for value in row)


def test_replacing_reference_labels_preserves_candidates_but_changes_scores():
    model = _zero_model()
    view = ReactantView.from_mapping(_row())
    labels_one = _labels(1.0, 1.2)
    labels_two = _labels(2.0, 1.3)
    invariant = label_replacement_invariant(
        model,
        view,
        (labels_one, labels_two),
        max_atoms=2,
        control_mode="joint_bidirectional",
        steps=4,
    )
    assert invariant["invariant"] is True
    first = generate_candidates(model, view, max_atoms=2, control_mode="joint_bidirectional", steps=4)
    second = generate_candidates(model, view, max_atoms=2, control_mode="joint_bidirectional", steps=4)
    assert candidate_fingerprints(first.candidates) == candidate_fingerprints(second.candidates)
    score_one = score_candidates(view, labels_one, first.candidates)
    score_two = score_candidates(view, labels_two, second.candidates)
    assert score_one[0]["event_exact_match"] is True
    assert score_two[0]["event_exact_match"] is False
    assert score_one[0]["candidate_fingerprint"] == score_two[0]["candidate_fingerprint"]


def test_checkpoint_atom_count_uses_packed_constraint_width():
    assert checkpoint_atom_count({"model": {"constraint_matrix": torch.zeros(1, 6)}}) == 3


def test_scored_candidate_fingerprint_cannot_be_rewritten():
    model = _zero_model()
    view = ReactantView.from_mapping(_row())
    candidate = generate_candidates(
        model, view, max_atoms=2, control_mode="joint_bidirectional", steps=4
    ).candidates[0]
    payload = candidate.to_mapping()
    payload["candidate_fingerprint"] = "tampered"
    path = Path(__file__).parents[1] / "scripts" / "score_reactant_generation.py"
    spec = importlib.util.spec_from_file_location("xtbflow_reactant_generation_scorer", path)
    assert spec is not None and spec.loader is not None
    scorer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = scorer
    spec.loader.exec_module(scorer)

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        scorer._candidate_from_mapping(payload)
