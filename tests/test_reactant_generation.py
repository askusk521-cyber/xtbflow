from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from xtbflow.evaluation import generation as generation_module
from xtbflow.evaluation.generation import (
    GeneratedCandidate,
    GenerationBatch,
    ReactantView,
    ReferenceLabels,
    candidate_fingerprints,
    checkpoint_atom_count,
    generate_candidates,
    generate_repeated_candidates,
    generate_repeated_candidates_with_accounting,
    deduplicate_generated_candidates,
    label_replacement_invariant,
    load_reactant_inputs,
    score_candidates,
    _initial_state,
    _model_inputs,
)
from xtbflow.evaluation.terminal import GenerationProtocol
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


def test_unresolved_label_metric_is_reported_without_becoming_zero():
    path = Path(__file__).parents[1] / "scripts" / "score_reactant_generation.py"
    spec = importlib.util.spec_from_file_location("xtbflow_reactant_generation_scorer_missing", path)
    assert spec is not None and spec.loader is not None
    scorer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = scorer
    spec.loader.exec_module(scorer)

    rows = [
        {
            "parent_reaction_id": "p1",
            "event_exact_match": 0,
            "event_bond_mse": None,
            "geometry_endpoint_mse": 1.0,
        },
        {
            "parent_reaction_id": "p2",
            "event_exact_match": 1,
            "event_bond_mse": None,
            "geometry_endpoint_mse": 3.0,
        },
    ]
    summary = scorer._summary_for_rows(rows)
    assert summary["record_weighted"] == {
        "event_exact_match": pytest.approx(0.5),
        "geometry_endpoint_mse": pytest.approx(2.0),
    }
    assert summary["metric_coverage"]["event_bond_mse"]["missing"] == 2
    assert summary["unavailable_metrics"] == ["event_bond_mse"]

    paired = scorer._paired_summary(
        rows,
        [
            {
                "parent_reaction_id": "p1",
                "event_exact_match": 1,
                "event_bond_mse": 0.0,
                "geometry_endpoint_mse": 2.0,
            },
            {
                "parent_reaction_id": "p2",
                "event_exact_match": 1,
                "event_bond_mse": 0.0,
                "geometry_endpoint_mse": 4.0,
            },
        ],
        left_label="learned",
        right_label="reference",
        bootstrap_samples=32,
        seed=5,
    )
    assert paired["unavailable_metrics"] == ["event_bond_mse"]
    assert "event_exact_match" in paired["metrics"]


def test_empty_method_row_retains_parent_denominator():
    path = Path(__file__).parents[1] / "scripts" / "score_reactant_generation.py"
    spec = importlib.util.spec_from_file_location("xtbflow_reactant_generation_scorer_empty", path)
    assert spec is not None and spec.loader is not None
    scorer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = scorer
    spec.loader.exec_module(scorer)
    view = ReactantView.from_mapping(_row())
    summary, scores = scorer._record_metrics(view, _labels(1.0, 1.2), (), method="learned", max_slots=2)
    assert scores == ()
    assert summary["status"] == "empty"
    assert summary["event_exact_match"] is None
    assert summary["geometry_endpoint_mse"] is None


def test_generation_runner_keeps_failure_stage_statuses_distinct():
    path = Path(__file__).parents[1] / "scripts" / "run_reactant_generation.py"
    spec = importlib.util.spec_from_file_location("xtbflow_reactant_generation_runner", path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    assert runner._status_for_failure_stage("report_write") == "report_write_failed"
    assert runner._status_for_failure_stage("settlement") == "settlement_failed"
    assert runner._status_for_failure_stage("budget_exhausted") == "budget_exhausted"
    assert runner._status_for_failure_stage("generation_preflight") == "not_run"
    assert runner._status_for_failure_stage("integration") == "execution_failed"


def test_repeated_generation_requires_and_uses_a_stochastic_initial_state():
    model = _zero_model()
    view = ReactantView.from_mapping(_row())
    with pytest.raises(ValueError, match="stochastic"):
        generate_repeated_candidates(
            model,
            view,
            max_atoms=2,
            control_mode="joint_bidirectional",
            protocol=GenerationProtocol(candidate_cap=2, attempt_seeds=(11, 17)),
            steps=4,
        )

    protocol = GenerationProtocol(
        candidate_cap=2,
        attempt_seeds=(11, 17),
        initial_state_policy="geometry_noise",
    )
    candidates = generate_repeated_candidates(
        model,
        view,
        max_atoms=2,
        control_mode="joint_bidirectional",
        protocol=protocol,
        steps=4,
        noise_scale=0.02,
    )
    assert len(candidates) == 2
    assert len({candidate.fingerprint() for candidate in candidates}) == 2


def test_repeated_generation_retains_event_slots_when_geometry_is_duplicate(monkeypatch):
    model = _zero_model()
    view = ReactantView.from_mapping(_row())
    packed_by_seed = {11: (1, 2, 3), 17: (4, 5, 6)}

    def fake_generate_candidates(*args, **kwargs):
        seed = kwargs["attempt_seed"]
        candidate = GeneratedCandidate(
            record_id="r0",
            control_mode="joint_bidirectional",
            coordinates=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            packed_event=packed_by_seed[seed],
            steps=4,
        )
        return GenerationBatch((candidate,), 1, 0, (), 4, 1)

    monkeypatch.setattr(generation_module, "generate_candidates", fake_generate_candidates)
    result = generate_repeated_candidates_with_accounting(
        model,
        view,
        max_atoms=2,
        control_mode="joint_bidirectional",
        protocol=GenerationProtocol(
            candidate_cap=2,
            attempt_seeds=(11, 17),
            initial_state_policy="geometry_noise",
        ),
        steps=4,
    )
    assert len(result.candidates) == 1
    assert len(result.event_candidates) == 2
    assert result.unique_candidates == 1
    assert result.event_unique_candidates == 2
    assert result.duplicate_geometry == 1
    assert result.duplicate_event == 0


def test_event_noise_initial_state_is_conservation_preserving():
    model = _zero_model()
    view = ReactantView.from_mapping(_row())
    state, coordinates, _, atom_mask, event_mask, _ = _model_inputs(
        view, 2, dtype=torch.float32, device=torch.device("cpu")
    )
    noisy, _ = _initial_state(
        state,
        coordinates,
        event_mask=event_mask,
        atom_mask=atom_mask,
        event_projector=model.event_projector,
        policy="event_noise",
        seed=11,
        noise_scale=0.2,
    )
    residual = model.event_projector.residual(noisy - state)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6, rtol=0)
    assert torch.isfinite(noisy).all()


def test_geometry_slot_deduplication_ignores_event_duplicates():
    first = GeneratedCandidate(
        record_id="r0",
        control_mode="joint_bidirectional",
        coordinates=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        packed_event=(1, 2, 3),
        steps=4,
        candidate_index=0,
    )
    duplicate_geometry = GeneratedCandidate(
        record_id="r0",
        control_mode="joint_bidirectional",
        coordinates=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        packed_event=(4, 5, 6),
        steps=4,
        candidate_index=1,
    )
    kept = deduplicate_generated_candidates((first, duplicate_geometry), max_slots=2)
    assert kept == (first,)


def test_event_slot_deduplication_preserves_distinct_events_at_one_geometry():
    first = GeneratedCandidate(
        record_id="r0",
        control_mode="reactant_rule",
        coordinates=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        packed_event=(1, 2, 3),
        steps=0,
        candidate_index=0,
    )
    second = GeneratedCandidate(
        record_id="r0",
        control_mode="reactant_rule",
        coordinates=first.coordinates,
        packed_event=(4, 5, 6),
        steps=0,
        candidate_index=1,
    )
    kept = deduplicate_generated_candidates(
        (first, second), max_slots=2, mode="geometry_event"
    )
    assert kept == (first, second)


def test_scorer_reads_event_candidate_slots_separately_from_geometry_slots():
    path = Path(__file__).parents[1] / "scripts" / "score_reactant_generation.py"
    spec = importlib.util.spec_from_file_location("xtbflow_reactant_generation_scorer_slots", path)
    assert spec is not None and spec.loader is not None
    scorer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = scorer
    spec.loader.exec_module(scorer)
    view = ReactantView.from_mapping(_row())
    labels = _labels(1.0, 1.2)
    first = GeneratedCandidate(
        record_id="r0",
        control_mode="joint_bidirectional",
        coordinates=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        packed_event=(1, 2, 3),
        steps=4,
        candidate_index=0,
    )
    second = GeneratedCandidate(
        record_id="r0",
        control_mode="joint_bidirectional",
        coordinates=first.coordinates,
        packed_event=(4, 5, 6),
        steps=4,
        candidate_index=1,
    )
    summary, _ = scorer._record_metrics(
        view,
        labels,
        (first,),
        event_candidates=(first, second),
        method="learned",
        max_slots=2,
    )
    assert summary["geometry_candidate_count"] == 1
    assert summary["event_candidate_count"] == 2
    assert summary["raw_event_candidate_count"] == 2


def test_decode_failure_is_unresolved_for_event_exact_match():
    view = ReactantView.from_mapping(_row())
    labels = _labels(1.0, 1.2)
    malformed = GeneratedCandidate(
        record_id="r0",
        control_mode="joint_bidirectional",
        coordinates=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        packed_event=(1,),
        steps=4,
    )
    rows = score_candidates(view, labels, (malformed,))
    assert rows[0]["decode_status"] == "decode_failed"
    assert rows[0]["event_exact_match"] is None
