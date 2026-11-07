from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from test_track_b_workflow import track_record
from xtbflow.data.dft_da import DftDaSample
from xtbflow.data.task_manifest import TaskLoader, build_task_manifest, verify_task_manifest
from xtbflow.data.track_b import TrackBLeakageError
from xtbflow.evaluation import BlindRolloutConfig, blind_rollout, make_reactant_input
from xtbflow.models import JointEventGeometryFlow, total_electron_projector


def sample():
    x = np.array([[0., 0., 0.], [0., 0., .8]])
    b = np.array([[0., 1.], [1., 0.]])
    return DftDaSample(track_record(), "[H:1][H:2]", "[H:1][H:2]", x, x.copy(), x.copy(), b, b.copy(), (1, 1), (0, 1))


def test_task_counts_are_overlapping_views_not_summed_discoveries():
    row = sample().record
    result = build_task_manifest([row], version="test", source_manifest_sha256="a" * 64)
    for task in ("event_only", "geometry_only", "paired_joint"):
        assert result["counts"][task]["train"] == {"records": 1, "parent_groups": 1, "family_groups": 1}
    assert result["counts"]["energy_force"]["train"]["records"] == 0
    assert result["confirmation_records"] == 0
    verify_task_manifest(result, [row], "a" * 64)
    altered = deepcopy(result)
    altered["task_index"]["energy_force"]["train"].append(row.record_id)
    with pytest.raises(ValueError, match="does not match"):
        verify_task_manifest(altered, [row], "a" * 64)


def test_quarantine_and_confirmation_are_never_promoted():
    row = sample().record
    for admission in ("quarantine", "confirmatory"):
        excluded = replace(row, admission=admission)
        result = build_task_manifest([excluded], version="test", source_manifest_sha256="a" * 64)
        assert result["records"] == [] and result["excluded_admissions"][admission] == 1
    with pytest.raises(ValueError, match="duplicate"):
        build_task_manifest([row, row], version="test", source_manifest_sha256="a" * 64)


def test_source_group_leakage_blocks_manifest():
    first = sample().record
    second = replace(first, record_id="other", admission="development_test")
    with pytest.raises(TrackBLeakageError):
        build_task_manifest([first, second], version="test", source_manifest_sha256="a" * 64)


def test_targets_project_by_task_but_input_stays_identical():
    batches = {task: next(iter(TaskLoader([sample()], task=task, split="train", purpose="training")))
               for task in ("event_only", "geometry_only", "paired_joint")}
    assert set(batches["event_only"].targets) == {"event_bond_edits", "event_mask"}
    assert set(batches["geometry_only"].targets) == {"ts_coordinates", "ts_mask"}
    for key, value in batches["paired_joint"].reactant_input.items():
        for batch in batches.values():
            torch.testing.assert_close(value, batch.reactant_input[key], rtol=0, atol=0)
    assert batches["paired_joint"].training_views()["training_supervision"] is batches["paired_joint"].targets


def test_evaluation_labels_cannot_be_requested_as_training():
    row = replace(sample(), record=replace(sample().record, admission="development_test"))
    with pytest.raises(ValueError, match="only the training"):
        TaskLoader([row], task="paired_joint", split="test", purpose="training")
    batch = next(iter(TaskLoader([row], task="paired_joint", split="test", purpose="evaluation")))
    with pytest.raises(ValueError, match="evaluation labels"):
        batch.training_views()
    with pytest.raises(ValueError, match="unavailable"):
        TaskLoader([sample()], task="energy_force", split="train", purpose="training")


def test_hidden_label_mutation_does_not_change_generated_candidates():
    original = sample()
    changed = replace(original, product_smiles="hidden replacement", product_bonds=original.product_bonds + 2,
                      product_coordinates=original.product_coordinates + 7, ts_coordinates=original.ts_coordinates - 3,
                      record=replace(original.record, event_label={**original.record.event_label, "sha256": "9" * 64},
                                     ts_geometry={**original.record.ts_geometry, "sha256": "8" * 64}))
    left = next(iter(TaskLoader([original], task="paired_joint", split="train", purpose="training")))
    right = next(iter(TaskLoader([changed], task="paired_joint", split="train", purpose="training")))
    for key in left.reactant_input:
        torch.testing.assert_close(left.reactant_input[key], right.reactant_input[key], rtol=0, atol=0)
    assert not torch.equal(left.targets["ts_coordinates"], right.targets["ts_coordinates"])
    with torch.random.fork_rng():
        torch.manual_seed(137)
        model = JointEventGeometryFlow(total_electron_projector(("H", "H")), node_feature_dim=8,
                                       condition_feature_dim=4, hidden_dim=8, radial_features=4)
    model.eval()
    config = BlindRolloutConfig(nfe=2, candidate_cap=2)
    a = blind_rollout(model, make_reactant_input(original, 2), arm_id="joint_bidirectional", seed=19, config=config)
    b = blind_rollout(model, make_reactant_input(changed, 2), arm_id="joint_bidirectional", seed=19, config=config)
    assert [r.candidate_hash for r in a.attempts] == [r.candidate_hash for r in b.attempts]
    assert a.attempts == b.attempts
