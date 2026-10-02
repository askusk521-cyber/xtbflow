from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from xtbflow.evaluation import (
    BlindRolloutConfig,
    benchmark_rollouts,
    blind_rollout,
    make_reactant_input,
)
from xtbflow.models import JointEventGeometryFlow, total_electron_projector


def _sample(record_id: str = "row-1", parent: str = "parent-1"):
    record = SimpleNamespace(
        record_id=record_id,
        parent_reaction_id=parent,
        reactant={"charge": 0, "multiplicity": 1},
    )
    bonds = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    return SimpleNamespace(
        record=record,
        atomic_numbers=(6, 8),
        reactant_coordinates=np.asarray([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=np.float64),
        reactant_bonds=bonds,
        product_bonds=bonds.copy(),
        ts_coordinates=np.asarray([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=np.float64),
    )


def _zero_model() -> JointEventGeometryFlow:
    model = JointEventGeometryFlow(
        total_electron_projector(("C", "O")),
        node_feature_dim=8,
        condition_feature_dim=4,
        hidden_dim=8,
        radial_features=4,
    )
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.0)
    return model


def test_blind_rollout_starts_at_reactant_and_accounts_duplicates() -> None:
    sample = _sample()
    reactant = make_reactant_input(sample, 2)
    result = blind_rollout(
        _zero_model(),
        reactant,
        arm_id="both_off",
        seed=19,
        config=BlindRolloutConfig(nfe=3, candidate_cap=4),
    )
    assert result.actual_candidate_attempts == 4
    assert result.actual_flow_evaluations == 12
    assert len(result.accepted) == 1
    assert sum(row.status == "duplicate" for row in result.attempts) == 3
    assert result.failure_counts == {}


def test_replacing_hidden_labels_does_not_change_reactant_input_or_rollout() -> None:
    sample = _sample()
    changed = _sample()
    changed.product_bonds = np.asarray([[0.0, 2.0], [2.0, 0.0]], dtype=np.float64)
    changed.ts_coordinates = np.asarray([[0.0, 0.0, 0.0], [2.1, 0.0, 0.0]], dtype=np.float64)
    left = make_reactant_input(sample, 2)
    right = make_reactant_input(changed, 2)
    torch.testing.assert_close(left.event_state, right.event_state)
    torch.testing.assert_close(left.coordinates, right.coordinates)
    torch.testing.assert_close(left.node_features, right.node_features)
    first = blind_rollout(_zero_model(), left, arm_id="both_off", seed=7, config=BlindRolloutConfig(candidate_cap=2))
    second = blind_rollout(_zero_model(), right, arm_id="both_off", seed=7, config=BlindRolloutConfig(candidate_cap=2))
    assert [row.candidate_hash for row in first.attempts] == [row.candidate_hash for row in second.attempts]


def test_parent_macro_benchmark_resamples_parents() -> None:
    model = _zero_model()
    results = []
    samples = {}
    for index, parent in enumerate(("parent-a", "parent-a", "parent-b")):
        sample = _sample(f"row-{index}", parent)
        samples[sample.record.record_id] = sample
        results.append(
            blind_rollout(
                model,
                make_reactant_input(sample, 2),
                arm_id="both_off",
                seed=index,
                config=BlindRolloutConfig(candidate_cap=2),
            )
        )
    report = benchmark_rollouts(results, samples, bootstrap_resamples=50, bootstrap_seed=5)
    assert report["record_count"] == 3
    assert report["parent_count"] == 2
    assert report["parent_macro"]["event_hit"] == 1.0
    assert report["parent_bootstrap"]["event_hit"]["parent_count"] == 2
    assert len(report["parent_metrics"]) == 2
