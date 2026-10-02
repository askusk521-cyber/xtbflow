from __future__ import annotations

import pytest

from xtbflow.training.task_routing import (
    LabelAvailability,
    TaskExample,
    TaskMode,
    TaskRoutingError,
    route_examples,
    views_for_example,
)


def example(**overrides):
    values = dict(
        record_id="r1",
        inputs={"reactant_coordinates": [[0.0, 0.0, 0.0]], "charge": 0, "multiplicity": 1},
        targets={
            "event": {"edits": []},
            "ts_geometry": [[0.1, 0.0, 0.0]],
            "energy": -1.0,
            "forces": [[0.0, 0.0, 0.0]],
        },
        availability=LabelAvailability(
            event=True, geometry=True, energy=True, forces=True,
            paired_identity=True, same_geometry_e_f=True,
        ),
    )
    values.update(overrides)
    return TaskExample(**values)


def test_one_record_can_feed_independent_views_and_explicit_joint_view():
    views = views_for_example(example())
    assert [view.mode for view in views] == [
        TaskMode.EVENT_ONLY,
        TaskMode.GEOMETRY_ONLY,
        TaskMode.ENERGY_FORCE,
        TaskMode.PAIRED_JOINT,
    ]
    assert all(view.input_fingerprint == views[0].input_fingerprint for view in views)
    assert views[0].loss_mask == {"event": True, "geometry": False, "energy": False, "forces": False}
    assert "ts_geometry" not in views[0].inputs
    assert "ts_geometry" in views[1].targets


def test_missing_pair_identity_never_creates_joint_view():
    views = views_for_example(example(availability=LabelAvailability(event=True, geometry=True)))
    assert [view.mode for view in views] == [TaskMode.EVENT_ONLY, TaskMode.GEOMETRY_ONLY]


def test_energy_force_requires_same_geometry_attestation():
    availability = LabelAvailability(energy=True, forces=True, same_geometry_e_f=False)
    assert views_for_example(example(availability=availability)) == ()


def test_route_is_deterministic_and_keeps_records_separate():
    grouped = route_examples([example(record_id="z"), example(record_id="a")])
    assert [view.record_id for view in grouped[TaskMode.EVENT_ONLY]] == ["a", "z"]
    assert len(grouped[TaskMode.PAIRED_JOINT]) == 2


def test_target_derived_fields_in_inputs_are_rejected():
    with pytest.raises(TaskRoutingError, match="target-derived"):
        example(inputs={"reactant_coordinates": [], "product": "answer"})


def test_availability_mask_is_authoritative():
    with pytest.raises(TaskRoutingError, match="claims event"):
        example(availability=LabelAvailability(event=True), targets={})


def test_mapping_aliases_and_conflicts_are_checked():
    parsed = LabelAvailability.from_mapping({"event_geometry_pair": True, "event": True})
    assert parsed.paired_identity is True
    with pytest.raises(TaskRoutingError, match="conflicting"):
        LabelAvailability.from_mapping({"paired_identity": True, "event_geometry_pair": False})
