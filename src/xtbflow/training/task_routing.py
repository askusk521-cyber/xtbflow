"""Route incomplete records into explicit training views.

The router is deliberately label-aware and answer-blind: one source record may
produce several independent task views, but an event/geometry joint view is
created only when the source explicitly attests that the two labels belong to
the same primitive step.  Missing labels are never filled, paired at random,
or represented as an ``all-zero`` target.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

from xtbflow.data.records import FORBIDDEN_INPUT_KEYS, canonical_hash


class TaskRoutingError(ValueError):
    """Raised when a record cannot be routed without inventing supervision."""


class TaskMode(str, Enum):
    EVENT_ONLY = "event_only"
    GEOMETRY_ONLY = "geometry_only"
    ENERGY_FORCE = "energy_force"
    PAIRED_JOINT = "paired_joint"


_LABEL_ALIASES: dict[str, frozenset[str]] = {
    "event": frozenset({"event", "event_features", "pair_edits", "event_label"}),
    "geometry": frozenset({
        "geometry", "ts_geometry", "geometry_coordinates", "geometry_velocity", "geometry_label",
    }),
    "energy": frozenset({"energy", "energy_label"}),
    "forces": frozenset({"forces", "force", "force_label"}),
}

# These fields are safe as targets but must never enter a reactant-only input.
_TARGET_DERIVED_FIELDS = frozenset().union(*_LABEL_ALIASES.values()) | FORBIDDEN_INPUT_KEYS


@dataclass(frozen=True)
class LabelAvailability:
    """Explicit evidence available for a record.

    ``paired_identity`` and ``same_geometry_e_f`` are attestations from the
    source adapter.  They are intentionally separate from label presence: two
    labels can both exist while belonging to different structures or steps.
    """

    event: bool = False
    geometry: bool = False
    energy: bool = False
    forces: bool = False
    paired_identity: bool = False
    same_geometry_e_f: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "LabelAvailability":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TaskRoutingError("label availability must be a mapping")
        aliases = {
            "event_geometry_pair": "paired_identity",
            "paired_event_geometry": "paired_identity",
            "energy_force_same_geometry": "same_geometry_e_f",
        }
        allowed = set(cls.__dataclass_fields__) | set(aliases)
        unknown = set(value) - allowed
        if unknown:
            raise TaskRoutingError(f"unknown label availability fields: {sorted(unknown)}")
        result: dict[str, bool] = {}
        for field in cls.__dataclass_fields__:
            supplied = value.get(field, False)
            for alias, canonical in aliases.items():
                if canonical == field and alias in value:
                    if field in value and value[field] != value[alias]:
                        raise TaskRoutingError(f"conflicting availability fields: {field}/{alias}")
                    supplied = value[alias]
            if type(supplied) is not bool:
                raise TaskRoutingError(f"availability field {field} must be boolean")
            result[field] = supplied
        return cls(**result)

    def as_dict(self) -> dict[str, bool]:
        return {
            "event": self.event,
            "geometry": self.geometry,
            "energy": self.energy,
            "forces": self.forces,
            "paired_identity": self.paired_identity,
            "same_geometry_e_f": self.same_geometry_e_f,
        }


@dataclass(frozen=True)
class TaskExample:
    """A source row with separate deployment inputs and training labels."""

    record_id: str
    inputs: Mapping[str, Any]
    targets: Mapping[str, Any]
    availability: LabelAvailability

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id.strip():
            raise TaskRoutingError("record_id must be a non-empty string")
        if not isinstance(self.inputs, Mapping) or not isinstance(self.targets, Mapping):
            raise TaskRoutingError("inputs and targets must be mappings")
        forbidden = _find_forbidden_keys(self.inputs)
        if forbidden:
            raise TaskRoutingError(
                "target-derived fields cannot enter a reactant input: " + ", ".join(forbidden)
            )
        # The mask is authoritative.  A target may be retained for provenance,
        # but it is not routed unless its availability bit is explicitly true.
        for group, available in (
            ("event", self.availability.event),
            ("geometry", self.availability.geometry),
            ("energy", self.availability.energy),
            ("forces", self.availability.forces),
        ):
            if available and not _has_label(self.targets, group):
                raise TaskRoutingError(f"availability claims {group}, but no {group} target is present")
        object.__setattr__(self, "inputs", deepcopy(dict(self.inputs)))
        object.__setattr__(self, "targets", deepcopy(dict(self.targets)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TaskExample":
        if not isinstance(value, Mapping):
            raise TaskRoutingError("task example must be a mapping")
        try:
            return cls(
                record_id=value["record_id"],
                inputs=value.get("inputs", {}),
                targets=value.get("targets", {}),
                availability=LabelAvailability.from_mapping(value.get("available_label_mask")),
            )
        except KeyError as exc:
            raise TaskRoutingError(f"task example missing field: {exc.args[0]}") from exc

    @property
    def input_fingerprint(self) -> str:
        return canonical_hash(self.inputs)


@dataclass(frozen=True)
class TaskView:
    """A loss-specific view with an explicit absence mask."""

    record_id: str
    mode: TaskMode
    inputs: Mapping[str, Any]
    targets: Mapping[str, Any]
    loss_mask: Mapping[str, bool]
    input_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.mode, TaskMode):
            raise TaskRoutingError("task view mode must be a TaskMode")
        if self.input_fingerprint != canonical_hash(self.inputs):
            raise TaskRoutingError("task view input fingerprint does not match inputs")
        if _find_forbidden_keys(self.inputs):
            raise TaskRoutingError("task view contains target-derived input fields")
        expected = {
            TaskMode.EVENT_ONLY: {"event": True, "geometry": False, "energy": False, "forces": False},
            TaskMode.GEOMETRY_ONLY: {"event": False, "geometry": True, "energy": False, "forces": False},
            TaskMode.ENERGY_FORCE: {"event": False, "geometry": False, "energy": True, "forces": True},
            TaskMode.PAIRED_JOINT: {"event": True, "geometry": True, "energy": False, "forces": False},
        }[self.mode]
        if dict(self.loss_mask) != expected:
            raise TaskRoutingError(f"loss mask for {self.mode.value} must be {expected}")


def _find_forbidden_keys(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            if key_text in _TARGET_DERIVED_FIELDS:
                found.append(".".join((*path, key_text)))
            found.extend(_find_forbidden_keys(nested, (*path, key_text)))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            found.extend(_find_forbidden_keys(nested, (*path, str(index))))
    return sorted(set(found))


def _has_label(targets: Mapping[str, Any], group: str) -> bool:
    return any(key in targets for key in _LABEL_ALIASES[group])


def _target_fields(targets: Mapping[str, Any], groups: Iterable[str]) -> dict[str, Any]:
    aliases = set().union(*(_LABEL_ALIASES[group] for group in groups))
    return deepcopy({key: targets[key] for key in targets if key in aliases})


def _view(example: TaskExample, mode: TaskMode, groups: tuple[str, ...]) -> TaskView:
    mask = {"event": False, "geometry": False, "energy": False, "forces": False}
    mask.update({group: True for group in groups})
    return TaskView(
        record_id=example.record_id,
        mode=mode,
        inputs=deepcopy(dict(example.inputs)),
        targets=_target_fields(example.targets, groups),
        loss_mask=mask,
        input_fingerprint=example.input_fingerprint,
    )


def views_for_example(example: TaskExample) -> tuple[TaskView, ...]:
    """Return every independently supported view without synthetic pairing."""

    views: list[TaskView] = []
    if example.availability.event:
        views.append(_view(example, TaskMode.EVENT_ONLY, ("event",)))
    if example.availability.geometry:
        views.append(_view(example, TaskMode.GEOMETRY_ONLY, ("geometry",)))
    if example.availability.energy and example.availability.forces and example.availability.same_geometry_e_f:
        views.append(_view(example, TaskMode.ENERGY_FORCE, ("energy", "forces")))
    if example.availability.event and example.availability.geometry and example.availability.paired_identity:
        views.append(_view(example, TaskMode.PAIRED_JOINT, ("event", "geometry")))
    return tuple(views)


def route_examples(examples: Iterable[TaskExample]) -> dict[TaskMode, tuple[TaskView, ...]]:
    """Group task views deterministically by mode and record ID."""

    grouped: dict[TaskMode, list[TaskView]] = {mode: [] for mode in TaskMode}
    for example in examples:
        for view in views_for_example(example):
            grouped[view.mode].append(view)
    return {
        mode: tuple(sorted(views, key=lambda item: (item.record_id, item.input_fingerprint)))
        for mode, views in grouped.items()
    }
