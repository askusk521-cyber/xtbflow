"""Immutable, variable-cardinality event actions for policy adapters."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable


@dataclass(frozen=True, order=True)
class EventEdit:
    operation: str
    left_map_id: int
    right_map_id: int

    def __post_init__(self) -> None:
        if self.operation not in {"add", "delete"}:
            raise ValueError("event edit operation must be add or delete")
        if type(self.left_map_id) is not int or type(self.right_map_id) is not int:
            raise ValueError("event edit map IDs must be integers")
        if self.left_map_id <= 0 or self.right_map_id <= 0 or self.left_map_id == self.right_map_id:
            raise ValueError("event edit map IDs must be distinct positive integers")
        if self.left_map_id > self.right_map_id:
            raise ValueError("event edit map IDs must be canonicalized ascending")

    @property
    def key(self) -> str:
        return f"{self.operation}:{self.left_map_id}-{self.right_map_id}"

    def to_mapping(self) -> dict[str, Any]:
        return {"operation": self.operation, "left_map_id": self.left_map_id, "right_map_id": self.right_map_id}


@dataclass(frozen=True)
class EventAction:
    """A complete event edit set; cardinality is not limited to one action."""

    edits: tuple[EventEdit, ...] = ()
    action_id: str = "event-action-v1"

    def __post_init__(self) -> None:
        edits = tuple(self.edits)
        if any(not isinstance(edit, EventEdit) for edit in edits):
            raise ValueError("edits must contain EventEdit values")
        if len({edit.key for edit in edits}) != len(edits):
            raise ValueError("event action cannot contain duplicate edits")
        if not isinstance(self.action_id, str) or not self.action_id.strip():
            raise ValueError("action_id must be non-empty")
        object.__setattr__(self, "edits", tuple(sorted(edits)))

    @classmethod
    def from_mappings(cls, values: Iterable[dict[str, Any]], *, action_id: str = "event-action-v1") -> "EventAction":
        return cls(tuple(EventEdit(str(value["operation"]), int(value["left_map_id"]), int(value["right_map_id"])) for value in values), action_id=action_id)

    @property
    def fingerprint(self) -> str:
        payload = {"action_id": self.action_id, "edits": [edit.to_mapping() for edit in self.edits]}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def to_mapping(self) -> dict[str, Any]:
        return {"action_id": self.action_id, "edits": [edit.to_mapping() for edit in self.edits], "fingerprint": self.fingerprint}


__all__ = ["EventEdit", "EventAction"]
