"""Runtime configuration for the bounded joint-flow software prototype."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .conserved_be import ConservationProjector
from .joint_flow import CONTROL_MODES, JointEventGeometryFlow


RUNTIME_CONFIG_SCHEMA = "xtbflow-joint-runtime/v1"


@dataclass(frozen=True)
class JointFlowRuntimeConfig:
    """Exact constructor settings separated from the broader research contract."""

    model_id: str
    status: str
    node_feature_dim: int
    condition_feature_dim: int
    hidden_dim: int
    radial_features: int
    control_modes: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "JointFlowRuntimeConfig":
        if value.get("schema") != RUNTIME_CONFIG_SCHEMA:
            raise ValueError("unsupported joint runtime config schema")
        representation = value.get("representation")
        if not isinstance(representation, Mapping):
            raise ValueError("joint runtime config requires a representation mapping")
        modes = tuple(value.get("control_modes", ()))
        if not modes or len(set(modes)) != len(modes) or any(mode not in CONTROL_MODES for mode in modes):
            raise ValueError("joint runtime config has invalid control modes")
        integers = {
            "node_feature_dim": representation.get("node_feature_dim"),
            "condition_feature_dim": representation.get("condition_feature_dim"),
            "hidden_dim": representation.get("hidden_dim"),
            "radial_features": representation.get("radial_features"),
        }
        if any(type(item) is not int for item in integers.values()):
            raise ValueError("joint runtime feature widths must be integers")
        if integers["condition_feature_dim"] < 0 or any(
            integers[name] < 1 for name in ("node_feature_dim", "hidden_dim", "radial_features")
        ):
            raise ValueError("joint runtime feature widths are outside their valid range")
        model_id = value.get("model_id")
        status = value.get("status")
        if not isinstance(model_id, str) or not model_id or not isinstance(status, str) or not status:
            raise ValueError("joint runtime config requires model_id and status")
        return cls(model_id=model_id, status=status, control_modes=modes, **integers)

    @classmethod
    def load(cls, path: str | Path) -> "JointFlowRuntimeConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("joint runtime config root must be an object")
        return cls.from_mapping(payload)

    def build(self, projector: ConservationProjector) -> JointEventGeometryFlow:
        return JointEventGeometryFlow(
            projector,
            self.node_feature_dim,
            condition_feature_dim=self.condition_feature_dim,
            hidden_dim=self.hidden_dim,
            radial_features=self.radial_features,
        )

    def constructor_record(self) -> dict[str, object]:
        return {
            "class": "xtbflow.models.JointEventGeometryFlow",
            "model_id": self.model_id,
            "status": self.status,
            "node_feature_dim": self.node_feature_dim,
            "condition_feature_dim": self.condition_feature_dim,
            "hidden_dim": self.hidden_dim,
            "radial_features": self.radial_features,
            "control_modes": list(self.control_modes),
        }
