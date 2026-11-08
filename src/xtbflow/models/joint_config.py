"""Runtime configuration for the bounded joint-flow software prototype."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .conserved_be import ConservationProjector
from .joint_flow import CONTROL_MODES, JointEventGeometryFlow


RUNTIME_CONFIG_SCHEMA = "xtbflow-joint-runtime/v1"


def joint_flow_parameter_count(
    node_feature_dim: int,
    condition_feature_dim: int,
    hidden_dim: int,
    radial_features: int,
) -> int:
    """Return the trainable parameter count for the current joint-flow graph.

    The count is computed from constructor dimensions rather than from a
    checkpoint.  It is used by the scale manifest to make equal-capacity
    comparisons auditable before a training job is launched.  Buffers such as
    the conservation matrix and radial basis centers are intentionally not
    included.
    """

    values = {
        "node_feature_dim": node_feature_dim,
        "condition_feature_dim": condition_feature_dim,
        "hidden_dim": hidden_dim,
        "radial_features": radial_features,
    }
    if any(type(value) is not int for value in values.values()):
        raise ValueError("joint-flow dimensions must be strict integers")
    if any(values[name] < 1 for name in ("node_feature_dim", "hidden_dim", "radial_features")):
        raise ValueError("node, hidden, and radial dimensions must be positive")
    if condition_feature_dim < 0:
        raise ValueError("condition_feature_dim must be nonnegative")

    n = node_feature_dim
    c = condition_feature_dim
    h = hidden_dim
    r = radial_features

    # Linear layers are counted as out * in + out.  The condition encoders
    # are omitted when no condition features are configured.
    count = (n + 1) * h  # node_encoder
    count += h * h + 3 * h  # time_to_hidden
    count += (n + 2) * h + n  # time_to_node
    if c:
        count += h * h + (c + 2) * h  # condition_to_hidden
        # condition_to_node has two linear layers: c -> h and h -> n.
        # The first layer contributes one hidden-width bias and the second
        # contributes one node-width bias, so the hidden-width coefficient is
        # ``c + n + 1``.
        count += (c + n + 1) * h + n  # condition_to_node
    count += 2 * h * h + 4 * h + 1  # event_base
    count += (r + 2) * h + 1  # geometry_to_event
    count += (n + 3) * h + n  # event_to_node
    count += (2 * n + r + 3) * h + 1  # geometry.message
    return int(count)


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

    @property
    def trainable_parameter_count(self) -> int:
        """Expected trainable parameter count for this constructor state."""

        return joint_flow_parameter_count(
            self.node_feature_dim,
            self.condition_feature_dim,
            self.hidden_dim,
            self.radial_features,
        )

    def with_hidden_dim(self, hidden_dim: int) -> "JointFlowRuntimeConfig":
        """Return this runtime contract at another declared capacity scale."""

        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer")
        return JointFlowRuntimeConfig(
            model_id=self.model_id,
            status=self.status,
            node_feature_dim=self.node_feature_dim,
            condition_feature_dim=self.condition_feature_dim,
            hidden_dim=hidden_dim,
            radial_features=self.radial_features,
            control_modes=self.control_modes,
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
            "trainable_parameter_count": self.trainable_parameter_count,
            "control_modes": list(self.control_modes),
        }
