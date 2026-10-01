"""Unified factory and auditable contracts for the W02 control directory."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .conserved_be import ConservationProjector
from .initial_geometry import BondEditHint, RuleGeometryInitializer
from .joint_flow import CONTROL_MODES, JointEventGeometryFlow, JointFlowOutput
from .serial_baseline import FullTwoStageFlow, SeparateIndependentFlow


BASELINE_ARM_IDS = (
    "strong_rule",
    "shared_no_exchange",
    "step_serial",
    "current_one_way",
    "current_two_way",
    "current_two_way_no_projection",
    "separate_independent",
    "full_two_stage",
)
LEARNED_ARM_IDS = frozenset(BASELINE_ARM_IDS[1:])


@dataclass(frozen=True)
class BaselineSpec:
    arm_id: str
    kind: str
    experiment_type: str
    state_dependencies: str
    conservation_projection: bool
    default_dt: float
    raw_candidate_cap: int = 64
    solver_steps: int = 1

    def __post_init__(self) -> None:
        if self.arm_id not in BASELINE_ARM_IDS:
            raise ValueError(f"unknown baseline arm: {self.arm_id}")
        if self.kind not in {"rule", "learned"}:
            raise ValueError("baseline kind must be rule or learned")
        if self.experiment_type not in {"method_comparison", "mechanism_intervention"}:
            raise ValueError("unsupported baseline experiment type")
        if self.default_dt <= 0 or self.raw_candidate_cap < 1 or self.solver_steps < 1:
            raise ValueError("baseline resource fields must be positive")


class StrongRuleControl:
    """Rule-only control with deterministic, bounded geometry guesses."""

    def __init__(self, *, initializer: RuleGeometryInitializer | None = None, raw_candidate_cap: int = 64) -> None:
        if type(raw_candidate_cap) is not int or raw_candidate_cap < 1:
            raise ValueError("raw_candidate_cap must be positive")
        self.initializer = initializer or RuleGeometryInitializer()
        self.raw_candidate_cap = raw_candidate_cap
        self.arm_id = "strong_rule"

    @property
    def parameter_count(self) -> int:
        return 0

    def sample(
        self,
        coordinates: Tensor,
        atom_mask: Tensor,
        *,
        bond_edits: tuple[BondEditHint | tuple[int, int, float], ...] = (),
        candidate_cap: int | None = None,
    ) -> tuple[dict[str, Any], ...]:
        cap = self.raw_candidate_cap if candidate_cap is None else candidate_cap
        if type(cap) is not int or cap < 1 or cap > self.raw_candidate_cap:
            raise ValueError("candidate_cap must be within the frozen raw candidate cap")
        if not bond_edits:
            return (
                {
                    "arm_id": self.arm_id,
                    "raw_slot_id": 0,
                    "event": None,
                    "geometry": coordinates.detach().clone(),
                    "status": "unsupported",
                    "failure_reason": "no_reactant_side_rule_event",
                    "calculator_calls": 0,
                },
            )
        geometry = self.initializer(coordinates, atom_mask, bond_edits)
        return (
            {
                "arm_id": self.arm_id,
                "raw_slot_id": 0,
                "event": {"bond_edits": [
                    (edit.i, edit.j, edit.delta) if isinstance(edit, BondEditHint) else tuple(edit)
                    for edit in bond_edits
                ]},
                "geometry": geometry,
                "status": "proposed",
                "failure_reason": None,
                "calculator_calls": 0,
            },
        )

    def contract(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "kind": "rule",
            "state_dependencies": "reactant_only",
            "target_fields_read": [],
            "calculator_calls": 0,
            "raw_candidate_cap": self.raw_candidate_cap,
        }


class BaselineControl(nn.Module):
    """Adapter exposing one frozen arm through a common forward signature."""

    def __init__(self, spec: BaselineSpec, module: nn.Module | None = None) -> None:
        super().__init__()
        self.spec = spec
        if module is not None:
            self.module = module

    @property
    def arm_id(self) -> str:
        return self.spec.arm_id

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(
        self,
        event_state: Tensor,
        coordinates: Tensor,
        node_features: Tensor,
        atom_mask: Tensor,
        *,
        tau: float | Tensor,
        dt: float | None = None,
        coupling_strength: float = 1.0,
        condition_features: Tensor | None = None,
        conservation_projection: bool | None = None,
    ) -> JointFlowOutput:
        if not hasattr(self, "module"):
            raise TypeError("strong_rule is sampled through StrongRuleControl.sample")
        projection = self.spec.conservation_projection if conservation_projection is None else conservation_projection
        step = self.spec.default_dt if dt is None else dt
        if self.arm_id == "separate_independent":
            return self.module(
                event_state,
                coordinates,
                node_features,
                atom_mask,
                tau=tau,
                condition_features=condition_features,
                conservation_projection=projection,
            )
        if self.arm_id == "full_two_stage":
            return self.module(
                event_state,
                coordinates,
                node_features,
                atom_mask,
                tau=tau,
                dt=step,
                coupling_strength=coupling_strength,
                condition_features=condition_features,
                conservation_projection=projection,
            )
        mode = {
            "shared_no_exchange": "both_off",
            "step_serial": "serial_independent",
            "current_one_way": "joint_unidirectional",
            "current_two_way": "joint_bidirectional",
            "current_two_way_no_projection": "joint_bidirectional",
        }[self.arm_id]
        return self.module.forward_control(
            mode,
            event_state,
            coordinates,
            node_features,
            atom_mask,
            tau=tau,
            dt=step if mode in {"serial_independent", "joint_unidirectional"} else None,
            coupling_strength=coupling_strength,
            condition_features=condition_features,
            conservation_projection=projection,
        )

    def state_identity(self) -> dict[str, Any]:
        state_hash = hashlib.sha256()
        if hasattr(self, "module"):
            for name, value in sorted(self.state_dict().items()):
                state_hash.update(name.encode())
                state_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
        return {
            "arm_id": self.arm_id,
            "trainable_parameters": self.parameter_count,
            "state_hash": state_hash.hexdigest(),
        }

    def contract(self) -> dict[str, Any]:
        return {
            "arm_id": self.spec.arm_id,
            "kind": self.spec.kind,
            "experiment_type": self.spec.experiment_type,
            "state_dependencies": self.spec.state_dependencies,
            "conservation_projection": self.spec.conservation_projection,
            "raw_candidate_cap": self.spec.raw_candidate_cap,
            "solver_steps": self.spec.solver_steps,
            "field_evaluations": {
                "event": self.spec.solver_steps,
                "geometry": self.spec.solver_steps,
            },
            "trainable_parameters": self.parameter_count,
        }


class BaselineFactory:
    """Build each arm with fresh parameters and explicit state semantics."""

    def __init__(
        self,
        event_projector: ConservationProjector,
        node_feature_dim: int,
        *,
        condition_feature_dim: int = 0,
        hidden_dim: int = 64,
        radial_features: int = 16,
        default_dt: float = 0.1,
        raw_candidate_cap: int = 64,
    ) -> None:
        if default_dt <= 0 or raw_candidate_cap < 1:
            raise ValueError("default_dt and raw_candidate_cap must be positive")
        self.event_projector = event_projector
        self.node_feature_dim = node_feature_dim
        self.condition_feature_dim = condition_feature_dim
        self.hidden_dim = hidden_dim
        self.radial_features = radial_features
        self.default_dt = float(default_dt)
        self.raw_candidate_cap = raw_candidate_cap

    def _joint(self) -> JointEventGeometryFlow:
        return JointEventGeometryFlow(
            self.event_projector,
            self.node_feature_dim,
            condition_feature_dim=self.condition_feature_dim,
            hidden_dim=self.hidden_dim,
            radial_features=self.radial_features,
        )

    def spec(self, arm_id: str) -> BaselineSpec:
        if arm_id not in BASELINE_ARM_IDS:
            raise ValueError(f"unknown baseline arm: {arm_id}")
        if arm_id == "strong_rule":
            return BaselineSpec(arm_id, "rule", "method_comparison", "reactant_only", True, self.default_dt, self.raw_candidate_cap, 0)
        if arm_id == "shared_no_exchange":
            return BaselineSpec(arm_id, "learned", "method_comparison", "reactant_condition_only", True, self.default_dt, self.raw_candidate_cap)
        if arm_id == "step_serial":
            return BaselineSpec(arm_id, "learned", "method_comparison", "event_then_geometry", True, self.default_dt, self.raw_candidate_cap)
        if arm_id == "current_one_way":
            return BaselineSpec(arm_id, "learned", "method_comparison", "current_event_only_for_geometry", True, self.default_dt, self.raw_candidate_cap)
        if arm_id == "current_two_way":
            return BaselineSpec(arm_id, "learned", "method_comparison", "current_bidirectional", True, self.default_dt, self.raw_candidate_cap)
        if arm_id == "current_two_way_no_projection":
            return BaselineSpec(arm_id, "learned", "mechanism_intervention", "current_bidirectional", False, self.default_dt, self.raw_candidate_cap)
        if arm_id == "separate_independent":
            return BaselineSpec(arm_id, "learned", "method_comparison", "two_independent_branches", True, self.default_dt, self.raw_candidate_cap)
        return BaselineSpec(arm_id, "learned", "method_comparison", "complete_event_then_frozen_geometry", True, self.default_dt, self.raw_candidate_cap, 1)

    def build(self, arm_id: str, *, seed: int | None = None) -> BaselineControl | StrongRuleControl:
        spec = self.spec(arm_id)
        rng = torch.random.fork_rng(devices=[]) if seed is not None else nullcontext()
        with rng:
            if seed is not None:
                torch.manual_seed(seed)
            if arm_id == "strong_rule":
                return StrongRuleControl(raw_candidate_cap=self.raw_candidate_cap)
            if arm_id == "separate_independent":
                module: nn.Module = SeparateIndependentFlow(self._joint(), self._joint())
            elif arm_id == "full_two_stage":
                module = FullTwoStageFlow(self._joint(), self._joint())
            else:
                module = self._joint()
        return BaselineControl(spec, module)

    def build_all(self, *, seed: int | None = None) -> dict[str, BaselineControl | StrongRuleControl]:
        return {arm_id: self.build(arm_id, seed=seed) for arm_id in BASELINE_ARM_IDS}

    def manifest(self, *, seed: int | None = None) -> dict[str, Any]:
        controls = self.build_all(seed=seed)
        records: list[dict[str, Any]] = []
        reference_parameters: int | None = None
        for arm_id in BASELINE_ARM_IDS:
            control = controls[arm_id]
            record = control.contract()
            record["identity"] = control.contract() | (control.state_identity() if isinstance(control, BaselineControl) else control.contract())
            if arm_id == "current_two_way":
                reference_parameters = int(record["trainable_parameters"])
            records.append(record)
        for record in records:
            params = int(record["trainable_parameters"])
            record["capacity_report"] = {
                "reference_arm": "current_two_way",
                "reference_trainable_parameters": reference_parameters,
                "trainable_parameter_ratio": None if not reference_parameters else params / reference_parameters,
                "within_ten_percent": None if not reference_parameters else abs(params - reference_parameters) / reference_parameters <= 0.10,
            }
        return {
            "schema": "xtbflow-baseline-controls/v1",
            "arm_ids": list(BASELINE_ARM_IDS),
            "seed": seed,
            "separately_trained_methods": list(BASELINE_ARM_IDS),
            "inference_interventions": ["current_two_way_no_projection"],
            "controls": records,
            "resource_contract": {
                "raw_candidate_cap": self.raw_candidate_cap,
                "candidate_refill": False,
                "rejected_slots_count_toward_cap": True,
                "physical_calls": 0,
            },
        }


def build_baseline(factory: BaselineFactory, arm_id: str, *, seed: int | None = None) -> BaselineControl | StrongRuleControl:
    """Functional alias used by runners and tests."""

    return factory.build(arm_id, seed=seed)


def build_control(factory: BaselineFactory, arm_id: str, *, seed: int | None = None) -> BaselineControl | StrongRuleControl:
    return factory.build(arm_id, seed=seed)
