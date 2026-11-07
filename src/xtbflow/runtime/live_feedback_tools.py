"""Scientific tool adapter: observed reference evidence followed by real generation."""
from __future__ import annotations

import json
from pathlib import Path

from xtbflow.governance.generation_bridge import validate_query
from xtbflow.runtime.feedback_loop import CachedReferenceTools, GENERATION_ACTIONS, candidate_view
from xtbflow.runtime.generation_actions import GenerationActions
from xtbflow.validation.certification import CertificationCandidate, file_hash, identity


class LiveGenerationTools(CachedReferenceTools):
    def __init__(self, attempts: dict[str, Path], pins: dict[str, str], *, query: dict,
                 reference_protocol_sha256: str, generator: GenerationActions):
        super().__init__(attempts, pins)
        validate_query(query)
        self.query, self.protocol, self.generator = query, reference_protocol_sha256, generator
        self.pool = candidate_view(validate_query(query), self.protocol)["pool_sha256"]

    @property
    def identity(self) -> dict:
        return {**super().identity, "profile": "reference_feedback_generation",
                "query_sha256": identity(self.query), "generator": self.generator.identity,
                "adapter_sha256": file_hash(Path(__file__))}

    def execute(self, action: dict) -> dict:
        if action["name"] not in GENERATION_ACTIONS:
            result = super().execute(action)
            result["tool_profile"] = "reference_feedback_generation"
            if result["observation"] and result["observation"]["candidate"]["pool_sha256"] != self.pool:
                raise ValueError("reference feedback and reactant packet have different fixed pools")
            return result
        condition = None
        if action["name"] == "resample_geometry":
            work = self.attempts[action["candidate_id"]]
            # Reverify before reading the generated hypothesis. No actual product,
            # optimized TS coordinates or reference geometry is sent to the model.
            check = {**action, "name": "validate_connectivity"}
            verified = super().execute(check)["observation"]
            request = json.loads((work / "request.json").read_text())
            c = CertificationCandidate(**request["candidate"])
            if verified["candidate"]["pool_sha256"] != self.pool or identity(c.content()) != verified["candidate"]["candidate_sha256"]:
                raise ValueError("conditioned event candidate identity mismatch")
            condition = c.planned_product_be
        record = self.generator.run(action, self.query, condition)
        candidates = [candidate_view(CertificationCandidate(**c), self.protocol) for c in record["candidates"]]
        return dict(schema="xtbflow-feedback-tool-result/v1", action_id=action["action_id"],
            tool_profile="reference_feedback_generation", new_calculator_calls=0, status="completed", observation=None,
            generation=dict(action_id=record["action_id"], status=record["status"], cost=record["cost"],
                cost_basis=record["cost_basis"], counter_basis=record["counter_basis"], candidates=candidates, result_sha256=record["content_sha256"],
                generation_report_sha256=record["generation_report_sha256"],
                certification_plan_sha256=record["certification_plan_sha256"],
                service_directory=record["action_id"], generation_directory=record["generation_directory"],
                generation_config=record["generation_config"]))
