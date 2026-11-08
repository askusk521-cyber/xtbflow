from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_research_dispatch import (  # noqa: E402
    DispatchValidationError,
    load_config,
    validate_dispatch,
)


CONFIG = ROOT / "configs/runs/research_dispatch_20261001.json"


class ResearchDispatchContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = load_config(CONFIG)

    def test_offline_manifest_is_valid_but_science_is_not_ready(self) -> None:
        report = validate_dispatch(self.payload, offline=True)
        self.assertTrue(report["valid"])
        self.assertFalse(report["science_ready"])
        self.assertEqual(report["work_package_count"], 12)
        self.assertEqual(report["artifact_count"], 20)
        self.assertEqual(report["dependency_count"], 34)
        self.assertEqual(report["remote_currentity"], "unknown_offline")
        self.assertIn("W01.paired_manifest", report["blocked_artifacts"])
        self.assertIn("paired_manifest_to_s1", report["unmet_dependency_ids"])

    def test_cycle_is_rejected_at_artifact_level(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["dependencies"].append(
            {
                "dependency_id": "intentional_cycle",
                "from_artifact": "W04.s1_results",
                "to_artifact": "W04.s1_plan",
                "requirement": "test cycle",
                "required_status": "implemented",
            }
        )
        with self.assertRaises(DispatchValidationError) as context:
            validate_dispatch(payload, offline=True)
        self.assertTrue(any("cycle" in message for message in context.exception.errors))

    def test_same_name_with_different_hash_is_rejected(self) -> None:
        payload = copy.deepcopy(self.payload)
        duplicate = copy.deepcopy(payload["artifacts"][0])
        duplicate["artifact_id"] = "W00.dispatch_contract_conflict"
        duplicate["artifact_hash"] = "f" * 64
        payload["artifacts"].append(duplicate)
        with self.assertRaises(DispatchValidationError) as context:
            validate_dispatch(payload, offline=True)
        self.assertTrue(any("conflicting hashes" in message for message in context.exception.errors))

    def test_unknown_cannot_be_marked_ready_or_passed(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["artifacts"][0]["status"] = "unknown"
        payload["artifacts"][0]["ready_software"] = True
        with self.assertRaises(DispatchValidationError) as context:
            validate_dispatch(payload, offline=True)
        self.assertTrue(any("ready_software" in message for message in context.exception.errors))

        payload = copy.deepcopy(self.payload)
        payload["artifacts"][0]["status"] = "passed"
        with self.assertRaises(DispatchValidationError) as context:
            validate_dispatch(payload, offline=True)
        self.assertTrue(any("not a dispatch status" in message or "status must be one of" in message for message in context.exception.errors))

    def test_offline_mode_rejects_a_live_remote_claim(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["verification"]["remote_checked"] = True
        with self.assertRaises(DispatchValidationError) as context:
            validate_dispatch(payload, offline=True)
        self.assertTrue(any("remote" in message for message in context.exception.errors))

    def test_required_handoff_fields_and_typed_paths_are_checked(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["handoff_fields"] = payload["handoff_fields"][:-1]
        with self.assertRaises(DispatchValidationError):
            validate_dispatch(payload, offline=True)

        payload = copy.deepcopy(self.payload)
        payload["artifacts"][0]["path"] = "/tmp/dispatch.json"
        with self.assertRaises(DispatchValidationError) as context:
            validate_dispatch(payload, offline=True)
        self.assertTrue(any("stay inside the repository" in message for message in context.exception.errors))

    def test_json_file_is_machine_readable_without_placeholder_pass(self) -> None:
        text = CONFIG.read_text(encoding="utf-8")
        payload = json.loads(text)
        self.assertEqual(payload["schema"], "xtbflow-research-dispatch/v1")
        self.assertEqual(payload["status"], "interface_contract_frozen_science_incomplete")
        self.assertNotIn('"passed"', text)


if __name__ == "__main__":
    unittest.main()
