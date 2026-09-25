"""Contract tests for the unified event--geometry sample manifest.

These tests exercise the semantic checks implemented in
``mechai.data.sample_manifest`` in addition to the structural JSON schema.
The fixtures are intentionally illustrative records: passing validation means
that the provenance and task fields are internally consistent, not that the
sample has passed a scientific training or evaluation gate.
"""
from __future__ import annotations

import json
from pathlib import Path
import unittest

from mechai.data.sample_manifest import ManifestValidationError, validate_manifest


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = ROOT / "configs" / "reusable" / "event_geometry_manifest_v1"


def _load_example(name: str) -> dict:
    with (EXAMPLE_DIR / name).open(encoding="utf-8") as handle:
        return json.load(handle)


class SampleManifestValidationTests(unittest.TestCase):
    def test_all_task_mode_examples_validate(self):
        """Each task mode has a self-consistent illustrative manifest."""
        examples = sorted(EXAMPLE_DIR.glob("*.json"))
        self.assertEqual(
            {path.name for path in examples},
            {
                "endpoint_geometry.json",
                "intent_event_geometry.json",
                "endpoint_microsequence_geometry.json",
            },
        )
        for path in examples:
            with self.subTest(path=path.name):
                validate_manifest(_load_example(path.name))

    def test_known_and_unknown_variables_must_be_disjoint(self):
        manifest = _load_example("endpoint_geometry.json")
        manifest["variables"]["known"].append("ts_geometry")

        with self.assertRaisesRegex(ManifestValidationError, "known and unknown overlap"):
            validate_manifest(manifest)

    def test_prediction_target_must_be_declared_unknown(self):
        manifest = _load_example("endpoint_geometry.json")
        manifest["variables"]["unknown"].remove("ts_geometry")

        with self.assertRaisesRegex(
            ManifestValidationError,
            "every prediction target must be listed as unknown",
        ):
            validate_manifest(manifest)

    def test_each_task_mode_requires_its_declared_targets(self):
        cases = (
            ("endpoint_geometry.json", "ts_geometry", "endpoint_geometry requires ts_geometry"),
            (
                "intent_event_geometry.json",
                "event_identity",
                "intent_event_geometry requires event_identity",
            ),
            (
                "endpoint_microsequence_geometry.json",
                "event_order",
                "endpoint_microsequence_geometry requires event_order",
            ),
        )
        for filename, target, message in cases:
            with self.subTest(filename=filename, target=target):
                manifest = _load_example(filename)
                manifest["variables"]["prediction_targets"].remove(target)
                with self.assertRaisesRegex(ManifestValidationError, message):
                    validate_manifest(manifest)

    def test_intent_event_mode_cannot_claim_full_mapped_product_permission(self):
        manifest = _load_example("intent_event_geometry.json")
        manifest["condition"]["known_condition"]["permission_mode"] = "full_mapped_product"

        with self.assertRaisesRegex(
            ManifestValidationError,
            "intent_event_geometry cannot claim full mapped product",
        ):
            validate_manifest(manifest)

    def test_independent_evaluation_requires_known_family_evidence(self):
        manifest = _load_example("endpoint_geometry.json")
        # The fixture deliberately leaves family_evidence unknown.  A passing
        # independent-evaluation gate would therefore overstate split quality.
        manifest["gate"]["independent_evaluation"] = "pass"

        with self.assertRaisesRegex(
            ManifestValidationError,
            "cannot pass while source.family_evidence is unknown/ambiguous",
        ):
            validate_manifest(manifest)

    def test_atom_metadata_arrays_must_match_atomic_number_length(self):
        manifest = _load_example("endpoint_geometry.json")
        manifest["atoms"]["isotopes"] = manifest["atoms"]["isotopes"][:-1]

        with self.assertRaisesRegex(
            ManifestValidationError,
            r"\$\.atoms\.isotopes.*same length as atomic_numbers",
        ):
            validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
