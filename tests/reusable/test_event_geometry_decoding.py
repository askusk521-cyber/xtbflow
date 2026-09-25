"""CPU checks of output budgets and label-free ranking, not chemical success."""
from dataclasses import asdict, replace
import inspect
import math
import unittest

import torch

from mechai.evaluation.event_geometry import decode_event_geometry
from mechai.models.event_geometry import CompatibilityScores


class EventGeometryDecodingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    @staticmethod
    def scores(values, valid=None, oracle=False):
        values = torch.as_tensor(values, dtype=torch.float64)
        valid = torch.ones_like(values, dtype=torch.bool) if valid is None else torch.as_tensor(valid, dtype=torch.bool)
        return CompatibilityScores(values, valid, oracle)

    @staticmethod
    def ids(decoded):
        return [(pair.event_id, pair.geometry_id) for pair in decoded.pairs]

    def test_full_grid_emits_only_one_pair_per_geometry_and_prefix_budgets(self):
        values = torch.zeros(1, 32, 8, dtype=torch.float64)
        for geometry in range(8):
            values[0, 31 - geometry, geometry] = geometry + 1
        scores = self.scores(values)
        prior = torch.zeros(1, 32, dtype=torch.float64)
        full, = decode_event_geometry(scores, prior, coupling_strength=1)
        self.assertEqual(self.ids(full), [(31 - geometry, geometry) for geometry in reversed(range(8))])
        self.assertEqual(len({pair.geometry_id for pair in full.pairs}), 8)
        self.assertEqual((full.supplied_event_count, full.supplied_geometry_count), (32, 8))
        self.assertEqual((full.eligible_geometry_count, full.unused_slots, full.status), (8, 0, "complete"))
        for k in (1, 4):
            partial, = decode_event_geometry(scores, prior, coupling_strength=1, k=k)
            self.assertEqual(partial.pairs, full.pairs[:k])
            self.assertEqual(partial.eligible_geometry_count, 8)
            self.assertEqual(partial.unused_slots, 0)

    def test_ties_choose_smallest_event_then_sort_event_and_geometry(self):
        scores = self.scores([[[0, 3, 3], [3, 2, 3]]])
        decoded, = decode_event_geometry(scores, torch.zeros(1, 2, dtype=torch.float64), coupling_strength=1)
        self.assertEqual(self.ids(decoded), [(0, 1), (0, 2), (1, 0)])
        self.assertTrue(all(pair.terminal_score == 3 - math.log(2) for pair in decoded.pairs))
        self.assertEqual((decoded.unused_slots, decoded.status), (5, "partial"))

    def test_combined_score_ranks_geometries_not_event_normalized_posterior(self):
        # Geometry 1 has a larger maximum posterior, but geometry 0 has the
        # larger terminal score. Posterior ranking would invert the result.
        scores = self.scores([[[100, 0], [99, -100]]])
        posterior = torch.softmax(scores.values[0], dim=0)
        self.assertGreater(float(posterior[:, 1].max()), float(posterior[:, 0].max()))
        decoded, = decode_event_geometry(scores, torch.zeros(1, 2, dtype=torch.float64), coupling_strength=1)
        self.assertEqual(self.ids(decoded), [(0, 0), (0, 1)])
        self.assertAlmostEqual(decoded.pairs[0].terminal_score, 100 - math.log(2))

    def test_prior_is_normalized_once_across_input_events(self):
        scores = self.scores([[[0, float("nan")], [float("nan"), 0]]],
                             [[[True, False], [False, True]]])
        prior = torch.tensor([[math.log(.1), math.log(.9)]], dtype=torch.float64)
        decoded, = decode_event_geometry(scores, prior, coupling_strength=1)
        self.assertEqual(self.ids(decoded), [(1, 1), (0, 0)])
        self.assertAlmostEqual(decoded.pairs[0].terminal_score, math.log(.9))
        self.assertAlmostEqual(decoded.pairs[1].terminal_score, math.log(.1))

    def test_zero_strength_bypasses_nonfinite_compatibility(self):
        scores = self.scores([[[float("nan"), float("inf")], [float("-inf"), float("nan")]]])
        prior = torch.tensor([[2., 1.]], dtype=torch.float64)
        decoded, = decode_event_geometry(scores, prior)
        self.assertEqual(self.ids(decoded), [(0, 0), (0, 1)])
        self.assertAlmostEqual(decoded.pairs[0].terminal_score, -math.log1p(math.exp(-1)))
        with self.assertRaisesRegex(ValueError, "compatibility scores must be finite"):
            decode_event_geometry(scores, prior, coupling_strength=.1)

    def test_zero_prior_support_and_nan_padding_are_excluded_without_renumbering(self):
        scores = self.scores([[[float("nan"), float("nan"), float("nan")],
                              [float("nan"), float("nan"), float("nan")],
                              [float("nan"), 7, float("nan")]]],
                             [[[False, False, False], [True, True, True], [False, True, False]]])
        prior = torch.tensor([[float("nan"), float("-inf"), 1000]], dtype=torch.float64)
        decoded, = decode_event_geometry(scores, prior, coupling_strength=.5)
        self.assertEqual(self.ids(decoded), [(2, 1)])
        self.assertEqual(decoded.pairs[0].terminal_score, 3.5)
        self.assertEqual((decoded.supplied_geometry_count, decoded.eligible_geometry_count,
                          decoded.unused_slots, decoded.status), (3, 1, 7, "partial"))

    def test_mixed_empty_batches_retain_reasons_and_all_unused_slots(self):
        scores = self.scores(torch.full((3, 2, 2), float("nan")),
                             [[[False, False], [False, False]],
                              [[True, True], [True, True]],
                              [[False, False], [True, False]]])
        prior = torch.tensor([[float("nan"), float("nan")],
                              [float("-inf"), float("-inf")],
                              [float("nan"), 0]], dtype=torch.float64)
        outputs = decode_event_geometry(scores, prior)
        self.assertEqual([item.status for item in outputs],
                         ["empty_no_valid_pairs", "empty_no_prior_support", "partial"])
        for item in outputs[:2]:
            self.assertEqual(item.pairs, ())
            self.assertEqual((item.eligible_geometry_count, item.unused_slots), (0, 8))
        self.assertEqual(self.ids(outputs[2]), [(1, 0)])
        # Scores at prior-unsupported pairs need not be finite even with λ>0.
        zero_supported = replace(scores, values=scores.values[:2], valid_mask=scores.valid_mask[:2])
        self.assertEqual(decode_event_geometry(zero_supported, prior[:2], coupling_strength=1), outputs[:2])

    def test_zero_event_or_geometry_dimensions_record_missing_candidates(self):
        for events, geometries in ((0, 3), (3, 0), (0, 0)):
            with self.subTest(events=events, geometries=geometries):
                scores = self.scores(torch.empty((1, events, geometries)))
                decoded, = decode_event_geometry(scores, torch.full((1, events), float("nan"), dtype=torch.float64),
                                                coupling_strength=1)
                self.assertEqual(decoded.status, "empty_no_valid_pairs")
                self.assertEqual(decoded.pairs, ())
                self.assertEqual(decoded.unused_slots, 8)
                self.assertEqual((decoded.supplied_event_count, decoded.supplied_geometry_count), (events, geometries))

    def test_supplied_budget_is_checked_before_invalid_slots_can_be_masked_out(self):
        for events, geometries, message in ((33, 8, "event slots"), (32, 9, "geometry slots")):
            scores = self.scores(torch.full((1, events, geometries), float("nan")),
                                 torch.zeros(1, events, geometries, dtype=torch.bool))
            with self.subTest(events=events, geometries=geometries), self.assertRaisesRegex(ValueError, message):
                decode_event_geometry(scores, torch.full((1, events), float("nan"), dtype=torch.float64))

    def test_k_and_limit_validation_is_explicit(self):
        scores, prior = self.scores([[[0]]]), torch.zeros(1, 1, dtype=torch.float64)
        for field in ("k", "max_geometry_slots", "max_events"):
            for value in (0, -1, True, 1.5):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                    decode_event_geometry(scores, prior, **{field: value})
        with self.assertRaisesRegex(ValueError, "k cannot exceed"):
            decode_event_geometry(scores, prior, k=9)
        decoded, = decode_event_geometry(scores, prior, k=1, max_geometry_slots=1, max_events=1)
        self.assertEqual(decoded.status, "complete")

    def test_shape_dtype_device_and_oracle_flag_errors_fail_explicitly(self):
        scores, prior = self.scores([[[0, 1], [2, 3]]]), torch.zeros(1, 2, dtype=torch.float64)
        bad_scores = [replace(scores, values=torch.zeros(2, 2, dtype=torch.float64)),
                      replace(scores, values=torch.zeros(1, 2, 2, dtype=torch.int64)),
                      replace(scores, values=torch.empty(0, 2, 2, dtype=torch.float64)),
                      replace(scores, valid_mask=torch.ones(1, 2, dtype=torch.bool)),
                      replace(scores, valid_mask=torch.ones(1, 2, 2)),
                      replace(scores, valid_mask=torch.ones(1, 2, 2, dtype=torch.bool, device="meta")),
                      replace(scores, is_oracle="false")]
        for case in bad_scores:
            with self.subTest(case=case), self.assertRaises(ValueError):
                decode_event_geometry(case, prior)
        for bad_prior in (torch.zeros(1, 2, 1, dtype=torch.float64),
                          torch.zeros(1, 2, dtype=torch.float32),
                          torch.zeros(1, 2, dtype=torch.float64, device="meta")):
            with self.subTest(prior=bad_prior), self.assertRaisesRegex(ValueError, "prior_logits"):
                decode_event_geometry(scores, bad_prior)
        with self.assertRaisesRegex(TypeError, "CompatibilityScores"):
            decode_event_geometry(scores.values, prior)

    def test_nonfinite_supported_priors_or_compatibility_are_not_silently_dropped(self):
        scores = self.scores([[[0], [1]]])
        for value in (float("nan"), float("inf")):
            with self.subTest(prior=value), self.assertRaisesRegex(ValueError, "Valid prior logits"):
                decode_event_geometry(scores, torch.tensor([[0, value]], dtype=torch.float64))
        for value in (float("nan"), float("inf"), float("-inf")):
            bad_scores = self.scores([[[0], [value]]])
            with self.subTest(score=value), self.assertRaisesRegex(ValueError, "compatibility scores must be finite"):
                decode_event_geometry(bad_scores, torch.zeros(1, 2, dtype=torch.float64), coupling_strength=1)

    def test_strength_validation_and_nonfinite_combined_scores(self):
        scores, prior = self.scores([[[2]]]), torch.zeros(1, 1, dtype=torch.float64)
        for strength in (-1, float("nan"), float("inf"), True, ".5", torch.tensor(.5)):
            with self.subTest(strength=strength), self.assertRaisesRegex(ValueError, "coupling_strength"):
                decode_event_geometry(scores, prior, coupling_strength=strength)
        with self.assertRaisesRegex(ValueError, "Combined terminal scores"):
            decode_event_geometry(scores, prior, coupling_strength=1e308)

    def test_large_common_prior_offset_does_not_destroy_normalization(self):
        scores = self.scores([[[0], [0]]])
        decoded, = decode_event_geometry(scores, torch.full((1, 2), 1e308, dtype=torch.float64))
        self.assertAlmostEqual(decoded.pairs[0].terminal_score, -math.log(2))
        with self.assertRaisesRegex(ValueError, "Normalized prior logits"):
            decode_event_geometry(scores, torch.tensor([[1e308, -1e308]], dtype=torch.float64))

    def test_outputs_are_detached_python_records_and_inputs_unchanged(self):
        values = torch.tensor([[[1., 2.], [3., 4.]]], dtype=torch.float64, requires_grad=True)
        mask = torch.ones_like(values, dtype=torch.bool)
        prior = torch.tensor([[.3, -.3]], dtype=torch.float64, requires_grad=True)
        before = (values.detach().clone(), mask.clone(), prior.detach().clone())
        scores = CompatibilityScores(values, mask, False)
        with torch.inference_mode():
            decoded, = decode_event_geometry(scores, prior, coupling_strength=.5)
        for pair in decoded.pairs:
            self.assertIs(type(pair.event_id), int)
            self.assertIs(type(pair.geometry_id), int)
            self.assertIs(type(pair.terminal_score), float)
        self.assertIsNone(values.grad)
        self.assertIsNone(prior.grad)
        for after, original in zip((values, mask, prior), before):
            torch.testing.assert_close(after, original, atol=0, rtol=0)
        self.assertEqual(asdict(decoded)["is_oracle"], False)

    def test_oracle_tag_survives_zero_strength_nonempty_and_empty_outputs(self):
        prior = torch.zeros(1, 1, dtype=torch.float64)
        for valid in ([[[True]]], [[[False]]]):
            for strength in (0, 1):
                for oracle in (False, True):
                    with self.subTest(valid=valid, strength=strength, oracle=oracle):
                        scores = self.scores([[[0]]], valid=valid, oracle=oracle)
                        decoded, = decode_event_geometry(scores, prior, coupling_strength=strength)
                        self.assertIs(decoded.is_oracle, oracle)

    def test_decoder_has_no_label_or_evidence_parameter(self):
        self.assertEqual(set(inspect.signature(decode_event_geometry).parameters),
                         {"scores", "prior_logits", "coupling_strength", "k", "max_geometry_slots", "max_events"})
        with self.assertRaises(TypeError):
            decode_event_geometry(self.scores([[[0]]]), torch.zeros(1, 1, dtype=torch.float64),
                                  evidence={"reference_event": 0})


if __name__ == "__main__":
    unittest.main()
