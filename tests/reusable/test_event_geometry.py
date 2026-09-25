"""CPU software properties of untrained event–geometry coupling, not chemistry."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from mechai.models.conditioning import ConditionSource
from mechai.models.event_geometry import (CandidateGeometry, CandidateGeometrySource,
    CompatibilityEvidence, EventGeometryCompatibility, EventGeometryCondition,
    compatibility_correspondence_loss)


class EventGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(20260922)
        self.model = EventGeometryCompatibility(4).double()
        self.edits = torch.zeros(2, 3, 5, 5, dtype=torch.float64)
        self.edits[:, 0, 0, 1] = self.edits[:, 0, 1, 0] = -1
        self.edits[:, 1, 1, 2] = self.edits[:, 1, 2, 1] = 1
        self.edits[:, 1, 1, 3] = self.edits[:, 1, 3, 1] = -1
        # Event 2 is a no-edit/identity candidate; no physical stop is implied.
        self.condition = EventGeometryCondition(torch.randn(2, 3, 4, dtype=torch.float64), self.edits,
            torch.tensor([[True, True, True], [True, True, False]]), ConditionSource.INDEPENDENT_METADATA)
        self.geometry = CandidateGeometry(torch.randn(2, 2, 5, 3, dtype=torch.float64),
            torch.tensor([[True, True, True, True, True], [True, True, True, True, False]]),
            torch.tensor([[True, True], [True, False]]), CandidateGeometrySource.GENERATED_CANDIDATE)
        self.prior = torch.tensor([[.3, -.7, 1.4], [.4, .2, float("nan")]], dtype=torch.float64)

    def couple(self, condition=None, geometry=None, **kwargs):
        return self.model.couple(condition or self.condition, geometry or self.geometry,
                                 self.prior, coupling_strength=kwargs.pop("coupling_strength", .7), **kwargs)

    def test_parameter_count_and_nonconstant_geometry_event_dependence(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 1409)
        scores = self.model(self.condition, self.geometry).values
        self.assertGreater(float((scores[0, :, 0] - scores[0, :, 1]).detach().abs().max()), 1e-6)
        changed = replace(self.condition, pair_edits=-self.edits)
        self.assertGreater(float((self.model(changed, self.geometry).values - scores).detach().abs().max()), 1e-6)
        moved = self.geometry.coordinates.clone()
        moved[0, :, 4] += torch.tensor([2.5, -.8, 1.1])
        self.assertGreater(float((self.model(self.condition, replace(self.geometry, coordinates=moved)).values - scores).detach().abs().max()), 1e-6)

    def test_rotation_translation_and_reflection_scores_and_feedback(self):
        before = self.couple()
        self.assertGreater(float(before.coordinate_feedback.abs().max()), 1e-6)
        q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
        for transform in (q, -q):
            moved = replace(self.geometry, coordinates=self.geometry.coordinates @ transform + 4.2)
            after = self.couple(geometry=moved)
            torch.testing.assert_close(after.scores.values, before.scores.values, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(after.posterior, before.posterior, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(after.coordinate_feedback, before.coordinate_feedback @ transform, atol=1e-12, rtol=1e-12)
        # Translation-invariant scalar derivatives have no net translation mode.
        torch.testing.assert_close(before.coordinate_feedback.sum(dim=-2), torch.zeros(2, 2, 3, dtype=torch.float64), atol=1e-12, rtol=0)

    def test_atom_permutation_preserves_score_and_permutates_feedback(self):
        permutation = torch.tensor([4, 2, 0, 3, 1])
        condition = replace(self.condition, pair_edits=self.edits[:, :, permutation][:, :, :, permutation])
        geometry = replace(self.geometry, coordinates=self.geometry.coordinates[:, :, permutation], atom_mask=self.geometry.atom_mask[:, permutation])
        before, after = self.couple(), self.couple(condition, geometry)
        torch.testing.assert_close(after.scores.values, before.scores.values, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(after.coordinate_feedback, before.coordinate_feedback[:, :, permutation], atol=1e-12, rtol=1e-12)

    def test_event_and_geometry_permutations_change_only_corresponding_axes(self):
        events, geometries = torch.tensor([2, 0, 1]), torch.tensor([1, 0])
        condition = replace(self.condition, event_features=self.condition.event_features[:, events],
            pair_edits=self.edits[:, events], event_valid_mask=self.condition.event_valid_mask[:, events])
        geometry = replace(self.geometry, coordinates=self.geometry.coordinates[:, geometries],
            geometry_valid_mask=self.geometry.geometry_valid_mask[:, geometries])
        after = self.model.couple(condition, geometry, self.prior[:, events], coupling_strength=.7)
        before = self.couple()
        torch.testing.assert_close(after.posterior, before.posterior[:, events][:, :, geometries], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(after.coordinate_feedback, before.coordinate_feedback[:, geometries], atol=1e-12, rtol=1e-12)

    def test_nan_padding_and_missing_candidates_do_not_change_real_results(self):
        coords, features, edits = self.geometry.coordinates.clone(), self.condition.event_features.clone(), self.edits.clone()
        coords[1, :, 4] = float("nan")
        coords[1, 1] = float("nan")
        features[1, 2] = float("nan")
        edits[1, :, 4, :] = edits[1, :, :, 4] = float("nan")
        edits[1, 2] = float("nan")
        result = self.couple(replace(self.condition, event_features=features, pair_edits=edits), replace(self.geometry, coordinates=coords))
        baseline = self.couple()
        torch.testing.assert_close(result.posterior, baseline.posterior, atol=0, rtol=0)
        torch.testing.assert_close(result.coordinate_feedback, baseline.coordinate_feedback, atol=0, rtol=0)
        self.assertEqual(int(torch.count_nonzero(result.coordinate_feedback[1, :, 4])), 0)
        self.assertEqual(int(torch.count_nonzero(result.coordinate_feedback[1, 1])), 0)

    def test_extra_padding_leaves_real_atom_scores_and_feedback_unchanged(self):
        coords = torch.full((2, 2, 7, 3), float("nan"), dtype=torch.float64)
        coords[:, :, :5] = self.geometry.coordinates
        atom_mask = torch.cat((self.geometry.atom_mask, torch.zeros(2, 2, dtype=torch.bool)), dim=-1)
        edits = torch.full((2, 3, 7, 7), float("nan"), dtype=torch.float64)
        edits[:, :, :5, :5] = self.edits
        after = self.couple(replace(self.condition, pair_edits=edits), replace(self.geometry, coordinates=coords, atom_mask=atom_mask))
        before = self.couple()
        torch.testing.assert_close(after.posterior, before.posterior, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(after.coordinate_feedback[:, :, :5], before.coordinate_feedback, atol=1e-12, rtol=1e-12)
        self.assertEqual(int(torch.count_nonzero(after.coordinate_feedback[:, :, 5:])), 0)

    def test_zero_strength_exact_prior_zero_feedback_and_no_network_call(self):
        before = self.geometry.coordinates.clone()
        with patch.object(self.model, "forward", side_effect=AssertionError("coupling network must be bypassed")):
            result = self.couple(coupling_strength=0)
        prior = torch.softmax(self.prior.masked_fill(~self.condition.event_valid_mask, -torch.inf), dim=1)
        expected = torch.where(result.valid_mask, prior[:, :, None].expand(2, 3, 2), 0)
        torch.testing.assert_close(result.posterior, expected, atol=0, rtol=0)
        torch.testing.assert_close(result.coordinate_feedback, torch.zeros_like(before), atol=0, rtol=0)
        torch.testing.assert_close(self.geometry.coordinates, before, atol=0, rtol=0)
        self.assertIsNone(result.scores)

    def test_feedback_is_derivative_of_exact_same_posterior_partition_once(self):
        strength = .37
        coords = self.geometry.coordinates.clone().requires_grad_(True)
        result = self.couple(geometry=replace(self.geometry, coordinates=coords), coupling_strength=strength)
        scores = self.model(self.condition, replace(self.geometry, coordinates=coords))
        log_prior = torch.log_softmax(self.prior.masked_fill(~self.condition.event_valid_mask, -torch.inf), dim=1)
        logits = log_prior[:, :, None] + strength * scores.values
        expected_posterior = torch.softmax(logits[0], dim=0)
        torch.testing.assert_close(result.posterior[0], expected_posterior, atol=1e-12, rtol=1e-12)
        objective = torch.logsumexp(logits[0], dim=0).sum() + torch.logsumexp(logits[1, :2, 0], dim=0)
        expected_gradient = torch.autograd.grad(objective, coords)[0]
        torch.testing.assert_close(result.coordinate_feedback, expected_gradient, atol=1e-12, rtol=1e-12)
        self.assertGreater(float((result.coordinate_feedback - strength * expected_gradient).abs().max()), 1e-6)

    def test_feedback_matches_finite_difference_and_parameters_receive_gradients(self):
        result = self.couple()
        plus, minus = self.geometry.coordinates.clone(), self.geometry.coordinates.clone()
        epsilon = 1e-5
        plus[0, 0, 0, 0] += epsilon
        minus[0, 0, 0, 0] -= epsilon
        def partition(coordinates):
            scores = self.model(self.condition, replace(self.geometry, coordinates=coordinates)).values
            return torch.logsumexp(torch.log_softmax(self.prior[0], dim=0) + .7 * scores[0, :, 0], dim=0)
        finite_difference = (partition(plus) - partition(minus)) / (2 * epsilon)
        torch.testing.assert_close(result.coordinate_feedback[0, 0, 0, 0], finite_difference, atol=1e-9, rtol=1e-7)
        result.posterior[0, 0, 0].backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.model.parameters() if p.grad is not None), 1e-7)

    def test_all_invalid_and_all_zero_prior_support_are_finite_zeros(self):
        cases = [(replace(self.condition, event_valid_mask=torch.zeros_like(self.condition.event_valid_mask)), self.geometry),
                 (self.condition, replace(self.geometry, geometry_valid_mask=torch.zeros_like(self.geometry.geometry_valid_mask))),
                 (self.condition, replace(self.geometry, atom_mask=torch.zeros_like(self.geometry.atom_mask)))]
        for condition, geometry in cases:
            for strength in (0, .7):
                with self.subTest(strength=strength):
                    result = self.couple(condition, geometry, coupling_strength=strength)
                    self.assertEqual(int(torch.count_nonzero(result.posterior)), 0)
                    self.assertEqual(int(torch.count_nonzero(result.coordinate_feedback)), 0)
                    self.assertFalse(bool(result.valid_mask.any()))
        zero_support = self.model.couple(self.condition, self.geometry, torch.full_like(self.prior, -torch.inf), coupling_strength=.7)
        self.assertEqual(int(torch.count_nonzero(zero_support.posterior)), 0)
        self.assertEqual(int(torch.count_nonzero(zero_support.coordinate_feedback)), 0)

    def test_coincident_atoms_and_higher_order_feedback_are_finite(self):
        coordinates = torch.zeros_like(self.geometry.coordinates)
        result = self.couple(geometry=replace(self.geometry, coordinates=coordinates), create_graph=True)
        self.assertTrue(bool(torch.isfinite(result.coordinate_feedback).all()))
        result.coordinate_feedback.square().sum().backward()
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in self.model.parameters()))
        result = self.couple(create_graph=True)
        self.model.zero_grad()
        result.coordinate_feedback.square().sum().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.model.parameters() if p.grad is not None), 1e-9)

    def test_low_precision_is_rejected_before_distance_or_autograd(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = EventGeometryCompatibility(4).to(dtype=dtype)
                condition = replace(self.condition, event_features=self.condition.event_features.to(dtype),
                                    pair_edits=self.edits.to(dtype))
                geometry = replace(self.geometry, coordinates=self.geometry.coordinates.to(dtype))
                with self.assertRaisesRegex(ValueError, "only float32/float64"):
                    model(condition, geometry)
                with self.assertRaisesRegex(ValueError, "only float32/float64"):
                    model.couple(condition, geometry, self.prior.to(dtype), coupling_strength=.7, create_graph=True)

    def test_supported_precision_collision_padding_and_derivatives_are_finite(self):
        for dtype in (torch.float32, torch.float64):
            for collide in (False, True):
                with self.subTest(dtype=dtype, coincide=collide):
                    model = EventGeometryCompatibility(4).to(dtype=dtype)
                    condition = replace(self.condition, event_features=self.condition.event_features.to(dtype),
                                        pair_edits=self.edits.to(dtype))
                    coordinates = self.geometry.coordinates.to(dtype).clone()
                    if collide:
                        coordinates[:, :, :2] = 0
                    coordinates[1, :, 4] = float("nan")
                    geometry = replace(self.geometry, coordinates=coordinates)
                    result = model.couple(condition, geometry, self.prior.to(dtype), coupling_strength=.7, create_graph=True)
                    for value in (result.scores.values, result.posterior, result.coordinate_feedback):
                        self.assertTrue(bool(torch.isfinite(value).all()))
                    self.assertEqual(int(torch.count_nonzero(result.coordinate_feedback[1, :, 4])), 0)
                    result.coordinate_feedback.square().sum().backward()
                    self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
                    self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None), 1e-10)
        # Supported tensor precision must not silently rescue an underflowing
        # user override of the smoothing scale either.
        model = EventGeometryCompatibility(4, distance_epsilon=1e-200).double()
        with self.assertRaisesRegex(ValueError, "epsilon squared"):
            model(self.condition, self.geometry)

    def test_no_grad_inference_can_request_feedback_without_mutating_requires_grad(self):
        self.assertFalse(self.geometry.coordinates.requires_grad)
        with torch.no_grad():
            result = self.couple()
        self.assertGreater(float(result.coordinate_feedback.abs().max()), 1e-6)
        self.assertFalse(result.coordinate_feedback.requires_grad)
        self.assertFalse(self.geometry.coordinates.requires_grad)
        with torch.inference_mode(), self.assertRaisesRegex(ValueError, "requires autograd"):
            self.couple()

    def test_evidence_masks_unknowns_without_changing_inference_candidates(self):
        scores = self.model(self.condition, self.geometry)
        scores.values.retain_grad()
        targets = torch.full_like(scores.values, float("nan"))
        observed = torch.zeros_like(scores.valid_mask)
        targets[0, 0, 0], observed[0, 0, 0] = 1, True
        loss = compatibility_correspondence_loss(scores, CompatibilityEvidence(targets, observed))
        loss.backward()
        self.assertGreater(float(scores.values.grad[0, 0, 0].abs()), 0)
        self.assertEqual(int(torch.count_nonzero(scores.values.grad[~observed])), 0)
        self.assertTrue(bool(scores.valid_mask[0].all()))
        unknown_scores = self.model(self.condition, self.geometry)
        zero_loss = compatibility_correspondence_loss(unknown_scores, CompatibilityEvidence(torch.full_like(targets, float("nan")), torch.zeros_like(observed)))
        self.assertEqual(float(zero_loss.detach()), 0)
        zero_loss.backward()
        observed[1, 2, 1] = True
        with self.assertRaisesRegex(ValueError, "invalid candidate"):
            compatibility_correspondence_loss(scores, CompatibilityEvidence(targets, observed))

    def test_weighted_reference_loss_and_oracle_guards(self):
        scores = self.model(self.condition, self.geometry)
        observed = torch.zeros_like(scores.valid_mask)
        observed[0, :2, 0] = True
        targets, weights = torch.full_like(scores.values, float("nan")), torch.full_like(scores.values, float("nan"))
        targets[0, :2, 0], weights[0, :2, 0] = torch.tensor([0., 1.]), torch.tensor([1., 3.])
        loss = compatibility_correspondence_loss(scores, CompatibilityEvidence(targets, observed, weights))
        expected = (torch.nn.functional.softplus(scores.values[0, 0, 0]) + 3 * torch.nn.functional.softplus(-scores.values[0, 1, 0])) / 4
        torch.testing.assert_close(loss, expected)
        for condition, geometry in ((replace(self.condition, source=ConditionSource.REFERENCE_TS_ORACLE), self.geometry),
                                    (self.condition, replace(self.geometry, source=CandidateGeometrySource.REFERENCE_TS_ORACLE))):
            with self.assertRaisesRegex(ValueError, "oracle"):
                self.model(condition, geometry)
        allowed = EventGeometryCompatibility(4, allow_reference_ts_oracle=True).double()
        self.assertTrue(allowed(self.condition, replace(self.geometry, source=CandidateGeometrySource.REFERENCE_TS_ORACLE)).is_oracle)

    def test_invalid_shapes_nonfinite_real_data_asymmetry_and_bad_strength_fail(self):
        edits = self.edits.clone()
        edits[0, 0, 0, 1] += 1
        with self.assertRaisesRegex(ValueError, "symmetric"):
            self.model(replace(self.condition, pair_edits=edits), self.geometry)
        edits[0, 0, 1, 0] += 1
        edits[0, 0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "diagonal"):
            self.model(replace(self.condition, pair_edits=edits), self.geometry)
        coordinates = self.geometry.coordinates.clone()
        coordinates[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.model(self.condition, replace(self.geometry, coordinates=coordinates))
        for strength in (-1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.couple(coupling_strength=strength)
        with self.assertRaises(ValueError):
            self.model(self.condition, replace(self.geometry, atom_mask=self.geometry.atom_mask.float()))


if __name__ == "__main__":
    unittest.main()
