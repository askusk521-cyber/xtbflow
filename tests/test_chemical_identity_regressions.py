"""Zero-QC labelled parity and current-input regressions for #185."""
from dataclasses import replace

import numpy as np
import pytest

from xtbflow.calculators import CalculationResult, CalculatorProtocol, MolecularSystem
from xtbflow.sampling.adapters import ProviderRequest
from xtbflow.sampling.live_physical import LiveXTBPhysicalProvider
from xtbflow.sampling.parallel_guidance import GenerationSnapshot, GuidanceLane
from xtbflow.sampling.reactant_candidates import _basis, _proper_signed_permutations, build_reactant_approach


P = np.array([(0, 0, 0), (1.1, .1, 0), (-.2, 1.4, .1), (.1, -.3, 1.7)])


def volume(x):
    return np.linalg.det(x[1:] - x[0]) / 6


@pytest.mark.parametrize('angle', [0, 37, 105, 260])
@pytest.mark.parametrize('order', [(0, 1, 2, 3), (2, 0, 3, 1)])
def test_all_24_transforms_preserve_labelled_parity_and_distances(angle, order):
    theta = np.deg2rad(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    p = P[list(order)]
    q = p @ rotation.T + (4, 1, -2)
    for signed in _proper_signed_permutations():
        transform = _basis(p) @ signed @ _basis(q).T
        assert np.linalg.det(transform) == pytest.approx(1)
        moved = q @ transform.T
        assert volume(moved) == pytest.approx(volume(q))
        np.testing.assert_allclose(np.linalg.norm(moved[:, None] - moved[None], axis=-1),
                                   np.linalg.norm(q[:, None] - q[None], axis=-1), atol=1e-12)


@pytest.mark.parametrize('points', [np.zeros((4, 3)), np.eye(3), np.array([[0, 0, 0], [1, 0, 0]])])
def test_degenerate_basis_is_finite_and_proper(points):
    basis = _basis(points)
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)
    assert np.linalg.det(basis) == pytest.approx(1)


def test_placement_keeps_atom_rows_and_bond_mapping_with_chiral_components():
    original = np.concatenate([P, P + (5, 0, 0)])
    system = MolecularSystem(('C', 'H', 'N', 'O') * 2, tuple(map(tuple, original)), 0, 1, {'kind': 'gas', 'periodic': False})
    result = build_reactant_approach(system, ((0, 1, 2, 3), (4, 5, 6, 7)))
    moved = np.asarray(result.coordinates)
    for group in (slice(0, 4), slice(4, 8)):
        assert volume(moved[group]) == pytest.approx(volume(original[group]))
        np.testing.assert_allclose(np.linalg.norm(moved[group, None] - moved[group][None], axis=-1),
                                   np.linalg.norm(original[group, None] - original[group][None], axis=-1), atol=1e-12)
    assert result.metadata['components'] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert result.metadata['enumerated_placements'] == 72
    assert result.metadata['seed_used_for_pose_selection'] is False


class RecordingBackend:
    def __init__(self, protocol, stale=False):
        self.protocol, self.stale, self.inputs = protocol, stale, []

    def evaluate(self, system, **kwargs):
        self.inputs.append(system)
        return CalculationResult(calculator=self.protocol.calculator, protocol_id=self.protocol.protocol_id,
            input_hash=self.inputs[0].input_hash if self.stale else system.input_hash,
            charge=system.charge, multiplicity=system.multiplicity, operation='energy_forces', energy=-1,
            forces=tuple((.1, 0., 0.) for _ in system.symbols), calculator_build_hash='fake')


@pytest.mark.parametrize('stale', [False, True])
def test_two_snapshots_verify_actual_payload_and_returned_input_hash(stale):
    protocol = CalculatorProtocol('identity-test', 'xtb_oracle', 'GFN2-xTB')
    original = MolecularSystem(('H', 'H'), ((0., 0., 0.), (1., 0., 0.)), 0, 1, {'periodic': False})
    backend = RecordingBackend(protocol, stale)
    provider = LiveXTBPhysicalProvider(backend, original, protocol)
    first = GenerationSnapshot('parent', 'attempt', 0, original.coordinates, (0., 0.), (True, True), 0,
                               metadata={'input_hash': original.input_hash})
    second = replace(first, generation_step=1, geometry=((0., 0., 0.), (1.2, 0., 0.)), snapshot_id='')
    def request(snapshot):
        return ProviderRequest(snapshot, GuidanceLane.GEOMETRY, provider.provider_revision,
                               provider.config_fingerprint, provider.protocol_fingerprint)
    provider.invoke(request(first))
    if stale:
        with pytest.raises(ValueError, match='current snapshot'):
            provider.invoke(request(second))
    else:
        out = provider.invoke(request(second))
        assert out.input_fingerprint == second.snapshot_id
        assert provider.last_feedback.input_hash == backend.inputs[1].input_hash
        assert provider.last_feedback.input_hash != original.input_hash
    assert backend.inputs[1].coordinates == second.geometry
    assert backend.inputs[1].symbols == original.symbols
    assert second.to_mapping()['metadata']['input_hash'] == original.input_hash
