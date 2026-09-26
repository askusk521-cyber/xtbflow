from __future__ import annotations

import pytest
import torch

from mechai.data.events import LewisState
from xtbflow.models import (
    ConservationProjector,
    ConservedEventFlow,
    DecodeError,
    decode_batch,
    decode_endpoint,
    pack_be,
    total_electron_projector,
    unpack_be,
    upper_triangle_weights,
)
from xtbflow.training import decode_summary, flow_matching_loss


def test_pack_unpack_and_weighted_upper_triangle_count():
    matrix = torch.tensor([[4.0, 1.0], [1.0, 6.0]])
    packed = pack_be(matrix)
    assert packed.tolist() == [4.0, 1.0, 6.0]
    assert upper_triangle_weights(2).tolist() == [1.0, 2.0, 1.0]
    assert torch.equal(unpack_be(packed, 2), matrix)
    assert torch.dot(packed, upper_triangle_weights(2)) == 12.0


def test_projector_uses_weighted_constraint_and_preserves_batch_shape():
    projector = total_electron_projector(("C", "O"))
    velocity = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 1.0, 0.0]])
    projected = projector.project(velocity)
    assert projected.shape == velocity.shape
    assert torch.allclose(projector.residual(projected), torch.zeros(2, 1), atol=1e-6)
    assert not torch.allclose(projected[0].sum(), torch.tensor(0.0))


def test_constructive_flow_preserves_reactant_total_and_zero_step():
    projector = total_electron_projector(("C", "O"))
    flow = ConservedEventFlow(projector)
    reactant = torch.tensor([[4.0, 0.0, 6.0]])
    noise = torch.tensor([[2.0, -1.0, 3.0]])
    start = flow.initial_state(reactant, noise)
    assert torch.allclose(projector.residual(start - reactant), torch.zeros(1, 1), atol=1e-6)
    assert torch.equal(flow.step(start, noise, 0.0), start)


def test_decoder_repairs_total_electrons_before_lewis_validation():
    decoded = decode_endpoint([4.0, 0.0, 4.0], ("C", "O"), 0)
    decoded.state.validate()
    assert sum(map(sum, decoded.state.be)) == 10
    assert all(type(value) is int and value >= 0 for value in decoded.packed)


def test_decoder_rejects_fractional_scope_and_reports_batch_diversity():
    with pytest.raises(DecodeError):
        decode_endpoint([0.0, 0.0, 0.0], ("C", "O"), 1)
    batch = decode_batch(([4.0, 0.0, 6.0], [4.0, float("nan"), 6.0]), ("C", "O"), 0)
    assert batch.attempted == 2
    assert batch.accepted == 1
    assert batch.rejected == 1
    assert batch.unique_endpoints == 1
    assert decode_summary(batch)["rejection_rate"] == 0.5


def test_flow_modes_share_the_same_tensor_contract():
    projector = total_electron_projector(("C", "O"))
    predicted = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    target = projector.project(torch.tensor([[0.5, -1.0, 0.5]]))
    constructive = flow_matching_loss(predicted, target, mode="constructive", projector=projector)
    soft = flow_matching_loss(predicted, target, mode="soft", projector=projector, penalty_weight=0.1)
    unconstrained = flow_matching_loss(predicted, target, mode="unconstrained")
    assert all(torch.isfinite(value) for value in (constructive, soft, unconstrained))
    constructive.backward()
    assert predicted.grad is not None
