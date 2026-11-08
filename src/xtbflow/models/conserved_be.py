"""Packed symmetric bond/electron coordinates and conservation projection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


def packed_size(n_atoms: int) -> int:
    if type(n_atoms) is not int or n_atoms < 1:
        raise ValueError("n_atoms must be a positive integer")
    return n_atoms * (n_atoms + 1) // 2


def upper_triangle_indices(n_atoms: int, *, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return diagonal-first-free upper-triangle indices in stable row order."""

    packed_size(n_atoms)
    rows, cols = torch.triu_indices(n_atoms, n_atoms, device=device)
    return rows, cols


def upper_triangle_weights(n_atoms: int, *, dtype: torch.dtype = torch.float32, device: torch.device | None = None) -> torch.Tensor:
    rows, cols = upper_triangle_indices(n_atoms, device=device)
    return torch.where(rows == cols, torch.ones_like(rows, dtype=dtype), torch.full_like(rows, 2, dtype=dtype))


def pack_be(matrix: torch.Tensor) -> torch.Tensor:
    """Pack the upper triangle; off-diagonal entries represent two electrons."""

    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("matrix must have a square final pair of dimensions")
    n = matrix.shape[-1]
    if matrix.is_floating_point() and not torch.isfinite(matrix).all():
        raise ValueError("matrix must be finite")
    symmetric = torch.allclose(matrix, matrix.transpose(-1, -2)) if matrix.is_floating_point() else torch.equal(matrix, matrix.transpose(-1, -2))
    if not symmetric:
        raise ValueError("bond/electron matrix must be symmetric")
    rows, cols = upper_triangle_indices(n, device=matrix.device)
    return matrix[..., rows, cols]


def unpack_be(packed: torch.Tensor, n_atoms: int) -> torch.Tensor:
    """Unpack a packed vector into a symmetric matrix without rounding."""

    if packed.ndim < 1 or packed.shape[-1] != packed_size(n_atoms):
        raise ValueError("packed vector has the wrong feature dimension")
    if packed.is_floating_point() and not torch.isfinite(packed).all():
        raise ValueError("packed vector must be finite")
    rows, cols = upper_triangle_indices(n_atoms, device=packed.device)
    matrix = torch.zeros(*packed.shape[:-1], n_atoms, n_atoms, dtype=packed.dtype, device=packed.device)
    matrix[..., rows, cols] = packed
    matrix[..., cols, rows] = packed
    return matrix


@dataclass(frozen=True)
class ConservationProjector:
    """Orthogonal projector onto ``A v = 0`` for arbitrary batch shapes."""

    constraint_matrix: torch.Tensor
    tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if self.constraint_matrix.ndim != 2 or self.constraint_matrix.shape[0] < 1:
            raise ValueError("constraint_matrix must be [constraints, features]")
        if not torch.isfinite(self.constraint_matrix).all():
            raise ValueError("constraint_matrix must be finite")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")

    @property
    def n_features(self) -> int:
        return int(self.constraint_matrix.shape[1])

    def project(self, vector: torch.Tensor) -> torch.Tensor:
        """Remove the constrained component while preserving all batch axes."""

        if vector.shape[-1] != self.n_features:
            raise ValueError("vector feature dimension does not match constraints")
        matrix = self.constraint_matrix.to(device=vector.device, dtype=vector.dtype)
        flat = vector.reshape(-1, self.n_features)
        gram_pinv = torch.linalg.pinv(matrix @ matrix.transpose(0, 1))
        residual = flat @ matrix.transpose(0, 1)
        correction = residual @ gram_pinv @ matrix
        return (flat - correction).reshape_as(vector)

    def project_masked(self, vector: torch.Tensor, feature_mask: torch.Tensor) -> torch.Tensor:
        """Project active features while keeping padded features exactly zero.

        Each batch item may expose a different active feature subset.  The
        orthogonal projection is therefore computed against the corresponding
        active columns of ``A`` rather than projecting first and masking later,
        which could reintroduce non-zero padded entries.
        """

        if vector.shape[-1] != self.n_features:
            raise ValueError("vector feature dimension does not match constraints")
        if feature_mask.shape != vector.shape or feature_mask.dtype is not torch.bool or feature_mask.device != vector.device:
            raise ValueError("feature_mask must be boolean and match vector")
        if not vector.is_floating_point() or not bool(torch.isfinite(vector).all()):
            raise ValueError("vector must be finite floating point")
        matrix = self.constraint_matrix.to(device=vector.device, dtype=vector.dtype)
        flat_vector = vector.reshape(-1, self.n_features)
        flat_mask = feature_mask.reshape(-1, self.n_features)
        # Samples in one batch usually share the same active atom count.  The
        # previous row-by-row implementation recomputed an identical
        # pseudoinverse for every sample (and twice per joint-flow step).  Group
        # rows by their boolean mask and factor each active constraint once;
        # ``index_copy`` keeps the result differentiable with respect to the
        # input values while preserving exact zero padding.
        # ``unique`` has broader backend support for byte tensors than for
        # boolean tensors; the conversion is local and keeps the grouping on
        # the original device without a host-side mask round trip.
        unique_masks, inverse = torch.unique(
            flat_mask.to(dtype=torch.uint8), dim=0, return_inverse=True
        )
        projected = torch.zeros_like(flat_vector)
        for group_index in range(unique_masks.shape[0]):
            active = unique_masks[group_index].to(dtype=torch.bool)
            active_indices = active.nonzero(as_tuple=False).flatten()
            if not active_indices.numel():
                continue
            active_matrix = matrix[:, active]
            gram_pinv = torch.linalg.pinv(active_matrix @ active_matrix.transpose(0, 1))
            row_indices = torch.where(inverse == group_index)[0]
            active_values = flat_vector.index_select(0, row_indices).index_select(1, active_indices)
            residual = active_values @ active_matrix.transpose(0, 1)
            projected_values = active_values - residual @ gram_pinv @ active_matrix
            rows = torch.zeros(
                (row_indices.numel(), self.n_features), dtype=vector.dtype, device=vector.device
            ).index_copy(1, active_indices, projected_values)
            projected = projected.index_copy(0, row_indices, rows)
        return projected.reshape_as(vector)

    def residual(self, vector: torch.Tensor) -> torch.Tensor:
        if vector.shape[-1] != self.n_features:
            raise ValueError("vector feature dimension does not match constraints")
        matrix = self.constraint_matrix.to(device=vector.device, dtype=vector.dtype)
        return vector @ matrix.transpose(0, 1)

    def assert_projected(self, vector: torch.Tensor) -> None:
        residual = self.residual(vector)
        if not torch.allclose(residual, torch.zeros_like(residual), atol=self.tolerance, rtol=0):
            raise ValueError("vector is outside the conservation subspace")


def total_electron_projector(symbols: Sequence[str], *, dtype: torch.dtype = torch.float32, device: torch.device | None = None) -> ConservationProjector:
    """Build the weighted upper-triangle projector for total BE electrons."""

    return ConservationProjector(upper_triangle_weights(len(symbols), dtype=dtype, device=device).unsqueeze(0))
