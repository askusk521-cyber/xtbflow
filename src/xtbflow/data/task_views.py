"""Strict task views for endpoint diagnostics and reactant-side generation."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class GeometryTaskMode(str, Enum):
    ENDPOINT_GEOMETRY = "endpoint_geometry"
    REACTANT_EVENT_GEOMETRY = "reactant_event_geometry"


def _mask(mask: torch.Tensor, shape: tuple[int, int], device: torch.device) -> None:
    if mask.shape != shape or mask.dtype is not torch.bool or mask.device != device:
        raise ValueError(f"atom_mask must be boolean {shape} on the coordinate device")


def _coords(value: torch.Tensor, name: str) -> None:
    if value.ndim != 3 or value.shape[-1] != 3 or not value.is_floating_point():
        raise ValueError(f"{name} must be floating [B,N,3]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class EndpointGeometryInput:
    """Declared-endpoint diagnostic input; it is never a reactant discovery view."""

    coordinates: torch.Tensor
    atom_mask: torch.Tensor
    endpoint_coordinates: torch.Tensor
    mode: GeometryTaskMode = GeometryTaskMode.ENDPOINT_GEOMETRY

    def validate(self) -> None:
        if self.mode is not GeometryTaskMode.ENDPOINT_GEOMETRY:
            raise ValueError("endpoint input must use endpoint_geometry mode")
        _coords(self.coordinates, "coordinates")
        _coords(self.endpoint_coordinates, "endpoint_coordinates")
        if self.endpoint_coordinates.shape != self.coordinates.shape or self.endpoint_coordinates.device != self.coordinates.device or self.endpoint_coordinates.dtype != self.coordinates.dtype:
            raise ValueError("endpoint_coordinates must match coordinates")
        _mask(self.atom_mask, self.coordinates.shape[:2], self.coordinates.device)


@dataclass(frozen=True)
class ReactantEventGeometryInput:
    """Reactant-side event geometry input with no target/product fields."""

    coordinates: torch.Tensor
    atom_mask: torch.Tensor
    event_features: torch.Tensor
    pair_edits: torch.Tensor
    mode: GeometryTaskMode = GeometryTaskMode.REACTANT_EVENT_GEOMETRY

    def validate(self) -> None:
        if self.mode is not GeometryTaskMode.REACTANT_EVENT_GEOMETRY:
            raise ValueError("reactant input must use reactant_event_geometry mode")
        _coords(self.coordinates, "coordinates")
        b, n, _ = self.coordinates.shape
        _mask(self.atom_mask, (b, n), self.coordinates.device)
        if self.event_features.ndim != 3 or self.event_features.shape[0] != b or self.event_features.shape[1] < 1 or not self.event_features.is_floating_point():
            raise ValueError("event_features must be floating [B,E,F]")
        if self.event_features.device != self.coordinates.device or self.event_features.dtype != self.coordinates.dtype:
            raise ValueError("event_features must match coordinates")
        e = self.event_features.shape[1]
        if self.pair_edits.shape != (b, e, n, n) or self.pair_edits.device != self.coordinates.device or self.pair_edits.dtype != self.coordinates.dtype:
            raise ValueError("pair_edits must be [B,E,N,N] matching coordinates")
        if not bool(torch.isfinite(self.event_features).all()) or not bool(torch.isfinite(self.pair_edits).all()):
            raise ValueError("event inputs must be finite")
        if not torch.allclose(self.pair_edits, self.pair_edits.transpose(-1, -2)):
            raise ValueError("pair_edits must be symmetric")
        if bool(self.pair_edits.diagonal(dim1=-2, dim2=-1).ne(0).any()):
            raise ValueError("pair_edits diagonal must be zero")


@dataclass(frozen=True)
class GeometryTarget:
    """Supervision view kept separate from inference inputs."""

    coordinates: torch.Tensor
    observed_mask: torch.Tensor

    def validate(self, *, batch: int, atoms: int, device: torch.device, dtype: torch.dtype) -> None:
        if self.coordinates.ndim != 3 or self.coordinates.shape != (batch, atoms, 3) or not self.coordinates.is_floating_point() or self.coordinates.device != device or self.coordinates.dtype != dtype:
            raise ValueError("target coordinates must match the input geometry")
        if self.observed_mask.shape != (batch, atoms) or self.observed_mask.dtype is not torch.bool or self.observed_mask.device != device:
            raise ValueError("observed_mask must be boolean [B,N] on the input device")
        if not bool(torch.isfinite(self.coordinates[self.observed_mask]).all()):
            raise ValueError("observed target coordinates must be finite")


@dataclass(frozen=True)
class GeometryFlowResult:
    mode: GeometryTaskMode
    coordinates: torch.Tensor
    direction: torch.Tensor | None
    information_access: tuple[str, ...]
