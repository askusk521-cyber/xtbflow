"""Reactant-only strong-rule geometry initialisation.

The initializer turns a rule proposal into a bounded coordinate guess.  It
uses only the supplied reactant coordinates, atom mask, and proposed bond
edits; no product, transition state, reference mode, or calculator is read.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class BondEditHint:
    """A reactant-side bond edit used only to orient an initial displacement."""

    i: int
    j: int
    delta: float

    def __post_init__(self) -> None:
        if type(self.i) is not int or type(self.j) is not int or self.i < 0 or self.j < 0 or self.i == self.j:
            raise ValueError("bond edit endpoints must be distinct nonnegative integers")
        if isinstance(self.delta, bool) or not isinstance(self.delta, (int, float)) or not torch.isfinite(torch.tensor(float(self.delta))):
            raise ValueError("bond edit delta must be finite")
        if float(self.delta) == 0.0:
            raise ValueError("bond edit delta must be nonzero")


def _validate_inputs(coordinates: Tensor, atom_mask: Tensor) -> None:
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3 or not coordinates.is_floating_point():
        raise ValueError("coordinates must be floating [B,N,3]")
    if atom_mask.shape != coordinates.shape[:2] or atom_mask.dtype is not torch.bool or atom_mask.device != coordinates.device:
        raise ValueError("atom_mask must be boolean [B,N] on the coordinate device")
    if not bool(torch.isfinite(coordinates).all()):
        raise ValueError("coordinates must be finite")
    if not bool(atom_mask.any(dim=1).all()):
        raise ValueError("each sample must contain at least one active atom")


def rule_geometry_initialization(
    coordinates: Tensor,
    atom_mask: Tensor,
    bond_edits: Sequence[BondEditHint | tuple[int, int, float]] = (),
    *,
    displacement: float = 0.15,
    max_displacement: float = 0.30,
) -> Tensor:
    """Return a bounded rule-based coordinate guess.

    Positive edits move the two endpoints towards one another and negative
    edits move them apart.  The direction is derived from the current
    reactant geometry; a zero-distance pair receives a deterministic x-axis
    direction.  The total displacement of any atom is clipped to the declared
    bound and padded atoms remain unchanged.
    """

    _validate_inputs(coordinates, atom_mask)
    if isinstance(displacement, bool) or not isinstance(displacement, (int, float)) or float(displacement) <= 0:
        raise ValueError("displacement must be positive")
    if isinstance(max_displacement, bool) or not isinstance(max_displacement, (int, float)) or float(max_displacement) <= 0:
        raise ValueError("max_displacement must be positive")
    if float(displacement) > float(max_displacement):
        raise ValueError("displacement cannot exceed max_displacement")
    result = coordinates.clone()
    b, n, _ = coordinates.shape
    delta = torch.zeros_like(coordinates)
    for raw in bond_edits:
        edit = raw if isinstance(raw, BondEditHint) else BondEditHint(*raw)
        if edit.i >= n or edit.j >= n:
            raise ValueError("bond edit endpoint is outside the atom inventory")
        active = atom_mask[:, edit.i] & atom_mask[:, edit.j]
        vector = coordinates[:, edit.j] - coordinates[:, edit.i]
        distance = vector.norm(dim=-1, keepdim=True)
        fallback = torch.zeros_like(vector)
        fallback[:, 0] = 1.0
        direction = torch.where(distance > 1e-8, vector / distance.clamp_min(1e-8), fallback)
        sign = -1.0 if float(edit.delta) > 0 else 1.0
        step = direction * (float(displacement) * sign)
        delta[:, edit.i] += torch.where(active, step, torch.zeros_like(step))
        delta[:, edit.j] -= torch.where(active, step, torch.zeros_like(step))
    delta = torch.where(atom_mask[..., None], delta, torch.zeros_like(delta))
    norms = delta.norm(dim=-1, keepdim=True)
    scale = torch.clamp(float(max_displacement) / norms.clamp_min(1e-12), max=1.0)
    delta = delta * scale
    return result + delta


@dataclass(frozen=True)
class RuleGeometryInitializer:
    """Reusable bounded initializer with an explicit software contract."""

    displacement: float = 0.15
    max_displacement: float = 0.30

    def __post_init__(self) -> None:
        if self.displacement <= 0 or self.max_displacement <= 0 or self.displacement > self.max_displacement:
            raise ValueError("initializer displacement bounds are invalid")

    def __call__(self, coordinates: Tensor, atom_mask: Tensor, bond_edits: Iterable[BondEditHint | tuple[int, int, float]] = ()) -> Tensor:
        return rule_geometry_initialization(
            coordinates,
            atom_mask,
            tuple(bond_edits),
            displacement=self.displacement,
            max_displacement=self.max_displacement,
        )
