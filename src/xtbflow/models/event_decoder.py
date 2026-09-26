"""Conservative discrete endpoint decoding for packed BE coordinates."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import torch

from mechai.data.events import LewisState, VALENCE_ELECTRONS

from .conserved_be import pack_be, packed_size, unpack_be


class DecodeError(ValueError):
    """Raised when no legal integer endpoint can be constructed."""


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def _even_round(value: float) -> int:
    return max(0, 2 * _round_half_up(value / 2.0))


@dataclass(frozen=True)
class DecodedEndpoint:
    state: LewisState
    packed: tuple[int, ...]


@dataclass(frozen=True)
class DecodeBatch:
    candidates: tuple[DecodedEndpoint, ...]
    attempted: int
    accepted: int
    rejected: int
    rejection_rate: float
    unique_endpoints: int
    rejection_reasons: tuple[str, ...]


def _repair_diagonal(preferences: list[int], total: int) -> list[int]:
    if total < 0 or total % 2:
        raise DecodeError("remaining diagonal electron count is not a nonnegative even integer")
    diagonal = [max(0, value) for value in preferences]
    current = sum(diagonal)
    if current < total:
        order = sorted(range(len(diagonal)), key=lambda index: (-preferences[index], index))
        cursor = 0
        while current < total:
            diagonal[order[cursor % len(order)]] += 2
            current += 2
            cursor += 1
    elif current > total:
        order = sorted(range(len(diagonal)), key=lambda index: (preferences[index], index), reverse=True)
        cursor = 0
        while current > total:
            index = order[cursor % len(order)]
            if diagonal[index] >= 2:
                diagonal[index] -= 2
                current -= 2
            cursor += 1
            if cursor > len(order) * (current + 1):
                raise DecodeError("could not repair diagonal electron count")
    return diagonal


def decode_endpoint(
    values: Sequence[float] | torch.Tensor,
    symbols: Sequence[str],
    charge: int,
    multiplicity: int = 1,
    *,
    max_bond_order: int = 3,
) -> DecodedEndpoint:
    """Repair a continuous vector into an integer, conserved, validated endpoint.

    Bond entries are rounded first only as a proposal.  The weighted total is
    then repaired by reducing bonds when necessary and assigning even diagonal
    electron counts before ``LewisState.validate`` is called.  Independent
    per-entry rounding is therefore never treated as a conservation proof.
    """

    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise DecodeError("decoder expects one packed vector")
        values = values.detach().cpu().tolist()
    if len(values) != packed_size(len(symbols)):
        raise DecodeError("packed vector has the wrong feature dimension")
    if type(charge) is not int or type(multiplicity) is not int:
        raise DecodeError("charge and multiplicity must be explicit integers")
    if type(max_bond_order) is not int or max_bond_order < 1:
        raise DecodeError("max_bond_order must be positive")
    if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in values):
        raise DecodeError("packed vector must be finite")
    n = len(symbols)
    rows, cols = torch.triu_indices(n, n)
    rounded = [max(0, min(max_bond_order, _round_half_up(float(values[index])))) for index in range(len(values))]
    offdiag_positions = [index for index, (row, col) in enumerate(zip(rows.tolist(), cols.tolist())) if row != col]
    total_electrons = sum(VALENCE_ELECTRONS.get(symbol, 0) for symbol in symbols) - charge
    if any(symbol not in VALENCE_ELECTRONS for symbol in symbols) or total_electrons < 0:
        raise DecodeError("symbols or charge are outside the explicit CHNOS scope")
    offdiag_total = sum(rounded[index] for index in offdiag_positions)
    while 2 * offdiag_total > total_electrons:
        choices = [index for index in offdiag_positions if rounded[index] > 0]
        if not choices:
            raise DecodeError("bond proposal consumes more electrons than the endpoint has")
        index = max(choices, key=lambda item: (rounded[item], abs(float(values[item]) - rounded[item]), -item))
        rounded[index] -= 1
        offdiag_total -= 1
    diagonal_positions = [index for index, (row, col) in enumerate(zip(rows.tolist(), cols.tolist())) if row == col]
    preferences = [_even_round(float(values[index])) for index in diagonal_positions]
    diagonal = _repair_diagonal(preferences, total_electrons - 2 * offdiag_total)
    for index, value in zip(diagonal_positions, diagonal):
        rounded[index] = value
    packed = torch.tensor(rounded, dtype=torch.int64)
    matrix = unpack_be(packed, n).to(torch.int64)
    try:
        state = LewisState(tuple(symbols), tuple(tuple(int(value) for value in row) for row in matrix.tolist()), charge, multiplicity)
        state.validate()
    except (TypeError, ValueError) as exc:
        raise DecodeError(f"integer endpoint failed LewisState validation: {exc}") from exc
    return DecodedEndpoint(state, tuple(int(value) for value in packed.tolist()))


def decode_batch(
    values: Iterable[Sequence[float] | torch.Tensor],
    symbols: Sequence[str],
    charge: int,
    multiplicity: int = 1,
    *,
    max_bond_order: int = 3,
) -> DecodeBatch:
    candidates: list[DecodedEndpoint] = []
    reasons: list[str] = []
    attempted = 0
    for item in values:
        attempted += 1
        try:
            candidates.append(decode_endpoint(item, symbols, charge, multiplicity, max_bond_order=max_bond_order))
        except DecodeError as exc:
            reasons.append(str(exc))
    unique = len({candidate.packed for candidate in candidates})
    rejected = attempted - len(candidates)
    return DecodeBatch(tuple(candidates), attempted, len(candidates), rejected, rejected / attempted if attempted else 0.0, unique, tuple(reasons))
