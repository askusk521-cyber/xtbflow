"""Quarantine-bound paired event/geometry development views.

The public Transition1x preprocessed asset contains aligned reactant, product,
and transition-state coordinates, but its source audit does not establish the
electronic-state or event-label requirements of a Track-B admission.  This
module therefore exposes only a deterministic *development* conversion.  It
uses endpoint distances to derive a binary bond-edit diagnostic and records
that the result is derived from the product endpoint.  Callers must keep the
result out of confirmatory or product-free evaluation.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


ELEMENT_VALENCE_ELECTRONS = {1: 1.0, 6: 4.0, 7: 5.0, 8: 6.0}
"""Valence-electron counts for the CHNO subset used by this diagnostic."""

_COVALENT_RADII = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66}


class Transition1xDevelopmentError(ValueError):
    """Raised when a development conversion cannot be made unambiguously."""


def _array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value)
    if ndim is not None and result.ndim != ndim:
        raise Transition1xDevelopmentError(f"{name} must have {ndim} dimensions")
    if not np.issubdtype(result.dtype, np.number) or not np.isfinite(result).all():
        raise Transition1xDevelopmentError(f"{name} must contain finite numeric values")
    return result


def _validate_geometry(atomic_numbers: Sequence[int], coordinates: Any, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    numbers = _array(atomic_numbers, name=f"{name}.atomic_numbers", ndim=1).astype(np.int64, copy=False)
    coords = _array(coordinates, name=f"{name}.coordinates", ndim=2).astype(np.float64, copy=False)
    if coords.shape != (numbers.size, 3):
        raise Transition1xDevelopmentError(f"{name} coordinates must be [N,3]")
    if numbers.size < 1 or any(int(number) not in _COVALENT_RADII for number in numbers.tolist()):
        raise Transition1xDevelopmentError("development conversion supports only H/C/N/O atoms")
    return numbers, coords


def infer_binary_bonds(
    atomic_numbers: Sequence[int],
    coordinates: Any,
    *,
    scale: float = 1.25,
) -> np.ndarray:
    """Infer a symmetric binary adjacency matrix from covalent radii.

    The threshold is a declared geometry diagnostic, not a replacement for a
    source-supplied chemical graph or bond-order assignment.
    """

    if not isinstance(scale, (int, float)) or not np.isfinite(scale) or scale <= 0:
        raise Transition1xDevelopmentError("scale must be finite and positive")
    numbers, coords = _validate_geometry(atomic_numbers, coordinates, name="geometry")
    radii = np.asarray([_COVALENT_RADII[int(number)] for number in numbers], dtype=np.float64)
    displacement = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt(np.sum(displacement * displacement, axis=-1))
    threshold = float(scale) * (radii[:, None] + radii[None, :])
    bonds = (distances > 0.4) & (distances <= threshold)
    np.fill_diagonal(bonds, False)
    return np.asarray(bonds, dtype=np.float64)


def conserved_event_delta(
    atomic_numbers: Sequence[int],
    reactant_coordinates: Any,
    product_coordinates: Any,
    *,
    bond_scale: float = 1.25,
) -> dict[str, Any]:
    """Return a conserved binary-edit diagnostic for one endpoint pair.

    A formed bond contributes ``+1`` to its off-diagonal pair and ``-1`` to
    each endpoint diagonal; a broken bond applies the reverse edit.  The
    weighted packed sum is therefore exactly zero.  The construction is
    explicitly endpoint-derived and is not an independent event annotation.
    """

    numbers_r, _ = _validate_geometry(atomic_numbers, reactant_coordinates, name="reactant")
    numbers_p, _ = _validate_geometry(atomic_numbers, product_coordinates, name="product")
    if not np.array_equal(numbers_r, numbers_p):
        raise Transition1xDevelopmentError("reactant and product atomic-number rows differ")
    reactant_bonds = infer_binary_bonds(numbers_r, reactant_coordinates, scale=bond_scale)
    product_bonds = infer_binary_bonds(numbers_p, product_coordinates, scale=bond_scale)
    delta_bonds = np.triu(product_bonds - reactant_bonds, k=1)
    delta = delta_bonds + delta_bonds.T
    degrees = delta_bonds.sum(axis=1) + delta_bonds.sum(axis=0)
    diagonal = -degrees
    np.fill_diagonal(delta, diagonal)
    weighted_total = float(np.trace(delta) + 2.0 * np.triu(delta, k=1).sum())
    if abs(weighted_total) > 1e-12:
        raise Transition1xDevelopmentError("derived event delta is not conserved")
    return {
        "delta_matrix": delta,
        "reactant_bond_matrix": reactant_bonds,
        "product_bond_matrix": product_bonds,
        "changed_pair_count": int(np.count_nonzero(delta_bonds)),
        "conservation_residual": weighted_total,
        "representation": "covalent-radius-binary-endpoint-edit-v1",
        "evidence": "derived_under_contract",
    }


def development_split_group(formula: str) -> str:
    """Use the recorded molecular formula as a stable development family."""

    if not isinstance(formula, str) or not formula.strip():
        raise Transition1xDevelopmentError("formula must be a non-empty string")
    return formula.strip()


def development_split(formula: str, *, seed: str = "transition1x-formula-v1") -> str:
    """Assign a deterministic formula-group split with 75/12.5/12.5 buckets."""

    group = development_split_group(formula)
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 8
    return "train" if bucket < 6 else "validation" if bucket == 6 else "test"


def development_sample(payload: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Build one validated, endpoint-derived sample without copying raw data."""

    if not isinstance(index, int) or index < 0:
        raise Transition1xDevelopmentError("index must be a nonnegative integer")
    try:
        reactant = payload["reactant"]
        product = payload["product"]
        transition_state = payload["transition_state"]
        numbers = reactant["charges"][index]
        reactant_coords = reactant["positions"][index]
        product_coords = product["positions"][index]
        ts_coords = transition_state["positions"][index]
        formula = reactant["formula"][index]
        reaction_id = reactant["rxn"][index]
    except (KeyError, IndexError, TypeError) as exc:
        raise Transition1xDevelopmentError(f"malformed Transition1x record {index}") from exc
    numbers, reactant_coords = _validate_geometry(numbers, reactant_coords, name="reactant")
    _, product_coords = _validate_geometry(numbers, product_coords, name="product")
    _, ts_coords = _validate_geometry(numbers, ts_coords, name="transition_state")
    event = conserved_event_delta(numbers, reactant_coords, product_coords)
    return {
        "index": index,
        "reaction_id": str(reaction_id),
        "formula": development_split_group(str(formula)),
        "split": development_split(str(formula)),
        "atomic_numbers": numbers,
        "reactant_coordinates": reactant_coords,
        "transition_state_coordinates": ts_coords,
        "event_delta": event["delta_matrix"],
        "event": event,
        "claim_limit": "quarantine endpoint-derived development diagnostic",
    }

