"""Reactant-only generation and post-hoc scoring helpers.

The real-data training runner uses product/event and TS labels to construct a
flow-matching target.  That is valid for optimization, but it must not be
reused as a deployment evaluation.  This module keeps the two boundaries
separate:

* :class:`ReactantView` contains only fields visible to a generator;
* :func:`generate_candidates` integrates a frozen model from that view;
* :func:`score_candidates` consumes labels only after candidates are frozen.

The functions are deterministic for a fixed model, input and integration
configuration.  Stochastic sampling is intentionally not invented here: the
current checkpoints were trained as deterministic vector fields, so the report
records the actual number of generated candidates instead of treating a
``candidate_cap`` as if it had been filled.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from xtbflow.data.records import canonical_hash
from xtbflow.models import (
    JointFlowRuntimeConfig,
    decode_endpoint,
    pack_be,
    total_electron_projector,
    unpack_be,
    upper_triangle_indices,
)
from xtbflow.sampling import control_euler_step
from xtbflow.training import load_joint_checkpoint


ELEMENT_SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O"}
ELEMENT_INDEX = {1: 0, 6: 1, 7: 2, 8: 3}
VALENCE = {1: 1.0, 6: 4.0, 7: 5.0, 8: 6.0}
_FORBIDDEN_INPUT_TOKENS = frozenset(
    {"product", "event", "ts", "reference", "target", "sealed", "label"}
)
_REACTANT_INPUT_KEYS = frozenset(
    {
        "schema",
        "record_id",
        "parent_reaction_id",
        "family_id",
        "symbols",
        "map_ids",
        "reactant_coordinates_angstrom",
        "reactant_bonds",
        "charge",
        "multiplicity",
    }
)


def _matrix(value: Sequence[Sequence[float]], name: str, *, size: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or not array.shape[0]:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if size is not None and array.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size})")
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f"{name} must contain finite nonnegative values")
    if not np.allclose(array, array.T, atol=1e-9, rtol=0):
        raise ValueError(f"{name} must be symmetric")
    if not np.allclose(np.diag(array), 0.0, atol=1e-9, rtol=0):
        raise ValueError(f"{name} must have a zero diagonal")
    return array


def _coordinates(value: Sequence[Sequence[float]], size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size, 3) or not np.isfinite(array).all():
        raise ValueError(f"coordinates must be finite with shape ({size}, 3)")
    return array


def _integer_state(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an explicit integer")
    return value


@dataclass(frozen=True)
class ReactantView:
    """The complete generator-visible view for one reaction record.

    No product, transition-state, event label, or reference score is allowed
    in this object.  The arrays are copied on construction so a scorer cannot
    mutate the input used by generation.
    """

    record_id: str
    parent_reaction_id: str
    family_id: str
    atomic_numbers: tuple[int, ...]
    map_ids: tuple[int, ...]
    coordinates: np.ndarray
    bonds: np.ndarray
    charge: int
    multiplicity: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReactantView":
        """Construct a view from the public reactant-only JSONL contract."""

        if not isinstance(value, Mapping):
            raise ValueError("reactant input must be a mapping")
        _reject_target_fields(value)
        unknown = set(value) - _REACTANT_INPUT_KEYS
        if unknown:
            raise ValueError(f"unsupported reactant input fields: {sorted(unknown)}")
        if value.get("schema") != "xtbflow-reactant-input/v1":
            raise ValueError("unsupported reactant input schema")
        symbols = value.get("symbols")
        if not isinstance(symbols, Sequence) or isinstance(symbols, (str, bytes)):
            raise ValueError("reactant input requires an ordered symbols list")
        atomic_numbers = []
        inverse = {symbol: number for number, symbol in ELEMENT_SYMBOLS.items()}
        for symbol in symbols:
            if symbol not in inverse:
                raise ValueError("reactant symbols must be CHNO")
            atomic_numbers.append(inverse[symbol])
        return cls(
            record_id=value.get("record_id"),
            parent_reaction_id=value.get("parent_reaction_id", value.get("record_id")),
            family_id=value.get("family_id", value.get("record_id")),
            atomic_numbers=tuple(atomic_numbers),
            map_ids=tuple(value.get("map_ids", ())),
            coordinates=value.get("reactant_coordinates_angstrom"),
            bonds=value.get("reactant_bonds"),
            charge=value.get("charge"),
            multiplicity=value.get("multiplicity"),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": "xtbflow-reactant-input/v1",
            "record_id": self.record_id,
            "parent_reaction_id": self.parent_reaction_id,
            "family_id": self.family_id,
            "symbols": list(self.symbols),
            "map_ids": list(self.map_ids),
            "reactant_coordinates_angstrom": self.coordinates.tolist(),
            "reactant_bonds": self.bonds.tolist(),
            "charge": self.charge,
            "multiplicity": self.multiplicity,
        }

    def __post_init__(self) -> None:
        for name in ("record_id", "parent_reaction_id", "family_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        numbers = tuple(self.atomic_numbers)
        if not numbers or any(type(value) is not int or value not in ELEMENT_SYMBOLS for value in numbers):
            raise ValueError("atomic_numbers must be a nonempty CHNO tuple")
        object.__setattr__(self, "atomic_numbers", numbers)
        maps = tuple(self.map_ids)
        if len(maps) != len(numbers) or any(type(value) is not int or value < 1 for value in maps) or len(set(maps)) != len(maps):
            raise ValueError("map_ids must be a unique positive integer per atom")
        object.__setattr__(self, "map_ids", maps)
        coordinates = _coordinates(self.coordinates, len(numbers)).copy()
        bonds = _matrix(self.bonds, "bonds", size=len(numbers)).copy()
        coordinates.setflags(write=False)
        bonds.setflags(write=False)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "bonds", bonds)
        charge = _integer_state(self.charge, "charge")
        multiplicity = _integer_state(self.multiplicity, "multiplicity")
        if multiplicity != 1:
            raise ValueError("generation V0 requires singlet multiplicity=1")
        object.__setattr__(self, "charge", charge)
        object.__setattr__(self, "multiplicity", multiplicity)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(ELEMENT_SYMBOLS[number] for number in self.atomic_numbers)

    @property
    def input_fingerprint(self) -> str:
        return canonical_hash(
            {
                "record_id": self.record_id,
                "parent_reaction_id": self.parent_reaction_id,
                "family_id": self.family_id,
                "atomic_numbers": list(self.atomic_numbers),
                "map_ids": list(self.map_ids),
                "coordinates": np.asarray(self.coordinates).round(12).tolist(),
                "bonds": np.asarray(self.bonds).round(12).tolist(),
                "charge": self.charge,
                "multiplicity": self.multiplicity,
            }
        )


ReactantInput = ReactantView


def _reject_target_fields(value: Any, path: str = "root") -> None:
    """Reject target-derived keys at every nesting level of an input row."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string field name")
            lowered = key.lower()
            if any(token in lowered for token in _FORBIDDEN_INPUT_TOKENS):
                raise ValueError(f"forbidden target field at {path}.{key}")
            _reject_target_fields(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_target_fields(nested, f"{path}[{index}]")


@dataclass(frozen=True)
class ReferenceLabels:
    """Labels used only after generation for a frozen candidate set."""

    record_id: str
    map_ids: tuple[int, ...]
    product_bonds: np.ndarray
    ts_coordinates: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id.strip():
            raise ValueError("record_id must be a nonempty string")
        maps = tuple(self.map_ids)
        if len(maps) != np.asarray(self.product_bonds).shape[0] or any(type(value) is not int or value < 1 for value in maps) or len(set(maps)) != len(maps):
            raise ValueError("map_ids must be a unique positive integer per atom")
        object.__setattr__(self, "map_ids", maps)
        product = _matrix(self.product_bonds, "product_bonds")
        coords = _coordinates(self.ts_coordinates, product.shape[0])
        object.__setattr__(self, "product_bonds", product.copy())
        object.__setattr__(self, "ts_coordinates", coords.copy())
        self.product_bonds.setflags(write=False)
        self.ts_coordinates.setflags(write=False)


@dataclass(frozen=True)
class GeneratedCandidate:
    """One frozen decoded endpoint and its generated geometry."""

    record_id: str
    control_mode: str
    coordinates: tuple[tuple[float, float, float], ...]
    packed_event: tuple[int, ...]
    steps: int
    candidate_index: int = 0

    def to_mapping(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "control_mode": self.control_mode,
            "coordinates_angstrom": [list(row) for row in self.coordinates],
            "packed_event": list(self.packed_event),
            "steps": self.steps,
            "candidate_index": self.candidate_index,
            "candidate_fingerprint": self.fingerprint(),
            "decode_status": "accepted",
        }

    def fingerprint(self) -> str:
        return canonical_hash(
            {
                "record_id": self.record_id,
                "control_mode": self.control_mode,
                "coordinates": self.coordinates,
                "packed_event": self.packed_event,
                "steps": self.steps,
                "candidate_index": self.candidate_index,
            }
        )


@dataclass(frozen=True)
class GenerationBatch:
    """Bounded output and explicit decode accounting for one input."""

    candidates: tuple[GeneratedCandidate, ...]
    attempted: int
    rejected: int
    rejection_reasons: tuple[str, ...]
    steps: int
    candidate_cap: int

    @property
    def accepted(self) -> int:
        return len(self.candidates)

    @property
    def rejection_rate(self) -> float:
        return self.rejected / self.attempted if self.attempted else 0.0


def reactant_view_from_sample(sample: Any) -> ReactantView:
    """Extract only reactant fields from a Diels--Alder adapter sample.

    This intentionally does not touch ``product_coordinates``,
    ``ts_coordinates`` or ``product_bonds``.  The type is kept structural so
    the helper can be tested with a fixture without loading source labels.
    """

    record = sample.record
    reactant = record.reactant
    return ReactantView(
        record_id=record.record_id,
        parent_reaction_id=record.parent_reaction_id,
        family_id=record.family_id,
        atomic_numbers=tuple(sample.atomic_numbers),
        map_ids=tuple(sample.map_ids),
        coordinates=sample.reactant_coordinates,
        bonds=sample.reactant_bonds,
        charge=reactant.get("charge"),
        multiplicity=reactant.get("multiplicity"),
    )


def reference_labels_from_sample(sample: Any) -> ReferenceLabels:
    """Extract labels for scoring, never for model input construction."""

    return ReferenceLabels(
        record_id=sample.record.record_id,
        map_ids=tuple(sample.map_ids),
        product_bonds=sample.product_bonds,
        ts_coordinates=sample.ts_coordinates,
    )


def _center(coordinates: np.ndarray) -> np.ndarray:
    return coordinates - coordinates.mean(axis=0, keepdims=True)


def _matrix_state(bonds: np.ndarray, atomic_numbers: tuple[int, ...], max_atoms: int) -> np.ndarray:
    matrix = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    n_atoms = len(atomic_numbers)
    matrix[:n_atoms, :n_atoms] = bonds.astype(np.float32)
    degree = bonds.sum(axis=1)
    for index, number in enumerate(atomic_numbers):
        matrix[index, index] = VALENCE[number] - float(degree[index])
    return matrix


def _node_features(view: ReactantView, max_atoms: int) -> np.ndarray:
    features = np.zeros((max_atoms, 8), dtype=np.float32)
    degree = view.bonds.sum(axis=1)
    for index, number in enumerate(view.atomic_numbers):
        features[index, ELEMENT_INDEX[number]] = 1.0
        features[index, 4] = VALENCE[number] / 6.0
        features[index, 5] = number / 8.0
        features[index, 6] = 1.0
        features[index, 7] = float(degree[index]) / 4.0
    return features


def _model_inputs(view: ReactantView, max_atoms: int, *, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, ...]:
    if len(view.atomic_numbers) > max_atoms:
        raise ValueError("max_atoms is smaller than the reactant atom count")
    n_atoms = len(view.atomic_numbers)
    state = pack_be(torch.from_numpy(_matrix_state(view.bonds, view.atomic_numbers, max_atoms)).unsqueeze(0)).to(dtype=dtype, device=device)
    coordinates = torch.zeros((1, max_atoms, 3), dtype=dtype, device=device)
    coordinates[0, :n_atoms] = torch.from_numpy(_center(view.coordinates).astype(np.float32).copy()).to(dtype=dtype, device=device)
    node_features = torch.from_numpy(_node_features(view, max_atoms)).unsqueeze(0).to(dtype=dtype, device=device)
    atom_mask = torch.zeros((1, max_atoms), dtype=torch.bool, device=device)
    atom_mask[0, :n_atoms] = True
    rows, cols = upper_triangle_indices(max_atoms, device=device)
    event_mask = ((rows < n_atoms) & (cols < n_atoms)).unsqueeze(0)
    condition = torch.tensor(
        [[0.0, 0.0, n_atoms / max_atoms, sum(number != 1 for number in view.atomic_numbers) / max_atoms]],
        dtype=dtype,
        device=device,
    )
    return state, coordinates, node_features, atom_mask, event_mask, condition


def reactant_packed_state(view: ReactantView) -> tuple[int, ...]:
    """Return the legal packed endpoint corresponding to the reactant state."""

    matrix = _matrix_state(view.bonds, view.atomic_numbers, len(view.atomic_numbers))
    return tuple(int(value) for value in pack_be(torch.from_numpy(matrix)).tolist())


def centered_reactant_coordinates(view: ReactantView) -> tuple[tuple[float, float, float], ...]:
    """Return the coordinate frame used by generation and scoring."""

    centered = _center(view.coordinates)
    return tuple(tuple(float(value) for value in row) for row in centered.tolist())


def _finite_positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _decode_generated_event(event_state: torch.Tensor, view: ReactantView, max_atoms: int) -> tuple[int, ...]:
    matrix = unpack_be(event_state.detach().cpu(), max_atoms)
    active = matrix[0, : len(view.atomic_numbers), : len(view.atomic_numbers)]
    packed = pack_be(active.unsqueeze(0))[0]
    decoded = decode_endpoint(packed, view.symbols, view.charge, view.multiplicity)
    return decoded.packed


@torch.no_grad()
def generate_candidates(
    model: Any,
    view: ReactantView,
    *,
    max_atoms: int,
    control_mode: str,
    steps: int = 16,
    candidate_cap: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    coupling_strength: float = 1.0,
    conservation_projection: bool = True,
) -> GenerationBatch:
    """Generate and decode candidates from a reactant-only view.

    ``candidate_cap`` is a hard upper bound.  Since the current field is
    deterministic, this function attempts one trajectory and reports one
    candidate at most; it never fabricates extra samples to satisfy the cap.
    """

    _finite_positive_int(max_atoms, "max_atoms")
    _finite_positive_int(steps, "steps")
    _finite_positive_int(candidate_cap, "candidate_cap")
    if not isinstance(control_mode, str) or not control_mode.strip():
        raise ValueError("control_mode must be a nonempty string")
    if not isinstance(conservation_projection, bool):
        raise ValueError("conservation_projection must be boolean")
    device = torch.device(device)
    state, coordinates, node_features, atom_mask, _, condition = _model_inputs(
        view, max_atoms, dtype=dtype, device=device
    )
    model.eval()
    dt = 1.0 / steps
    for step in range(steps):
        state, coordinates, _ = control_euler_step(
            model,
            state,
            coordinates,
            node_features,
            atom_mask,
            mode=control_mode,
            tau=step * dt,
            dt=dt,
            coupling_strength=coupling_strength,
            condition_features=condition,
            conservation_projection=conservation_projection,
        )
    try:
        packed = _decode_generated_event(state, view, max_atoms)
    except (ValueError, RuntimeError, TypeError) as exc:
        return GenerationBatch(
            candidates=(),
            attempted=1,
            rejected=1,
            rejection_reasons=(f"{type(exc).__name__}:{exc}",),
            steps=steps,
            candidate_cap=candidate_cap,
        )
    active_coordinates = coordinates[0, : len(view.atomic_numbers)].detach().cpu().numpy()
    candidate = GeneratedCandidate(
        record_id=view.record_id,
        control_mode=control_mode,
        coordinates=tuple(tuple(float(value) for value in row) for row in active_coordinates.tolist()),
        packed_event=tuple(int(value) for value in packed),
        steps=steps,
    )
    return GenerationBatch((candidate,), 1, 0, (), steps, candidate_cap)


def _kabsch_align(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    ref = _center(reference)
    moving = _center(target)
    covariance = moving.T @ ref
    left, _, right_transpose = np.linalg.svd(covariance)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1.0
        rotation = left @ right_transpose
    return moving @ rotation


def _candidate_bonds(candidate: GeneratedCandidate, n_atoms: int, symbols: Sequence[str], charge: int, multiplicity: int) -> np.ndarray:
    decoded = decode_endpoint(candidate.packed_event, symbols, charge, multiplicity)
    matrix = np.asarray(decoded.state.be, dtype=np.float64)
    matrix[np.diag_indices_from(matrix)] = 0.0
    if matrix.shape != (n_atoms, n_atoms):
        raise ValueError("decoded candidate atom count does not match labels")
    return matrix


def score_candidates(
    view: ReactantView,
    labels: ReferenceLabels,
    candidates: Iterable[GeneratedCandidate],
) -> tuple[dict[str, Any], ...]:
    """Score a frozen candidate collection against reference labels.

    Candidate fingerprints are computed before any labels are touched by the
    caller; this function only returns post-hoc diagnostics.  It does not
    reorder or mutate the candidate collection.
    """

    if labels.record_id != view.record_id:
        raise ValueError("view and labels record_id must match")
    if labels.map_ids != view.map_ids:
        raise ValueError("view and labels map_ids must match")
    if labels.product_bonds.shape != view.bonds.shape:
        raise ValueError("product_bonds must match the reactant atom inventory")
    reference_ts = _kabsch_align(view.coordinates, labels.ts_coordinates)
    rows: list[dict[str, Any]] = []
    for candidate in tuple(candidates):
        if candidate.record_id != view.record_id:
            raise ValueError("candidate record_id does not match the labels")
        coordinates = np.asarray(candidate.coordinates, dtype=np.float64)
        if coordinates.shape != view.coordinates.shape or not np.isfinite(coordinates).all():
            raise ValueError("candidate coordinates are malformed")
        try:
            predicted_bonds = _candidate_bonds(
                candidate,
                len(view.atomic_numbers),
                view.symbols,
                view.charge,
                view.multiplicity,
            )
            event_exact = bool(np.array_equal(predicted_bonds, labels.product_bonds))
            decode_status = "accepted"
            decode_error = None
        except (ValueError, RuntimeError, TypeError) as exc:
            predicted_bonds = None
            event_exact = False
            decode_status = "decode_failed"
            decode_error = f"{type(exc).__name__}:{exc}"
        geometry_mse = float(np.mean((coordinates - reference_ts) ** 2))
        rows.append(
            {
                "record_id": view.record_id,
                "parent_reaction_id": view.parent_reaction_id,
                "family_id": view.family_id,
                "control_mode": candidate.control_mode,
                "candidate_index": candidate.candidate_index,
                "candidate_fingerprint": candidate.fingerprint(),
                "decode_status": decode_status,
                "decode_error": decode_error,
                "event_exact_match": event_exact,
                "event_bond_mse": None
                if predicted_bonds is None
                else float(np.mean((predicted_bonds - labels.product_bonds) ** 2)),
                "geometry_endpoint_mse": geometry_mse,
                "steps": candidate.steps,
            }
        )
    return tuple(rows)


def candidate_fingerprints(candidates: Iterable[GeneratedCandidate]) -> tuple[str, ...]:
    """Return stable candidate identities for label-replacement tests."""

    return tuple(candidate.fingerprint() for candidate in candidates)


def candidate_fingerprint(candidate: GeneratedCandidate) -> str:
    """Compatibility helper for reports that contain one candidate per row."""

    return candidate.fingerprint()


def load_reactant_inputs(path: str | Path) -> tuple[ReactantView, ...]:
    """Load a target-free JSONL input manifest without silently dropping rows."""

    source = Path(path)
    rows: list[ReactantView] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
            view = ReactantView.from_mapping(value)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid reactant input line {line_number}: {exc}") from exc
        if view.record_id in seen:
            raise ValueError(f"duplicate reactant record_id: {view.record_id}")
        seen.add(view.record_id)
        rows.append(view)
    if not rows:
        raise ValueError("reactant input manifest is empty")
    return tuple(rows)


def checkpoint_atom_count(payload: Mapping[str, Any]) -> int:
    """Infer the packed event atom count from a checkpoint state dictionary."""

    model = payload.get("model") if isinstance(payload, Mapping) else None
    if not isinstance(model, Mapping):
        raise ValueError("checkpoint does not contain a model state")
    matrix = model.get("constraint_matrix")
    if isinstance(matrix, torch.Tensor):
        shape = tuple(matrix.shape)
    else:
        shape = getattr(matrix, "shape", None)
    if not shape or len(shape) != 2 or shape[0] != 1:
        raise ValueError("checkpoint constraint_matrix has an invalid shape")
    features = int(shape[1])
    atoms = int((math.isqrt(8 * features + 1) - 1) // 2)
    if atoms * (atoms + 1) // 2 != features:
        raise ValueError("checkpoint event width is not a packed triangular size")
    return atoms


def load_generation_model(
    checkpoint: str | Path,
    runtime_config: str | Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.nn.Module, int, dict[str, Any]]:
    """Load a frozen checkpoint for inference and return its measured identity."""

    checkpoint_path = Path(checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    max_atoms = checkpoint_atom_count(payload)
    device = torch.device(device)
    projector = total_electron_projector(["C"] * max_atoms, dtype=dtype)
    config = JointFlowRuntimeConfig.load(runtime_config)
    model = config.build(projector).to(device=device, dtype=dtype)
    restore = load_joint_checkpoint(checkpoint_path, model, restore_rng=False)
    model.eval()
    return model, max_atoms, restore


def generate_candidate(
    model: Any,
    reactant: ReactantView,
    *,
    max_atoms: int,
    mode: str = "joint_bidirectional",
    steps: int = 16,
    candidate_cap: int = 1,
    conservation_projection: bool = True,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> GeneratedCandidate | None:
    """Generate one candidate for JSONL runners, returning ``None`` on reject."""

    batch = generate_candidates(
        model,
        reactant,
        max_atoms=max_atoms,
        control_mode=mode,
        steps=steps,
        candidate_cap=candidate_cap,
        conservation_projection=conservation_projection,
        device=device,
        dtype=dtype,
    )
    return batch.candidates[0] if batch.candidates else None


def label_replacement_invariant(
    model: Any,
    view: ReactantView,
    labels: Iterable[ReferenceLabels],
    *,
    max_atoms: int,
    control_mode: str,
    steps: int = 16,
    candidate_cap: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Run generation once per label set and assert labels cannot influence it."""

    label_sets = tuple(labels)
    if not label_sets:
        raise ValueError("at least one label set is required")
    if dtype is None:
        try:
            dtype = next(model.parameters()).dtype
        except (StopIteration, AttributeError):
            dtype = torch.float32
    batches = tuple(
        generate_candidates(
            model,
            view,
            max_atoms=max_atoms,
            control_mode=control_mode,
            steps=steps,
            candidate_cap=candidate_cap,
            device=device,
            dtype=dtype,
        )
        for _ in label_sets
    )
    fingerprints = tuple(candidate_fingerprints(batch.candidates) for batch in batches)
    invariant = all(value == fingerprints[0] for value in fingerprints[1:])
    if not invariant:
        raise AssertionError("generated candidates changed when reference labels were replaced")
    return {
        "invariant": True,
        "label_sets": len(label_sets),
        "candidate_fingerprints": fingerprints[0],
    }


__all__ = [
    "GeneratedCandidate",
    "GenerationBatch",
    "ReactantView",
    "ReferenceLabels",
    "candidate_fingerprints",
    "centered_reactant_coordinates",
    "generate_candidates",
    "label_replacement_invariant",
    "reactant_view_from_sample",
    "reactant_packed_state",
    "reference_labels_from_sample",
    "score_candidates",
]
