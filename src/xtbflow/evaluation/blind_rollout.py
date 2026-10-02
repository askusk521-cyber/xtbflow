"""Reactant-only rollout and parent-level benchmark helpers.

The training runner for the paired pilot intentionally computes teacher-forced
diagnostics.  This module is the small inference boundary used by issue #92:
all state used by :func:`blind_rollout` is constructed from the reactant
endpoint, while labels are accepted only by :func:`benchmark_rollouts` after
generation has finished.

The implementation is deliberately calculator-free.  It records candidate
attempts, failures and de-duplication so a later physical-validation stage can
consume the same ledger without changing the model-side budget.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from xtbflow.data.records import canonical_hash
from xtbflow.models.conserved_be import pack_be, packed_size, upper_triangle_indices
from xtbflow.models.event_decoder import DecodeError, decode_endpoint
from xtbflow.models.joint_flow import CONTROL_MODES


@dataclass(frozen=True)
class BlindRolloutConfig:
    """Fixed generation and accounting limits for one blind rollout."""

    nfe: int = 4
    candidate_cap: int = 64
    dedup_tolerance: float = 1.0e-4
    geometry_collision_tolerance: float = 0.50
    initial_event_noise_scale: float = 0.0
    initial_geometry_noise_scale: float = 0.0
    coupling_strength: float = 1.0
    conservation_projection: bool = True

    def __post_init__(self) -> None:
        if type(self.nfe) is not int or self.nfe < 1:
            raise ValueError("nfe must be a positive integer")
        if type(self.candidate_cap) is not int or self.candidate_cap < 1:
            raise ValueError("candidate_cap must be a positive integer")
        for name in ("dedup_tolerance", "geometry_collision_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("initial_event_noise_scale", "initial_geometry_noise_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if isinstance(self.coupling_strength, bool) or not isinstance(self.coupling_strength, (int, float)) or not math.isfinite(float(self.coupling_strength)) or self.coupling_strength < 0:
            raise ValueError("coupling_strength must be finite and nonnegative")
        if not isinstance(self.conservation_projection, bool):
            raise ValueError("conservation_projection must be boolean")


@dataclass(frozen=True)
class ReactantRolloutInput:
    """Model-visible tensors and chemistry identity for one reactant."""

    record_id: str
    parent_reaction_id: str
    symbols: tuple[str, ...]
    charge: int
    multiplicity: int
    event_state: torch.Tensor
    coordinates: torch.Tensor
    node_features: torch.Tensor
    atom_mask: torch.Tensor
    condition_features: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not self.record_id or not self.parent_reaction_id:
            raise ValueError("record and parent identities are required")
        n = len(self.symbols)
        if n < 1:
            raise ValueError("at least one atom is required")
        if type(self.charge) is not int or type(self.multiplicity) is not int or self.multiplicity < 1:
            raise ValueError("charge and multiplicity must be explicit integers")
        if self.event_state.ndim != 1 or self.coordinates.ndim != 2 or self.coordinates.shape[1:] != (3,):
            raise ValueError("reactant tensors have invalid shapes")
        if self.node_features.ndim != 2 or self.node_features.shape[0] != self.coordinates.shape[0]:
            raise ValueError("node_features must be [N,F]")
        if self.atom_mask.ndim != 1 or self.atom_mask.shape[0] != self.coordinates.shape[0] or self.atom_mask.dtype is not torch.bool:
            raise ValueError("atom_mask must be boolean [N]")
        if int(self.atom_mask.sum()) != n:
            raise ValueError("symbols must match the active atom mask")
        if self.event_state.shape[0] != packed_size(self.coordinates.shape[0]):
            raise ValueError("event_state width does not match coordinates")
        if self.coordinates.dtype != self.event_state.dtype or self.node_features.dtype != self.coordinates.dtype:
            raise ValueError("reactant tensors must share dtype")
        if self.coordinates.device != self.event_state.device or self.node_features.device != self.coordinates.device or self.atom_mask.device != self.coordinates.device:
            raise ValueError("reactant tensors must share device")
        if not all(torch.isfinite(value).all() for value in (self.event_state, self.coordinates, self.node_features)):
            raise ValueError("reactant tensors must be finite")
        if self.condition_features is not None:
            if self.condition_features.ndim != 1 or self.condition_features.dtype != self.coordinates.dtype or self.condition_features.device != self.coordinates.device or not torch.isfinite(self.condition_features).all():
                raise ValueError("condition_features must be finite 1-D tensor on the reactant device")


@dataclass(frozen=True)
class CandidateAttempt:
    """One candidate request, including failures before de-duplication."""

    candidate_index: int
    candidate_hash: str | None
    event_key: str | None
    geometry_key: str | None
    status: str
    failure_reason: str | None
    event_legal: bool
    geometry_valid: bool
    geometry_quality: Mapping[str, float | bool]
    event_packed: tuple[int, ...] | None = None
    geometry: tuple[tuple[float, float, float], ...] | None = None


@dataclass(frozen=True)
class BlindRolloutResult:
    """Auditable output of one reactant-only rollout."""

    record_id: str
    parent_reaction_id: str
    arm_id: str
    seed: int
    nfe: int
    candidate_cap: int
    actual_candidate_attempts: int
    actual_flow_evaluations: int
    attempts: tuple[CandidateAttempt, ...]
    accepted: tuple[CandidateAttempt, ...]
    physical_validation_calls: int = 0

    @property
    def failure_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.attempts:
            if row.status == "failure" and row.failure_reason:
                counts[row.failure_reason] = counts.get(row.failure_reason, 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failure_counts"] = self.failure_counts
        return payload


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=0, keepdims=True)


def _matrix_state(bonds: np.ndarray, atomic_numbers: Sequence[int], max_atoms: int) -> np.ndarray:
    # Match the state representation used by the pilot training runner.  The
    # diagonal stores remaining valence electrons, while off-diagonals store
    # bond orders.
    valence = {1: 1.0, 6: 4.0, 7: 5.0, 8: 6.0}
    matrix = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    count = len(atomic_numbers)
    matrix[:count, :count] = np.asarray(bonds, dtype=np.float32)
    degree = np.asarray(bonds, dtype=np.float32).sum(axis=1)
    for index, number in enumerate(atomic_numbers):
        matrix[index, index] = valence[int(number)] - float(degree[index])
    return matrix


def _node_features(sample: Any, max_atoms: int) -> np.ndarray:
    element_index = {1: 0, 6: 1, 7: 2, 8: 3}
    valence = {1: 1.0, 6: 4.0, 7: 5.0, 8: 6.0}
    values = np.zeros((max_atoms, 8), dtype=np.float32)
    degree = np.asarray(sample.reactant_bonds, dtype=np.float32).sum(axis=1)
    for index, number in enumerate(sample.atomic_numbers):
        number = int(number)
        values[index, element_index[number]] = 1.0
        values[index, 4] = valence[number] / 6.0
        values[index, 5] = number / 8.0
        values[index, 6] = 1.0
        values[index, 7] = float(degree[index]) / 4.0
    return values


def make_reactant_input(sample: Any, max_atoms: int, *, dtype: torch.dtype = torch.float32, device: torch.device | None = None) -> ReactantRolloutInput:
    """Build tensors solely from a sample's reactant-side fields.

    ``sample`` may carry product/TS labels (as ``DftDaSample`` does), but this
    function never reads them.  It is therefore suitable for label-replacement
    invariance checks and for inference code that intentionally seals labels.
    """

    n_atoms = len(sample.atomic_numbers)
    if type(max_atoms) is not int or max_atoms < n_atoms:
        raise ValueError("max_atoms must cover the sample atom count")
    reactant = _center(np.asarray(sample.reactant_coordinates, dtype=np.float32))
    state_matrix = _matrix_state(np.asarray(sample.reactant_bonds), sample.atomic_numbers, max_atoms)
    coordinates = torch.zeros((max_atoms, 3), dtype=dtype, device=device)
    coordinates[:n_atoms] = torch.from_numpy(reactant).to(device=device, dtype=dtype)
    node_features = torch.from_numpy(_node_features(sample, max_atoms)).to(device=device, dtype=dtype)
    event_state = pack_be(torch.from_numpy(state_matrix).to(device=device, dtype=dtype))
    atom_mask = torch.zeros((max_atoms,), dtype=torch.bool, device=device)
    atom_mask[:n_atoms] = True
    condition = torch.tensor(
        [0.0, 0.0, n_atoms / max_atoms, sum(int(number) != 1 for number in sample.atomic_numbers) / max_atoms],
        dtype=dtype,
        device=device,
    )
    record = sample.record
    symbols = tuple({1: "H", 6: "C", 7: "N", 8: "O"}[int(number)] for number in sample.atomic_numbers)
    return ReactantRolloutInput(
        record_id=record.record_id,
        parent_reaction_id=record.parent_reaction_id,
        symbols=symbols,
        charge=int(record.reactant["charge"]),
        multiplicity=int(record.reactant["multiplicity"]),
        event_state=event_state,
        coordinates=coordinates,
        node_features=node_features,
        atom_mask=atom_mask,
        condition_features=condition,
    )


def _active_packed_indices(max_atoms: int, n_atoms: int, *, device: torch.device) -> torch.Tensor:
    rows, cols = upper_triangle_indices(max_atoms, device=device)
    return torch.nonzero((rows < n_atoms) & (cols < n_atoms), as_tuple=False).flatten()


def _rounded_geometry_key(geometry: np.ndarray, tolerance: float) -> str:
    rounded = tuple(tuple(int(round(float(value) / tolerance)) for value in row) for row in geometry.tolist())
    return canonical_hash({"geometry": rounded})


def _geometry_quality(geometry: np.ndarray, tolerance: float) -> tuple[bool, dict[str, float | bool], str | None]:
    if geometry.ndim != 2 or geometry.shape[1:] != (3,) or not np.isfinite(geometry).all():
        return False, {"finite": False}, "geometry_nonfinite"
    centered = geometry - geometry.mean(axis=0, keepdims=True)
    displacement = np.linalg.norm(centered, axis=1)
    if len(geometry) > 1:
        differences = geometry[:, None, :] - geometry[None, :, :]
        distances = np.linalg.norm(differences, axis=-1)
        distances = distances[np.triu_indices(len(geometry), k=1)]
        min_distance = float(np.min(distances)) if distances.size else float("inf")
    else:
        min_distance = float("inf")
    quality: dict[str, float | bool] = {
        "finite": True,
        "rms_displacement": float(np.sqrt(np.mean(centered * centered))),
        "max_abs_coordinate": float(np.max(np.abs(geometry))),
        "min_pair_distance": min_distance,
        "collision": bool(min_distance < tolerance),
    }
    if bool(quality["collision"]):
        return False, quality, "geometry_collision"
    return True, quality, None


def _decoded_event_key(decoded: Any) -> str:
    """Hash only the discrete bond endpoint, excluding repaired lone pairs."""

    matrix = decoded.state.be
    bonds = tuple(
        (int(matrix[left][right]) if left != right else 0)
        for left in range(len(matrix))
        for right in range(left, len(matrix))
    )
    return canonical_hash({"bond_endpoint": bonds})


def _candidate_key(event_key: str, geometry_key: str) -> str:
    return canonical_hash({"event": event_key, "geometry": geometry_key})


def blind_rollout(
    model: Any,
    reactant: ReactantRolloutInput,
    *,
    arm_id: str,
    seed: int,
    config: BlindRolloutConfig | None = None,
) -> BlindRolloutResult:
    """Generate bounded candidates from ``t=0`` using no target fields.

    Candidate requests are batched for deterministic accounting.  Every
    request receives the same model input; optional initial noise is generated
    from ``seed`` and is independent of labels.  The returned attempt list is
    retained even when all candidates de-duplicate to one endpoint.
    """

    config = config or BlindRolloutConfig()
    if not isinstance(arm_id, str) or arm_id not in CONTROL_MODES:
        raise ValueError(f"arm_id must be one of {CONTROL_MODES}")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    model.eval()
    device = reactant.coordinates.device
    dtype = reactant.coordinates.dtype
    count = config.candidate_cap
    event_state = reactant.event_state.unsqueeze(0).expand(count, -1).clone()
    coordinates = reactant.coordinates.unsqueeze(0).expand(count, -1, -1).clone()
    node_features = reactant.node_features.unsqueeze(0).expand(count, -1, -1).clone()
    atom_mask = reactant.atom_mask.unsqueeze(0).expand(count, -1).clone()
    condition = None if reactant.condition_features is None else reactant.condition_features.unsqueeze(0).expand(count, -1).clone()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed & ((1 << 63) - 1))
    if config.initial_event_noise_scale:
        noise = torch.randn(event_state.shape, dtype=dtype, device=device, generator=generator)
        # A noise perturbation is part of the event branch only and is
        # projected using the exact model projector before generation starts.
        projector = model.event_projector
        event_state = event_state + float(config.initial_event_noise_scale) * projector.project(noise)
    if config.initial_geometry_noise_scale:
        noise = torch.randn(coordinates.shape, dtype=dtype, device=device, generator=generator)
        noise = noise - noise.mean(dim=1, keepdim=True)
        coordinates = coordinates + float(config.initial_geometry_noise_scale) * noise * atom_mask[..., None]

    dt = 1.0 / config.nfe
    with torch.no_grad():
        for step in range(config.nfe):
            tau = float(step) * dt
            output = model.forward_control(
                arm_id,
                event_state,
                coordinates,
                node_features,
                atom_mask,
                tau=tau,
                dt=dt,
                coupling_strength=config.coupling_strength,
                condition_features=condition,
                conservation_projection=config.conservation_projection,
            )
            event_state = event_state + dt * output.event_velocity
            coordinates = coordinates + dt * output.geometry_velocity

    n_atoms = len(reactant.symbols)
    packed_indices = _active_packed_indices(int((math.isqrt(8 * reactant.event_state.shape[0] + 1) - 1) // 2), n_atoms, device=device)
    attempts: list[CandidateAttempt] = []
    accepted: list[CandidateAttempt] = []
    seen: set[str] = set()
    for index in range(count):
        geometry_array = coordinates[index, :n_atoms].detach().cpu().numpy().astype(np.float64, copy=False)
        geometry_valid, quality, geometry_failure = _geometry_quality(geometry_array, config.geometry_collision_tolerance)
        event_values = event_state[index].index_select(0, packed_indices).detach().cpu()
        event_key: str | None = None
        event_packed: tuple[int, ...] | None = None
        event_failure: str | None = None
        try:
            decoded = decode_endpoint(event_values, reactant.symbols, reactant.charge, reactant.multiplicity)
            event_packed = decoded.packed
            event_key = _decoded_event_key(decoded)
        except (DecodeError, ValueError, RuntimeError) as exc:
            event_failure = f"event_illegal:{type(exc).__name__}"
        if event_failure or geometry_failure:
            reason = event_failure or geometry_failure
            attempts.append(CandidateAttempt(index, None, event_key, None, "failure", reason, False, geometry_valid, quality, event_packed, tuple(tuple(float(value) for value in row) for row in geometry_array)))
            continue
        geometry_key = _rounded_geometry_key(geometry_array, config.dedup_tolerance)
        candidate_hash = _candidate_key(event_key, geometry_key)
        if candidate_hash in seen:
            attempts.append(CandidateAttempt(index, candidate_hash, event_key, geometry_key, "duplicate", "duplicate_candidate", True, True, quality, event_packed, tuple(tuple(float(value) for value in row) for row in geometry_array)))
            continue
        seen.add(candidate_hash)
        row = CandidateAttempt(index, candidate_hash, event_key, geometry_key, "accepted", None, True, True, quality, event_packed, tuple(tuple(float(value) for value in row) for row in geometry_array))
        attempts.append(row)
        accepted.append(row)
    return BlindRolloutResult(
        record_id=reactant.record_id,
        parent_reaction_id=reactant.parent_reaction_id,
        arm_id=arm_id,
        seed=seed,
        nfe=config.nfe,
        candidate_cap=config.candidate_cap,
        actual_candidate_attempts=count,
        actual_flow_evaluations=count * config.nfe,
        physical_validation_calls=0,
        attempts=tuple(attempts),
        accepted=tuple(accepted),
    )


def _kabsch_rmsd(predicted: np.ndarray, reference: np.ndarray) -> float:
    if predicted.shape != reference.shape or predicted.ndim != 2 or predicted.shape[1:] != (3,):
        return float("inf")
    pred = _center(predicted)
    ref = _center(reference)
    covariance = pred.T @ ref
    left, _, right_transpose = np.linalg.svd(covariance)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1.0
        rotation = left @ right_transpose
    aligned = pred @ rotation
    return float(np.sqrt(np.mean((aligned - ref) ** 2)))


def _record_benchmark(result: BlindRolloutResult, sample: Any, geometry_threshold: float) -> dict[str, Any]:
    product_bonds = np.asarray(sample.product_bonds)
    target_event_key = canonical_hash(
        {
            "bond_endpoint": tuple(
                (int(product_bonds[left, right]) if left != right else 0)
                for left in range(len(product_bonds))
                for right in range(left, len(product_bonds))
            )
        }
    )
    target_geometry = np.asarray(sample.ts_coordinates, dtype=np.float64)
    event_hits = [row.event_key == target_event_key for row in result.accepted]
    rmsds = [_kabsch_rmsd(np.asarray(row.geometry, dtype=np.float64), target_geometry) for row in result.accepted if row.geometry is not None]
    best_rmsd = min(rmsds) if rmsds else float("inf")
    paired_hit = any(hit and _kabsch_rmsd(np.asarray(row.geometry, dtype=np.float64), target_geometry) <= geometry_threshold for row, hit in zip(result.accepted, event_hits) if row.geometry is not None)
    return {
        "record_id": result.record_id,
        "parent_reaction_id": result.parent_reaction_id,
        "arm_id": result.arm_id,
        "seed": result.seed,
        "candidate_attempts": result.actual_candidate_attempts,
        "flow_evaluations": result.actual_flow_evaluations,
        "accepted_candidates": len(result.accepted),
        "failure_count": sum(row.status == "failure" for row in result.attempts),
        "duplicate_count": sum(row.status == "duplicate" for row in result.attempts),
        "event_hit": bool(any(event_hits)),
        "geometry_best_rmsd": float(best_rmsd),
        "geometry_hit": bool(best_rmsd <= geometry_threshold),
        "paired_event_geometry_hit": bool(paired_hit),
    }


def benchmark_rollouts(results: Iterable[BlindRolloutResult], samples_by_record: Mapping[str, Any], *, geometry_threshold: float = 0.50, bootstrap_resamples: int = 0, bootstrap_seed: int = 0) -> dict[str, Any]:
    """Score blind candidates record-wise and parent-macro-wise.

    ``samples_by_record`` is intentionally a separate argument from rollout
    inputs, making it impossible for generation to access labels by accident.
    The parent bootstrap resamples parent IDs, so repeated records or random
    seeds are never counted as independent chemistry samples.
    """

    if geometry_threshold <= 0 or not math.isfinite(float(geometry_threshold)):
        raise ValueError("geometry_threshold must be finite and positive")
    rows = []
    for result in results:
        sample = samples_by_record.get(result.record_id)
        if sample is None:
            raise KeyError(f"missing benchmark sample: {result.record_id}")
        rows.append(_record_benchmark(result, sample, geometry_threshold))
    if not rows:
        raise ValueError("at least one rollout result is required")
    metrics = ("event_hit", "geometry_hit", "paired_event_geometry_hit", "geometry_best_rmsd", "accepted_candidates", "failure_count", "duplicate_count", "candidate_attempts", "flow_evaluations")
    record_weighted: dict[str, float] = {}
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        record_weighted[metric] = float(np.mean(values))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["parent_reaction_id"], []).append(row)
    parent_rows = []
    for parent_id in sorted(grouped):
        group = grouped[parent_id]
        parent_rows.append({"parent_reaction_id": parent_id, "record_count": len(group), **{metric: float(np.mean([float(row[metric]) for row in group])) for metric in metrics}})
    parent_macro = {metric: float(np.mean([row[metric] for row in parent_rows])) for metric in metrics}
    bootstrap: dict[str, Any] = {}
    if type(bootstrap_resamples) is not int or bootstrap_resamples < 0:
        raise ValueError("bootstrap_resamples must be a nonnegative integer")
    if bootstrap_resamples:
        rng = np.random.default_rng(bootstrap_seed)
        draws = rng.integers(0, len(parent_rows), size=(bootstrap_resamples, len(parent_rows)))
        for metric in metrics:
            values = np.asarray([row[metric] for row in parent_rows], dtype=np.float64)
            means = values[draws].mean(axis=1)
            bootstrap[metric] = {
                "parent_count": len(parent_rows),
                "resamples": bootstrap_resamples,
                "seed": bootstrap_seed,
                "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
            }
    return {
        "schema": "xtbflow-blind-rollout-benchmark/v1",
        "record_count": len(rows),
        "parent_count": len(parent_rows),
        "geometry_threshold": float(geometry_threshold),
        "record_weighted": record_weighted,
        "parent_macro": parent_macro,
        "parent_metrics": parent_rows,
        "record_metrics": rows,
        "parent_bootstrap": bootstrap,
    }
