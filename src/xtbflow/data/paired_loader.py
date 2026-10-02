"""Leakage-safe loader for the real paired pilot.

The loader exposes two views only: ``reactant_input`` and
``training_supervision``.  Product coordinates and source/reference metadata
are never returned in a model batch.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch

from .dft_da import DftDaSample, load_dft_da_samples
from .records import canonical_hash
from .track_b import TrackBRecord, load_track_b_jsonl, audit_track_b_leakage


@dataclass(frozen=True)
class PairedBatch:
    reactant_input: dict[str, torch.Tensor]
    training_supervision: dict[str, torch.Tensor]
    metadata: tuple[dict[str, str], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "reactant_input": self.reactant_input,
            "training_supervision": self.training_supervision,
            "metadata": self.metadata,
        }


def _event_matrix(sample: DftDaSample) -> np.ndarray:
    return (sample.product_bonds - sample.reactant_bonds).astype(np.float32, copy=False)


def reactant_input_fingerprint(record: TrackBRecord) -> str:
    """Return the contract fingerprint used for exact-input leakage checks."""

    return record.input_fingerprint()


class PairedPilotLoader:
    """Small deterministic batch loader backed by audited source rows."""

    def __init__(self, samples: Sequence[DftDaSample], *, batch_size: int = 8, max_atoms: int | None = None, shuffle: bool = False, seed: int = 0):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not samples:
            raise ValueError("paired pilot cannot be empty")
        self.samples = tuple(sorted(samples, key=lambda row: row.record.record_id))
        self.batch_size = int(batch_size)
        self.max_atoms = max_atoms or max(len(row.atomic_numbers) for row in self.samples)
        if self.max_atoms < max(len(row.atomic_numbers) for row in self.samples):
            raise ValueError("max_atoms is smaller than a sample")
        self.shuffle = bool(shuffle)
        self.seed = int(seed)

    def _order(self) -> list[int]:
        order = list(range(len(self.samples)))
        if self.shuffle:
            np.random.default_rng(self.seed).shuffle(order)
        return order

    def _collate(self, rows: Sequence[DftDaSample]) -> PairedBatch:
        batch_size = len(rows)
        nmax = self.max_atoms
        atomic_numbers = torch.zeros((batch_size, nmax), dtype=torch.long)
        coordinates = torch.zeros((batch_size, nmax, 3), dtype=torch.float32)
        atom_mask = torch.zeros((batch_size, nmax), dtype=torch.bool)
        charge = torch.zeros(batch_size, dtype=torch.long)
        multiplicity = torch.zeros(batch_size, dtype=torch.long)
        event = torch.zeros((batch_size, nmax, nmax), dtype=torch.float32)
        event_mask = torch.zeros((batch_size, nmax, nmax), dtype=torch.bool)
        ts = torch.zeros((batch_size, nmax, 3), dtype=torch.float32)
        ts_mask = torch.zeros((batch_size, nmax), dtype=torch.bool)
        metadata: list[dict[str, str]] = []
        for index, sample in enumerate(rows):
            count = len(sample.atomic_numbers)
            atomic_numbers[index, :count] = torch.tensor(sample.atomic_numbers, dtype=torch.long)
            coordinates[index, :count] = torch.from_numpy(np.asarray(sample.reactant_coordinates, dtype=np.float32))
            ts[index, :count] = torch.from_numpy(np.asarray(sample.ts_coordinates, dtype=np.float32))
            atom_mask[index, :count] = True
            ts_mask[index, :count] = True
            charge[index] = int(sample.record.reactant["charge"])
            multiplicity[index] = int(sample.record.reactant["multiplicity"])
            event[index, :count, :count] = torch.from_numpy(_event_matrix(sample))
            event_mask[index, :count, :count] = True
            metadata.append({
                "sample_id": sample.record.record_id,
                "source_dataset": sample.record.source_dataset,
                "source_record_id": sample.record.source_record_id,
                "parent_group": sample.record.parent_reaction_id,
                "split": sample.record.admission.removeprefix("development_"),
            })
        return PairedBatch(
            reactant_input={
                "atomic_numbers": atomic_numbers,
                "reactant_coordinates": coordinates,
                "charge": charge,
                "multiplicity": multiplicity,
                "atom_mask": atom_mask,
            },
            training_supervision={
                "event_bond_edits": event,
                "event_mask": event_mask,
                "ts_coordinates": ts,
                "ts_mask": ts_mask,
            },
            metadata=tuple(metadata),
        )

    def __iter__(self) -> Iterator[PairedBatch]:
        order = self._order()
        for start in range(0, len(order), self.batch_size):
            yield self._collate([self.samples[index] for index in order[start : start + self.batch_size]])

    def first_batch(self) -> PairedBatch:
        return next(iter(self))


def load_paired_pilot(manifest_path: str | Path, cache_root: str | Path, *, batch_size: int = 8, shuffle: bool = False, seed: int = 0) -> PairedPilotLoader:
    """Reconstruct audited samples by source identity and build the loader."""

    records = load_track_b_jsonl(str(manifest_path))
    if not records:
        raise ValueError("paired pilot manifest is empty")
    samples, _ = load_dft_da_samples(cache_root)
    by_id = {sample.record.record_id: sample for sample in samples}
    selected: list[DftDaSample] = []
    for record in records:
        sample = by_id.get(record.record_id)
        if sample is None:
            raise ValueError(f"manifest record is not present in audited source: {record.record_id}")
        source_record = sample.record
        if source_record.source_asset_sha256 != record.source_asset_sha256:
            raise ValueError(f"source asset hash mismatch for {record.record_id}")
        if source_record.source_record_sha256 != record.source_record_sha256:
            raise ValueError(f"source record hash mismatch for {record.record_id}")
        for section, key in (("reactant", "coordinates_sha256"), ("product_label", "coordinates_sha256"), ("ts_geometry", "sha256")):
            expected_section = getattr(record, section) if section != "reactant" else record.reactant
            observed_section = getattr(source_record, section) if section != "reactant" else source_record.reactant
            expected = expected_section[key]
            observed = observed_section[key]
            if observed != expected:
                raise ValueError(f"{section} provenance hash mismatch for {record.record_id}")
        # The manifest is authoritative for admission and group split.  The
        # source parser reconstructs the same physical arrays, but its
        # provisional record defaults every row to development_train.
        selected.append(replace(sample, record=record))
    audit_track_b_leakage(records, strict_family_holdout=True)
    return PairedPilotLoader(selected, batch_size=batch_size, shuffle=shuffle, seed=seed)


def input_view_fingerprint(records: Iterable[TrackBRecord]) -> str:
    values = {record.record_id: reactant_input_fingerprint(record) for record in records}
    return canonical_hash(values)
