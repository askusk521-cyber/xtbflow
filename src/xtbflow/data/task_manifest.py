"""Task projections of already-admitted records; never a new admission route."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator, Sequence

import torch

from .dft_da import DftDaSample
from .paired_loader import PairedPilotLoader, load_paired_pilot
from .records import canonical_hash
from .track_b import TrackBRecord, audit_track_b_leakage


SCHEMA = "xtbflow-task-manifest/v1"
TASK_TARGETS = {
    "event_only": ("event_bond_edits", "event_mask"),
    "geometry_only": ("ts_coordinates", "ts_mask"),
    "paired_joint": ("event_bond_edits", "event_mask", "ts_coordinates", "ts_mask"),
    "energy_force": (),
}
SPLITS = {"train": "development_train", "validation": "development_validation", "test": "development_test"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def build_task_manifest(records: Sequence[TrackBRecord], *, version: str,
                        source_manifest_sha256: str) -> dict:
    """Count existing permissions and label projections without reading targets."""
    if not version or len(source_manifest_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_manifest_sha256):
        raise ValueError("version and explicit source manifest SHA256 required")
    if len({r.record_id for r in records}) != len(records):
        raise ValueError("duplicate source record identity")
    # Reconstruct to recheck mutable nested values and all admission gates.
    records = [TrackBRecord.from_dict(r.to_dict()) for r in records]
    accepted = [r for r in records if r.admission in SPLITS.values()]
    audit = audit_track_b_leakage(accepted, strict_family_holdout=True)
    rows = []
    for record in sorted(accepted, key=lambda r: r.record_id):
        rows.append(dict(record_id=record.record_id, parent_id=record.parent_reaction_id,
                         family_id=record.family_id, split=record.admission.removeprefix("development_"),
                         record_sha256=canonical_hash(record.to_dict()), input_sha256=record.input_fingerprint(),
                         source_dataset=record.source_dataset, source_revision=record.source_revision,
                         source_record_sha256=record.source_record_sha256,
                         source_asset_sha256=record.source_asset_sha256,
                         claim_limit=record.claim_limit, geometry_evidence=record.ts_geometry.get("evidence"),
                         license_record=record.license_record,
                         coordinate_unit=record.reactant["coordinate_unit"],
                         charge=record.reactant["charge"], multiplicity=record.reactant["multiplicity"],
                         atom_mapping_sha256=canonical_hash(record.atoms["map_ids"]),
                         reference_protocol_id=record.reference_protocol["protocol_id"],
                         reference_protocol_sha256=record.reference_protocol["protocol_sha256"],
                         environment={"status": "unresolved_for_matched_geometry_pairing",
                                      "source_protocol_declaration": record.reference_protocol["environment"]},
                         tasks=["event_only", "geometry_only", "paired_joint"]))
    index = {task: {split: [r["record_id"] for r in rows if task in r["tasks"] and r["split"] == split]
                    for split in SPLITS} for task in TASK_TARGETS}
    counts = {}
    by_id = {r["record_id"]: r for r in rows}
    for task, splits in index.items():
        counts[task] = {}
        for split, identifiers in splits.items():
            counts[task][split] = dict(records=len(identifiers),
                parent_groups=len({by_id[i]["parent_id"] for i in identifiers}),
                family_groups=len({by_id[i]["family_id"] for i in identifiers}))
    excluded = Counter(r.admission for r in records if r.admission not in SPLITS.values())
    payload = dict(schema=SCHEMA, version=version, source_manifest_sha256=source_manifest_sha256,
                   records=rows, task_index=index, counts=counts, split_audit=audit,
                   excluded_admissions=dict(sorted(excluded.items())), confirmation_records=0,
                   unavailable_tasks={"energy_force": "This endpoint/TS source adapter supplies no matched-geometry energy/force labels."},
                   claim_limits=["Projections of an existing development source, not new data admission.",
                                 "The three event/geometry views overlap; their counts must not be added as independent examples.",
                                 "No confirmation set, cross-source overlap clearance or new reaction-family generalization claim.",
                                 "Single-target source admission, energy-force ingestion, declared partial conditions and feedback promotion remain separate work."])
    payload["content_sha256"] = canonical_hash(payload)
    return payload


def verify_task_manifest(manifest: dict, records: Sequence[TrackBRecord], source_hash: str) -> None:
    rebuilt = build_task_manifest(records, version=manifest["version"], source_manifest_sha256=source_hash)
    if rebuilt != manifest:
        raise ValueError("task manifest does not match reconstructed source records or permissions")


@dataclass(frozen=True)
class TaskBatch:
    task: str
    split: str
    purpose: str
    reactant_input: dict[str, torch.Tensor]
    targets: dict[str, torch.Tensor]
    metadata: tuple[dict[str, str], ...]

    def training_views(self) -> dict:
        if self.purpose != "training" or self.split != "train":
            raise ValueError("evaluation labels cannot be exposed as training supervision")
        return dict(reactant_input=self.reactant_input, training_supervision=self.targets, metadata=self.metadata)


class TaskLoader:
    """Explicit split and purpose. Targets are projected; reactants stay fixed."""

    def __init__(self, samples: Sequence[DftDaSample], *, task: str, split: str,
                 purpose: str, batch_size: int = 8, seed: int = 0):
        if task not in TASK_TARGETS or split not in SPLITS or purpose not in {"training", "evaluation"}:
            raise ValueError("unsupported task, split or purpose")
        if purpose == "training" and split != "train":
            raise ValueError("only the training split can supply training supervision")
        if task == "energy_force":
            raise ValueError("energy_force unavailable for the endpoint/TS source adapter")
        selected = [s for s in samples if s.record.admission == SPLITS[split]]
        if not selected:
            raise ValueError("no admitted records for the requested split")
        for sample in selected:
            TrackBRecord.from_dict(sample.record.to_dict())
        audit_track_b_leakage([s.record for s in samples if s.record.admission in SPLITS.values()], strict_family_holdout=True)
        self.task, self.split, self.purpose = task, split, purpose
        self.loader = PairedPilotLoader(selected, batch_size=batch_size, shuffle=purpose == "training", seed=seed)

    def __iter__(self) -> Iterator[TaskBatch]:
        for batch in self.loader:
            yield TaskBatch(self.task, self.split, self.purpose, batch.reactant_input,
                            {key: batch.training_supervision[key] for key in TASK_TARGETS[self.task]}, batch.metadata)


def load_task_views(index_path: Path, source_manifest_path: Path, cache_root: Path, *,
                    task: str, split: str, purpose: str, batch_size: int = 8, seed: int = 0) -> TaskLoader:
    source = load_paired_pilot(source_manifest_path, cache_root, batch_size=batch_size)
    manifest = json.loads(Path(index_path).read_text())
    verify_task_manifest(manifest, [s.record for s in source.samples], digest(Path(source_manifest_path)))
    return TaskLoader(source.samples, task=task, split=split, purpose=purpose, batch_size=batch_size, seed=seed)
