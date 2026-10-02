#!/usr/bin/env python3
"""Build the real, non-quarantine paired pilot from an audited public source."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from xtbflow.data.dft_da import load_dft_da_samples, with_splits, write_manifest
from xtbflow.data.paired_loader import input_view_fingerprint
from xtbflow.data.track_b import audit_track_b_leakage


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/manifests/paired_pilot_v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/data/paired_pilot_v1.json"))
    args = parser.parse_args()
    samples, source_audit = load_dft_da_samples(args.cache_root)
    samples = with_splits(samples)
    if len(samples) < 32:
        raise RuntimeError(f"real admitted paired records below required minimum: {len(samples)}")
    records = [sample.record for sample in samples]
    leakage = audit_track_b_leakage(records, strict_family_holdout=True)
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    write_manifest(manifest, samples)
    split_index: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        rows = [sample for sample in samples if sample.record.admission.removeprefix("development_") == split]
        split_index[split] = {
            "record_ids": [sample.record.record_id for sample in rows],
            "parent_groups": sorted({sample.record.parent_reaction_id for sample in rows}),
            "family_groups": sorted({sample.record.family_id for sample in rows}),
            "count": len(rows),
        }
    _write_json(out / "split_index.json", split_index)
    _write_json(out / "input_view_fingerprint.json", {
        "schema": "xtbflow-paired-pilot-input-view/v1",
        "aggregate_sha256": input_view_fingerprint(records),
        "records": {record.record_id: record.input_fingerprint() for record in records},
        "firewall": {
            "reactant_input_fields": ["atomic_numbers", "map_ids", "reactant_graph_sha256", "reactant_coordinates_sha256", "coordinate_unit", "charge", "multiplicity", "microstate_id"],
            "excluded_fields": ["product_label", "event_label", "ts_geometry", "reference_protocol"],
        },
    })
    blockers = source_audit.get("rejection_counts", {})
    funnel = [
        {"stage": "source_records", "count": source_audit["source_row_count"], "semantics": "CSV rows in pinned archive"},
        {"stage": "source_asset_hash_valid", "count": source_audit["source_row_count"], "semantics": "archive hash matches recorded asset"},
        {"stage": "electronic_state_log_candidates", "count": source_audit["state_log_candidates"], "semantics": "both monomer logs exist; state parser is stricter"},
        {"stage": "graph_and_state_admitted", "count": source_audit["graph_and_state_admitted_count"], "semantics": "mapped graph, XYZ order/connectivity, event edits, state and CHNO gate"},
        {"stage": "unified_track_b_admitted", "count": len(records), "semantics": "non-quarantine TrackBRecord"},
    ]
    _write_json(out / "admission_report.json", {
        "schema": "xtbflow-paired-pilot-admission/v1",
        "source_dataset": "diels-alder-reaction-space",
        "source_revision": source_audit["source_revision"],
        "source_asset_sha256": source_audit["source_archive_sha256"],
        "funnel": funnel,
        "blocker_counts": blockers,
        "admitted_record_count": len(records),
        "admitted_parent_count": len({record.parent_reaction_id for record in records}),
        "admitted_family_count": len({record.family_id for record in records}),
        "split_counts": dict(Counter(record.admission for record in records)),
        "leakage_audit": leakage,
        "manifest_sha256": _sha256(manifest),
        "source_audit": source_audit,
    })
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config.is_file() else {}
    config.update({"manifest": str(manifest), "source_cache": "external-cache:dft-da-full-29118509-v1", "record_count": len(records), "minimum_real_records": 32})
    _write_json(args.config, config)
    print(json.dumps({"status": "built", "records": len(records), "parents": len({record.parent_reaction_id for record in records}), "manifest": str(manifest)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
