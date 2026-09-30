#!/usr/bin/env python3
"""Export a target-free reactant JSONL manifest from the audited DA cache.

The adapter may inspect product and TS files while auditing the public archive,
but the resulting manifest contains only the fields accepted by the generation
firewall.  Candidate generation can therefore run from this file without
opening the source labels again.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

from xtbflow.data.dft_da import load_dft_da_samples, with_splits
from xtbflow.data.records import canonical_hash
from xtbflow.evaluation.generation import reactant_view_from_sample


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    args = parser.parse_args()
    stage_start = time.monotonic()
    output = args.output.expanduser()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    state_path = output.with_name(f"{output.stem}.run.json")
    if state_path.exists():
        raise SystemExit(f"refusing to overwrite existing run state: {state_path}")
    state = {
        "schema": "xtbflow-dft-da-reactant-export/v1",
        "status": "prepared",
        "split": args.split,
        "output": output.name,
        "execution_host": platform.node(),
        "python": sys.version.split()[0],
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
    try:
        samples, audit = load_dft_da_samples(args.cache_root)
        selected = [
            sample
            for sample in with_splits(samples)
            if sample.record.admission == f"development_{args.split}"
        ]
        if not selected:
            raise ValueError(f"selected split is empty: {args.split}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for sample in selected:
                view = reactant_view_from_sample(sample)
                handle.write(json.dumps(view.to_mapping(), ensure_ascii=False, sort_keys=True) + "\n")
        temporary.replace(output)
        state.update(
            {
                "status": "completed",
                "record_count": len(selected),
                "parent_count": len({sample.record.parent_reaction_id for sample in selected}),
                "source_archive_sha256": audit.get("source_archive_sha256"),
                "source_csv_sha256": audit.get("source_csv_sha256"),
                "split_fingerprint": canonical_hash({sample.record.record_id: sample.record.admission for sample in selected}),
                "output_sha256": _sha256(output),
                "target_fields_written": False,
                "elapsed_seconds": time.monotonic() - stage_start,
            }
        )
        state_path.write_text(json.dumps(state, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": "completed",
                    "split": args.split,
                    "records": len(selected),
                    "parents": len({sample.record.parent_reaction_id for sample in selected}),
                    "source_archive_sha256": audit.get("source_archive_sha256"),
                    "source_csv_sha256": audit.get("source_csv_sha256"),
                    "output": str(output),
                    "output_sha256": _sha256(output),
                    "target_fields_written": False,
                },
                sort_keys=True,
            )
        )
    except Exception as exc:
        state.update({"status": "execution_failed", "error": f"{type(exc).__name__}: {exc}"})
        state_path.write_text(json.dumps(state, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
