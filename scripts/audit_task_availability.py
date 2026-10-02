#!/usr/bin/env python3
"""Audit task-level record availability from a JSONL public manifest."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from xtbflow.data import audit_task_availability, load_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSONL manifest of PublicRecord rows")
    parser.add_argument("--output", type=Path, required=True, help="JSON audit output")
    args = parser.parse_args()
    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    output = args.output if args.output.is_absolute() else ROOT / args.output
    rows = load_jsonl(str(manifest))
    audit = audit_task_availability(rows)
    audit.write_json(output)
    print(
        f"audited {audit.record_count} records, {audit.identified_parent_reaction_count} identified parents; "
        f"fingerprint={audit.fingerprint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
