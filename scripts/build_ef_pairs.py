#!/usr/bin/env python3
"""Build append-only same-geometry E/F pairs from observed calculator records.

The input is JSONL produced by an actual calculator runner. This command does
not invoke xTB, CP2K, or a reference program and therefore cannot fabricate
missing labels. Rows with calculator failures are retained as failure pairs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xtbflow.calculators.base import CalculationResult, MolecularSystem  # noqa: E402
from xtbflow.data.cache import append_pair_cache  # noqa: E402
from xtbflow.data.pairing import build_ef_pair, failure_pair  # noqa: E402


def _system(payload: dict) -> MolecularSystem:
    return MolecularSystem(tuple(payload["symbols"]), tuple(tuple(float(value) for value in row) for row in payload["coordinates"]), int(payload["charge"]), int(payload["multiplicity"]), dict(payload.get("environment", {})), str(payload.get("system_id", "")))


def _result(payload: dict, system: MolecularSystem) -> CalculationResult:
    value = dict(payload)
    value.setdefault("input_hash", system.input_hash)
    value.setdefault("charge", system.charge)
    value.setdefault("multiplicity", system.multiplicity)
    value.setdefault("operation", "energy_forces")
    if value.get("forces") is not None:
        value["forces"] = tuple(tuple(float(component) for component in row) for row in value["forces"])
    return CalculationResult(**value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSONL observed calculator rows")
    parser.add_argument("--output", required=True, type=Path, help="append-only E/F pair JSONL cache")
    args = parser.parse_args()
    pairs = []
    with args.input.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                system = _system(row["system"])
                source_id = str(row["source_record_id"])
                se_protocol = str(row.get("semi_empirical_protocol_id", row["semi_empirical"].get("protocol_id", "unknown")))
                ref_protocol = str(row.get("reference_protocol_id", row["reference"].get("protocol_id", "unknown")))
                if row.get("error"):
                    failure = row["error"]
                    pairs.append(failure_pair(system, source_record_id=source_id, semi_empirical_protocol_id=se_protocol, reference_protocol_id=ref_protocol, category=str(failure.get("category", "execution")), message=str(failure.get("message", "calculator failure"))))
                else:
                    pairs.append(build_ef_pair(system, _result(row["semi_empirical"], system), _result(row["reference"], system), source_record_id=source_id, semi_empirical_protocol_id=se_protocol, reference_protocol_id=ref_protocol))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise SystemExit(f"invalid observed row at line {line_number}: {exc}") from exc
    written, hits = append_pair_cache(args.output, pairs)
    print(json.dumps({"rows": len(pairs), "written": written, "cache_hits": hits, "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

