#!/usr/bin/env python3
"""Validate a reproducible handoff/freeze manifest without network access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from xtbflow.governance.handoff import load_manifest, validate_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="JSON handoff manifest")
    parser.add_argument("--output", type=Path, required=True, help="JSON audit output")
    parser.add_argument("--offline", action="store_true", help="do not treat a declared remote check as verified")
    parser.add_argument("--require-ready", action="store_true", help="return non-zero when any freeze blocker remains")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output if args.output.is_absolute() else ROOT / args.output
    try:
        payload = load_manifest(config)
        audit = validate_manifest(payload, root=ROOT, offline=args.offline)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not audit["contract_valid"]:
        print(f"FAIL: handoff contract invalid; see {output}", file=sys.stderr)
        return 1
    if args.require_ready and not audit["ready"]:
        print(f"BLOCKED: handoff is not ready; see {output}", file=sys.stderr)
        return 1
    state = "READY" if audit["ready"] else "VALID-BLOCKED"
    print(f"{state}: {audit.get('manifest_id')} (audit: {output})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
