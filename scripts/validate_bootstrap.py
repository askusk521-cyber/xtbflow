#!/usr/bin/env python3
"""Validate the bootstrap inventory; never launches jobs or accesses the network."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "bootstrap_inventory.json"

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def tracked_paths() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return sorted(p for p in result.stdout.decode("utf-8").split("\0")
                  if p and p != INVENTORY.name)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="Explicitly regenerate the inventory from tracked files after intended changes.")
    args = parser.parse_args()
    try:
        if args.refresh:
            paths = tracked_paths()
            if not paths:
                raise ValueError("No tracked files; stage intended sources before refreshing.")
            for relative in paths:
                target = ROOT / relative
                if not target.is_file() or target.is_symlink():
                    raise ValueError(f"Not a regular tracked file: {relative}")
            payload = {"schema": "xtbflow-bootstrap-inventory/v1",
                       "purpose": "File integrity only; no scientific validation implied.",
                       "files": [{"path": p, "sha256": digest(ROOT / p),
                                  "bytes": (ROOT / p).stat().st_size} for p in paths]}
            INVENTORY.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
            print(f"Refreshed {len(paths)} tracked files; commit the inventory with the changes.")
            return 0
        payload = json.loads(INVENTORY.read_text(encoding="utf-8"))
        rows = payload["files"]
        if not rows:
            raise ValueError("Empty inventory.")
        failures: list[str] = []
        seen: set[str] = set()
        for row in rows:
            relative = row["path"]
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or relative in seen:
                failures.append(f"Invalid or duplicate path: {relative}")
                continue
            seen.add(relative)
            target = ROOT / path
            if not target.is_file() or target.is_symlink():
                failures.append(f"Missing/nonregular file: {relative}")
            elif digest(target) != row["sha256"] or target.stat().st_size != row["bytes"]:
                failures.append(f"Changed file: {relative}")
        for path in ROOT.glob("schemas/*.json"):
            json.loads(path.read_text(encoding="utf-8"))
        if failures:
            for failure in failures:
                print(failure, file=sys.stderr)
            return 1
        print(f"PASS: {len(rows)} files match the bootstrap inventory.")
        print("This verifies files only; no training, quantum chemistry, remote upload or model validation is implied.")
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
