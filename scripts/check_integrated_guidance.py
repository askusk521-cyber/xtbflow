#!/usr/bin/env python3
"""Bounded fake/cached integrated guidance entry; no scientific calculations."""
import argparse
import json
from pathlib import Path
from xtbflow.sampling.guidance_fixture import run_fixture


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fake", "cached"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run_fixture(args.out, args.mode, Path(__file__).resolve().parents[1]/"configs/research/integrated_entry_20261111.json")
    print(json.dumps({"mode": args.mode, "new_calls": result["result"]["new_calculator_calls"],
                      "cache_hit": result["result"]["cache_hit"]}))
