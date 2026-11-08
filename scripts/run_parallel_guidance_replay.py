#!/usr/bin/env python3
"""Run the deterministic #166 guidance ablation replay."""
from __future__ import annotations

import argparse
from pathlib import Path

from xtbflow.sampling import GenerationSnapshot, OfflineGuidanceReplay, ReplayBudget, write_replay_artifacts


def build_snapshot(seed: int) -> GenerationSnapshot:
    return GenerationSnapshot(
        parent_id="synthetic-parent-166",
        attempt_id="synthetic-attempt-166",
        generation_step=0,
        geometry=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        event=(0.0, 0.0, 0.0, 0.0),
        event_mask=(True, True, False, True),
        random_seed=seed,
        metadata={"temperature": 298.15, "source": "synthetic-reactant-only"},
        provenance={"provider_revision": "parallel-guidance-replay-v1"},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=166)
    args = parser.parse_args()
    replay = OfflineGuidanceReplay(
        build_snapshot(args.seed),
        candidate_pool_id="synthetic-candidate-pool-166-v1",
        budget=ReplayBudget(model_forwards=3, calculator_calls=0, wall_seconds=1.0),
    )
    artifacts = replay.run_all()
    paths = write_replay_artifacts(artifacts, args.output_dir)
    print(f"Wrote {len(paths)} deterministic replay artifacts to {Path(args.output_dir)}")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
