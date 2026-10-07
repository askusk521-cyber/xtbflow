"""Validation-only selections for M0 (guide §6.3 and §7.1). Never reads test outputs.

noise:       <root>/val_{arm}_b{σ_b}_x{σ_x}/eval.jsonl, 4 per arm (seed 0, 20k steps)
             -> noise_selection.json, the σ pair with the highest validation M per arm.
checkpoint:  <root>/val_{arm}_s{seed}_step{step}/eval.jsonl for steps 20k..60k
             -> frozen_selection.json, the best step per arm and seed, with checkpoint
             SHA-256 (the A arm pairs event and geometry at the same step) and the git commit.

M is the guide's primary metric: mean over reactant parents of the per-parent mean hit
rate at δ = 0.5 Å. Ties go to the lower σ_b, then lower σ_x, and to the earlier step;
both rules are fixed here before any result is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from xtbflow.m0.metrics import parent_means

ARMS = ("joint", "cascade")
SIGMAS = (0.5, 1.0)
STEPS = (20000, 30000, 40000, 50000, 60000)
SEEDS = (0, 1, 2)


def validation_metric(path: Path, n_queries: int, n_parents: int) -> dict:
    rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != n_queries or len({r["index"] for r in rows}) != n_queries:
        raise ValueError(f"expected {n_queries} distinct validation queries in {path}")
    per_parent = parent_means({r["index"]: float(r["hit"]["0.5"]) for r in rows},
                              {r["index"]: r["parent_id"] for r in rows})
    if len(per_parent) != n_parents:
        raise ValueError(f"expected {n_parents} validation parents in {path}")
    return {"M": float(np.mean(list(per_parent.values()))), "n_queries": len(rows),
            "n_parents": len(per_parent), "eval_sha256": sha256(path)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_noise(root: Path, n_queries: int, n_parents: int) -> dict:
    rows = []
    for arm in ARMS:
        for sb in SIGMAS:
            for sx in SIGMAS:
                m = validation_metric(root / f"val_{arm}_b{sb}_x{sx}" / "eval.jsonl", n_queries, n_parents)
                rows.append(dict(arm=arm, sigma_b=sb, sigma_x=sx, **m))
    selected = {arm: min((r for r in rows if r["arm"] == arm),
                         key=lambda r: (-r["M"], r["sigma_b"], r["sigma_x"])) for arm in ARMS}
    return dict(stage="validation_noise_selection", seed=0, steps=20000, n_candidates=len(rows),
                tie_rule="lower sigma_b, then lower sigma_x", selected=selected, all_runs=rows)


def select_checkpoints(root: Path, runs: Path, noise: dict, n_queries: int, n_parents: int) -> dict:
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    chosen, all_rows = {}, []
    for arm in ARMS:
        for seed in SEEDS:
            rows = []
            for step in STEPS:
                m = validation_metric(root / f"val_{arm}_s{seed}_step{step}" / "eval.jsonl", n_queries, n_parents)
                rows.append(dict(arm=arm, seed=seed, step=step, **m))
            all_rows += rows
            best = min(rows, key=lambda r: (-r["M"], r["step"]))
            roles = ("joint",) if arm == "joint" else ("event", "geometry")
            ckpts = {role: runs / f"{role}_s{seed}" / f"ckpt_{best['step']}.pt" for role in roles}
            chosen[f"{arm}_s{seed}"] = dict(
                arm=arm, seed=seed, step=best["step"], val_M=best["M"],
                sigma_b=noise["selected"][arm]["sigma_b"], sigma_x=noise["selected"][arm]["sigma_x"],
                checkpoints={role: {"path": str(p), "sha256": sha256(p)} for role, p in ckpts.items()})
    return dict(stage="frozen_checkpoint_selection", git_sha=git_sha, steps=list(STEPS),
                tie_rule="earlier step", noise_selection=noise["selected"], selected=chosen,
                all_runs=all_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("noise", "checkpoint"))
    ap.add_argument("--val-root", type=Path, required=True)
    ap.add_argument("--runs", type=Path, help="formal run root (checkpoint stage)")
    ap.add_argument("--noise-selection", type=Path, help="noise_selection.json (checkpoint stage)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-queries", type=int, default=220)
    ap.add_argument("--n-parents", type=int, default=33)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    if a.stage == "noise":
        result = select_noise(a.val_root, a.n_queries, a.n_parents)
    else:
        if a.runs is None or a.noise_selection is None:
            ap.error("checkpoint stage needs --runs and --noise-selection")
        noise = json.loads(a.noise_selection.read_text(encoding="utf-8"))
        result = select_checkpoints(a.val_root, a.runs, noise, a.n_queries, a.n_parents)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["selected"], indent=2))


if __name__ == "__main__":
    main()
