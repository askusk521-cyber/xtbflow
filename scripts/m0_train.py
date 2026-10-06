"""Train one M0 role.
Example: python scripts/m0_train.py --config configs/m0/base.json --role joint --seed 0 \
         --out $XTBFLOW_RUNS/m0/full/joint_s0
"""
import argparse, copy, hashlib, json, math, os, random, subprocess, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from xtbflow.m0.batching import T1xDataset, collate
from xtbflow.m0.flow import flow_loss, training_inputs
from xtbflow.m0.model import ReactionFlowNet, count_parameters
from xtbflow.m0.t1x_data import T1xCache


def lr_at(step: int, tc: dict) -> float:
    if step < tc["warmup_steps"]:
        return tc["lr"] * (step + 1) / tc["warmup_steps"]
    p = (step - tc["warmup_steps"]) / max(1, tc["steps"] - tc["warmup_steps"])
    return tc["min_lr"] + 0.5 * (tc["lr"] - tc["min_lr"]) * (1.0 + math.cos(math.pi * p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--role", choices=("joint", "event", "geometry"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=None)       # 调参时用 20000
    ap.add_argument("--sigma-b", type=float, default=None)
    ap.add_argument("--sigma-x", type=float, default=None)
    ap.add_argument("--wide-cascade", action="store_true")   # 只用于可选的 A2x 对照
    ap.add_argument("--limit", type=int, default=None)       # 调试用
    a = ap.parse_args()

    cfg = json.loads(a.config.read_text())
    tc = dict(cfg["train"], steps=a.steps or cfg["train"]["steps"])
    sigma_b = cfg["flow"]["sigma_b"] if a.sigma_b is None else a.sigma_b
    sigma_x = cfg["flow"]["sigma_x"] if a.sigma_x is None else a.sigma_x
    model_cfg = cfg["model"] if (a.role == "joint" or a.wide_cascade) else cfg["cascade_model"]

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed_all(a.seed)

    model = ReactionFlowNet(a.role, **model_cfg).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])

    cache_path = Path(os.path.expandvars(cfg["data"]["cache"]))
    ds = T1xDataset(T1xCache(cache_path), split=0, limit=a.limit)
    loader_gen = torch.Generator().manual_seed(a.seed)     # 三个角色同种子 => 同样的批次顺序
    loader = DataLoader(ds, batch_size=tc["batch_size"], shuffle=True, drop_last=True,
                        collate_fn=collate, num_workers=tc["num_workers"], generator=loader_gen,
                        persistent_workers=tc["num_workers"] > 0)
    noise_gen = torch.Generator(device=device).manual_seed(a.seed + 1)

    a.out.mkdir(parents=True, exist_ok=True)
    eval_steps = set(range(tc["first_eval"], tc["steps"] + 1, tc["eval_every"])) | {tc["steps"]}
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = {"role": a.role, "seed": a.seed, "model_cfg": model_cfg, "sigma_b": sigma_b, "sigma_x": sigma_x,
            "train": tc, "params": count_parameters(model), "git_sha": git_sha,
            "config_sha256": hashlib.sha256(a.config.read_bytes()).hexdigest(),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
    (a.out / "run_meta.json").write_text(json.dumps(meta, indent=2))

    log = (a.out / "train_log.jsonl").open("w")
    step, t0 = 0, time.time()
    while step < tc["steps"]:
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            for group in opt.param_groups:
                group["lr"] = lr_at(step, tc)
            inputs, targets = training_inputs(a.role, batch, sigma_b, sigma_x, noise_gen)
            losses = flow_loss(model(**inputs), targets, batch["atom_mask"], cfg["flow"]["geometry_loss_weight"])
            if not torch.isfinite(losses["total"]):
                (a.out / "NONFINITE.json").write_text(json.dumps(
                    {"step": step, "index": batch["index"].tolist()}))
                raise SystemExit(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(p, 1.0 - tc["ema_decay"])
            step += 1
            if step % 100 == 0:
                row = {"step": step, "lr": lr_at(step, tc), "grad_norm": float(gnorm),
                       "elapsed_s": time.time() - t0}
                row.update({k: float(v) for k, v in losses.items()})
                log.write(json.dumps(row) + "\n"); log.flush()
            if step in eval_steps:
                torch.save({"ema": ema.state_dict(), "model": model.state_dict(), "step": step, **meta},
                           a.out / f"ckpt_{step}.pt")
            if step >= tc["steps"]:
                break
    log.close()


if __name__ == "__main__":
    main()
