#!/usr/bin/env python3
"""Isolated worker: load a pinned research checkout and reactant-only queries."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
import types


def bind_parent_lifetime() -> None:
    """Linux workers cannot outlive their runner after a crash or timeout."""
    import ctypes
    import signal
    parent = os.getppid()
    expected = int(os.environ.get("XTBFlow_PARENT_PID", parent))
    if parent != expected or parent == 1:
        raise RuntimeError("generation_parent_already_exited")
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG failed")
    if os.getppid() != expected:
        raise RuntimeError("generation_parent_exited_during_setup")


def write(path: Path, value: dict) -> None:
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".write-")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class ForwardMeter:
    def __init__(self, path: Path, cap: int):
        if type(cap) is not int or cap < 1:
            raise ValueError("positive finite forward budget required")
        self.path, self.cap = path, cap
        self.rows = []
        self.consumed = 0
        self.context = {}
        self.started = 0.
        self.flush()

    def flush(self):
        write(self.path, dict(calls=self.rows, consumed_forward_evaluations=self.consumed))

    def before(self, role, module, args, kwargs):
        systems = int(kwargs["z"].shape[0])
        if self.consumed + systems > self.cap:
            raise RuntimeError("generation_forward_budget_exhausted")
        self.consumed += systems
        self.rows.append(dict(sequence=len(self.rows) + 1, role=role, systems=systems,
                              status="started", **self.context))
        self.flush()  # The forward cannot start until this durable write returns.
        self.started = time.monotonic()

    def after(self, module, args, output):
        import torch
        torch.cuda.synchronize()
        self.rows[-1].update(status="completed", wall_seconds=time.monotonic() - self.started)
        self.flush()


def main() -> int:
    bind_parent_lifetime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-source", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    packet = json.loads(args.packet.read_text())
    cfg, queries = packet["config"], packet["queries"]
    available = sorted(os.sched_getaffinity(0))
    if cfg["generation"]["cpu_cores"] > len(available):
        raise ValueError("insufficient allocated CPU affinity")
    os.sched_setaffinity(0, available[:cfg["generation"]["cpu_cores"]])
    # All project imports in this interpreter come from the pinned research tree.
    sys.path.insert(0, str(args.research_source.resolve() / "src"))
    import numpy as np
    import torch
    # The archived trunk imports two vocabulary constants from a training
    # batching module that also imports the target cache. Supply only those
    # exact frozen constants in this inference interpreter; no dataset loader.
    vocabulary = types.ModuleType("xtbflow.m0.batching")
    vocabulary.ELEMENT_INDEX = {1: 1, 6: 2, 7: 3, 8: 4}
    vocabulary.N_ELEMENT_TYPES = 5
    sys.modules[vocabulary.__name__] = vocabulary
    from xtbflow.m0.model import ReactionFlowNet
    from xtbflow.m0.sampler import decode_be
    from xtbflow.v1.interfaces import State, assert_query_fields, query_from_parents
    from xtbflow.v1.sampler import initial_state, rollout
    if not torch.cuda.is_available():
        raise RuntimeError("allocated GPU required")
    torch.set_num_threads(cfg["generation"]["cpu_cores"])
    torch.backends.cuda.matmul.allow_tf32 = False
    resource.setrlimit(resource.RLIMIT_FSIZE, (cfg["generation"]["max_output_bytes"],) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (4800, 4801))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    for query in queries:
        assert_query_fields(query)
        if set(query) != {"query_id", "atomic_numbers", "x_r", "b_r", "charge", "multiplicity"}:
            raise ValueError("query whitelist mismatch")
    meter = ForwardMeter(args.out / "calls.json", cfg["generation"]["max_forward_evaluations"])
    models, metadata = {}, {}
    for role, expected in cfg["checkpoints"].items():
        path = args.model_root / "generators" / f"{role}_s0" / "ckpt_60000.pt"
        if path.stat().st_size != expected["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError("checkpoint_identity_mismatch")
        status_path = path.parent / "status.json"
        if hashlib.sha256(status_path.read_bytes()).hexdigest() != expected["training_status_sha256"]:
            raise ValueError("training_status_identity_mismatch")
        status = json.loads(status_path.read_text())
        if status.get("status") != "complete" or status.get("steps") != cfg["checkpoint_step"]:
            raise ValueError("checkpoint_training_incomplete")
        obj = torch.load(path, map_location="cuda", weights_only=True)
        meta = obj["meta"]
        if meta["role"] != role or meta["seed"] != cfg["checkpoint_seed"] or obj["step"] != cfg["checkpoint_step"]:
            raise ValueError("checkpoint_metadata_mismatch")
        model = ReactionFlowNet(role, **meta["model_cfg"], dual_time=meta["dual_time"]).cuda()
        model.load_state_dict(obj["ema"])
        model.eval().requires_grad_(False)
        model.register_forward_pre_hook(lambda m, a, k, r=role: meter.before(r, m, a, k), with_kwargs=True)
        model.register_forward_hook(meter.after)
        models[role] = model
        metadata[role] = {key: meta[key] for key in ("role", "seed", "model_cfg", "dual_time", "sigma_b", "sigma_x", "split_hash")}
        metadata[role]["checkpoint_sha256"] = expected["sha256"]
        metadata[role]["training_status_sha256"] = expected["training_status_sha256"]
    if len({m["split_hash"] for m in metadata.values()}) != 1:
        raise ValueError("generator_training_split_mismatch")
    for key in ("sigma_b", "sigma_x"):
        if len({m[key] for m in metadata.values()}) != 1:
            raise ValueError("incompatible_generator_noise_scales")
    outputs = []
    report = dict(status="executing", models=metadata, candidates=outputs,
                  compatibility_adapter="inference-only frozen CHNO batching vocabulary; dataset classes unavailable",
                  environment=dict(torch=torch.__version__, cuda=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name(0), numpy=np.__version__))
    write(args.out / "generation.json", report)
    try:
        for query in queries:
            q = query_from_parents([query], "cuda")
            for method in cfg["methods"]:
                for proposal in range(cfg["proposals_per_query"]):
                    meter.context = dict(query_id=query["query_id"], method=method, proposal=proposal)
                    before = meter.consumed
                    started = time.monotonic()
                    state = initial_state(q, [proposal], cfg["checkpoint_seed"], sampling_seed=cfg["sampling_seed"],
                                          namespace=cfg["noise_namespace"], group="common",
                                          sigma_b=metadata["joint"]["sigma_b"], sigma_x=metadata["joint"]["sigma_x"])
                    initial_x = state.x.clone()
                    status, b_dec, final_x = "generated", None, None
                    if method == "joint_generation":
                        result = rollout(models["joint"], q, state, path=cfg["joint_clock"], n_steps=cfg["steps_per_branch"])
                    elif method == "independent_event_then_geometry":
                        result = rollout(models["event"], q, state, role="event", n_steps=cfg["steps_per_branch"], dual_time=False)
                        b_dec, _ = decode_be(result.state.b[0], q.element_index[0], len(query["atomic_numbers"]))
                        if b_dec is None or bool(result.failed[0]):
                            status = "event_decode_failed"
                        else:
                            b = torch.tensor(b_dec, device="cuda", dtype=state.b.dtype)[None]
                            conditioned = State(b, initial_x, torch.ones_like(state.t_b), torch.zeros_like(state.t_x))
                            result = rollout(models["geometry"], q, conditioned, role="geometry", n_steps=cfg["steps_per_branch"], dual_time=False)
                    elif method == "conditioned_geometry":
                        b_dec = np.asarray(packet["conditioned_events"][query["query_id"]], dtype=np.int64)
                        b = torch.tensor(b_dec, device="cuda", dtype=state.b.dtype)[None]
                        conditioned = State(b, initial_x, torch.ones_like(state.t_b), torch.zeros_like(state.t_x))
                        result = rollout(models["geometry"], q, conditioned, role="geometry", n_steps=cfg["steps_per_branch"], dual_time=False)
                        if not torch.equal(result.state.b, b):
                            raise ValueError("geometry_resampling_changed_fixed_event")
                    else:
                        raise ValueError("unregistered generation method")
                    if status == "generated":
                        if method != "conditioned_geometry":
                            b_dec, _ = decode_be(result.state.b[0], q.element_index[0], len(query["atomic_numbers"]))
                        final_x = result.state.x[0].cpu().tolist()
                        if bool(result.failed[0]) or b_dec is None:
                            status = "numerical_failure" if bool(result.failed[0]) else "decode_failed"
                    suffix = f":{cfg['candidate_namespace']}" if cfg.get("candidate_namespace") else ""
                    row = dict(candidate_id=f"{query['query_id']}:{method}:p{proposal}{suffix}", query_id=query["query_id"],
                               method=method, proposal=proposal, status=status,
                               b_dec=None if b_dec is None else b_dec.tolist(), x=final_x,
                               forward_evaluations=meter.consumed-before, wall_seconds=time.monotonic()-started)
                    outputs.append(row)
                    write(args.out / "generation.json", report)
        report["status"] = "completed"
    except Exception as exc:
        if meter.context:
            outputs.append(dict(candidate_id=f"{query['query_id']}:{method}:p{proposal}",
                **meter.context, status="execution_failed", b_dec=None, x=None,
                forward_evaluations=meter.consumed-before, wall_seconds=time.monotonic()-started,
                failure_type=type(exc).__name__))
        report.update(status="execution_failed", failure_type=type(exc).__name__, failure=str(exc))
    report["forward_evaluations"] = meter.consumed
    # Importing a catalogue or target cache is outside this worker's contract.
    report["forbidden_imports"] = sorted(k for k in sys.modules if k in {"xtbflow.v1.data", "xtbflow.m0.t1x_data"})
    if report["forbidden_imports"]:
        report["status"] = "input_isolation_failed"
    write(args.out / "generation.json", report)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
