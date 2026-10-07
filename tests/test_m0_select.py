"""Synthetic checks of the M0 validation selection rules (no model or chemistry claim)."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("m0_select", ROOT / "scripts" / "m0_select.py")
m0_select = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m0_select)


def write_eval(path: Path, hits: list[float]) -> None:
    # Two parents with two queries each.
    path.parent.mkdir(parents=True)
    rows = [{"index": i, "parent_id": f"p{i // 2}", "hit": {"0.5": h}} for i, h in enumerate(hits)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_noise_selection_metric_and_tie_rule(tmp_path):
    for arm in m0_select.ARMS:
        for sb in m0_select.SIGMAS:
            for sx in m0_select.SIGMAS:
                hits = [1.0, 0.0, 0.0, 0.0]  # parent means 0.5, 0.0 -> M = 0.25 everywhere
                if (arm, sb, sx) == ("cascade", 1.0, 0.5):
                    hits = [1.0, 1.0, 0.0, 0.5]  # M = (1.0 + 0.25) / 2
                write_eval(tmp_path / f"val_{arm}_b{sb}_x{sx}" / "eval.jsonl", hits)
    result = m0_select.select_noise(tmp_path, n_queries=4, n_parents=2)
    joint, cascade = result["selected"]["joint"], result["selected"]["cascade"]
    assert (joint["sigma_b"], joint["sigma_x"], joint["M"]) == (0.5, 0.5, 0.25)  # tie -> lowest σ
    assert (cascade["sigma_b"], cascade["sigma_x"], cascade["M"]) == (1.0, 0.5, 0.625)


def test_checkpoint_selection_pairs_cascade_roles_at_one_step(tmp_path):
    val, runs = tmp_path / "val", tmp_path / "runs"
    for arm in m0_select.ARMS:
        for seed in m0_select.SEEDS:
            for step in m0_select.STEPS:
                best = step == 40000 and arm == "cascade"
                write_eval(val / f"val_{arm}_s{seed}_step{step}" / "eval.jsonl",
                           [1.0, 1.0, 1.0, 1.0] if best else [1.0, 0.0, 1.0, 0.0])
    for role in ("joint", "event", "geometry"):
        for seed in m0_select.SEEDS:
            for step in m0_select.STEPS:
                p = runs / f"{role}_s{seed}" / f"ckpt_{step}.pt"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(f"{role}{seed}{step}".encode())
    noise = {"selected": {a: {"sigma_b": 0.5, "sigma_x": 1.0} for a in m0_select.ARMS}}
    result = m0_select.select_checkpoints(val, runs, noise, n_queries=4, n_parents=2)
    assert result["selected"]["joint_s1"]["step"] == 20000  # all tied -> earliest step
    cascade = result["selected"]["cascade_s2"]
    assert cascade["step"] == 40000 and set(cascade["checkpoints"]) == {"event", "geometry"}
    assert all(c["path"].endswith("ckpt_40000.pt") for c in cascade["checkpoints"].values())
    assert cascade["checkpoints"]["event"]["sha256"] == m0_select.sha256(runs / "event_s2" / "ckpt_40000.pt")
