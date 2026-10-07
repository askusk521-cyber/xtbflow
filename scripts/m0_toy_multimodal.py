"""Two-channel toy: both arms must cover both channels with matching geometry."""
import numpy as np
import torch

from xtbflow.m0.flow import flow_loss, training_inputs
from xtbflow.m0.model import ReactionFlowNet
from xtbflow.m0.sampler import decode_be, sample_cascade, sample_joint
from xtbflow.m0.t1x_data import kabsch_align

X_R = np.array([[0, 0, 0], [0.74, 0, 0], [0, 2.0, 0], [0.74, 2.0, 0]], dtype=float)
TS = {1: np.array([[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [1.0, 1.0, 0]], dtype=float),   # 形成 0-2、1-3
      2: np.array([[0, 0, 0], [1.0, 0, 0], [1.0, 1.0, 0], [0, 1.0, 0]], dtype=float)}   # 形成 0-3、1-2


def be(pairs):
    b = np.zeros((4, 4))
    for i, j in pairs:
        b[i, j] = b[j, i] = 1
    return b


B_R = be([(0, 1), (2, 3)])
B_P = {1: be([(0, 2), (1, 3)]), 2: be([(0, 3), (1, 2)])}
XR_C = X_R - X_R.mean(0)
TS_AL = {c: kabsch_align(TS[c], XR_C) for c in TS}
SMALL = dict(scalar_dim=32, vector_dim=8, edge_dim=16, n_layers=2)


def batch_of(channels):
    k = len(channels)
    t = lambda a: torch.tensor(np.stack(a), dtype=torch.float32)
    return {"z": torch.ones(k, 4, dtype=torch.long), "atom_mask": torch.ones(k, 4, dtype=torch.bool),
            "x_r": t([XR_C] * k), "b_r": t([B_R] * k),
            "x_ts": t([TS_AL[c] for c in channels]), "b_p": t([B_P[c] for c in channels]),
            "index": torch.arange(k)}


def train(role, steps=4000, sigma=0.5, seed=0):
    torch.manual_seed(seed)
    net = ReactionFlowNet(role, **SMALL)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed + 1)
    for _ in range(steps):
        channels = torch.randint(1, 3, (64,), generator=gen).tolist()
        batch = batch_of(channels)
        inputs, targets = training_inputs(role, batch, sigma, sigma, gen)
        loss = flow_loss(net(**inputs), targets, batch["atom_mask"])["total"]
        opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


def rmsd(a, b):
    return float(np.sqrt(((kabsch_align(a, b) - b) ** 2).sum(1).mean()))


def judge(result, name):
    x = result["x"].numpy()
    counts, own_ok, own_rmsd = {1: 0, 2: 0, "other": 0}, 0, []
    for k in range(x.shape[0]):
        mat, _ = decode_be(result["b_raw"][k], result["rv"]["z"][k], 4)
        offd = None if mat is None else (mat - np.diag(np.diag(mat)))
        ch = next((c for c in (1, 2) if offd is not None and np.array_equal(offd, B_P[c])), "other")
        counts[ch] += 1
        if ch != "other":
            r_own, r_other = rmsd(x[k], TS_AL[ch]), rmsd(x[k], TS_AL[3 - ch])
            own_rmsd.append(r_own)
            own_ok += int(r_own < r_other)
    s = x.shape[0]
    matched = counts[1] + counts[2]
    report = {"arm": name, "share_1": counts[1] / s, "share_2": counts[2] / s, "valid": matched / s,
              "median_rmsd_own": float(np.median(own_rmsd)) if own_rmsd else None,
              "closer_to_own": own_ok / max(1, matched)}
    print(report)
    assert report["valid"] >= 0.8, report
    assert 0.25 <= report["share_1"] <= 0.75 and 0.25 <= report["share_2"] <= 0.75, report
    assert report["median_rmsd_own"] <= 0.3 and report["closer_to_own"] >= 0.8, report


if __name__ == "__main__":
    rv = {k: v for k, v in batch_of([1]).items() if k not in ("x_ts", "b_p")}
    joint = train("joint")
    judge(sample_joint(joint, rv, 128, 50, 0.5, 0.5, torch.Generator().manual_seed(7)), "B_joint")
    event, geom = train("event"), train("geometry")
    judge(sample_cascade(event, geom, rv, 128, 50, 0.5, 0.5, torch.Generator().manual_seed(7)), "A_cascade")
    print("TOY PASS")
