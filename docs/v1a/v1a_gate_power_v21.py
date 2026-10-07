"""xtbflow V1a v2.1 闸门功效工作模型（不读取任何实验数据）。

母体摘要按分子式聚类生成：
    d_p = mu + u_g + e_p,  Var(u_g) = ICC*SD^2,  Var(e_p) = (1-ICC)*SD^2
主比较为 B1-A2 的归一化 log-budget AUC 差；机制为两个对比
    M_0 = E[U(F_S) - U(F_0)],  M_R = E[U(F_S) - U(F_R)]，
二者在母体与分子式两个层面按相关系数 rho 相关（共享 U(F_S)）。

v2.1 闸门（与 V1a 第 10.2 节一致）：
    1. U(A) < 0.05                      -> NO_GO_RESOURCE_SMALL
    2. U(M_0) < 0.02 或 U(M_R) < 0.02    -> NO_GO_MECHANISM_SMALL
    3. L(A) > 0 且 L(M_0) > 0 且 L(M_R) > 0 -> GO_V1b
    4. 其余                              -> HOLD_INCONCLUSIVE
L/U 为分子式聚类 sandwich SE 加 t(G-1) 的单侧 95% 界限。

另附 v2.0 单一机制量 M（逐对取 max）的偏差模拟，说明为什么删除它。
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.stats import t as student_t


def cluster_bounds(values: np.ndarray, groups_per: int):
    """values: [reps, P]，按每组 groups_per 个母体连续排列。返回单侧 95% 下/上界。"""
    reps, p = values.shape
    g = p // groups_per
    mean = values.mean(axis=1, keepdims=True)
    scores = (values - mean).reshape(reps, g, groups_per).sum(axis=2)
    se = np.sqrt(g / (g - 1) * np.square(scores).sum(axis=1)) / p
    q = student_t.ppf(0.95, g - 1)
    m = mean[:, 0]
    return m - q * se, m + q * se


def clustered_normal(rng, reps, p, groups_per, mu, sd, icc, shared=None):
    g = p // groups_per
    if shared is None:
        zg = rng.standard_normal((reps, g))
        ze = rng.standard_normal((reps, p))
    else:
        zg, ze = shared
    u = np.repeat(zg, groups_per, axis=1) * np.sqrt(icc) * sd
    e = ze * np.sqrt(1 - icc) * sd
    return mu + u + e


def correlated_pair(rng, reps, p, groups_per, rho):
    g = p // groups_per
    zg1, zg2 = rng.standard_normal((2, reps, g))
    ze1, ze2 = rng.standard_normal((2, reps, p))
    zg2 = rho * zg1 + np.sqrt(1 - rho**2) * zg2
    ze2 = rho * ze1 + np.sqrt(1 - rho**2) * ze2
    return (zg1, ze1), (zg2, ze2)


def gate_v20(rng, reps, p, gp, auc, m, sd_a, sd_m, icc):
    a = clustered_normal(rng, reps, p, gp, auc, sd_a, icc)
    mm = clustered_normal(rng, reps, p, gp, m, sd_m, icc)
    la, ua = cluster_bounds(a, gp)
    lm, um = cluster_bounds(mm, gp)
    ng_r = ua < 0.05
    ng_m = ~ng_r & (um < 0.02)
    go = ~ng_r & ~ng_m & (la > 0) & (lm > 0)
    return go.mean(), (ng_r | ng_m).mean(), 1 - go.mean() - (ng_r | ng_m).mean()


def gate_v21(rng, reps, p, gp, auc, m0, mr, sd_a, sd_m, icc, rho):
    a = clustered_normal(rng, reps, p, gp, auc, sd_a, icc)
    s0, sr = correlated_pair(rng, reps, p, gp, rho)
    x0 = clustered_normal(rng, reps, p, gp, m0, sd_m, icc, shared=s0)
    xr = clustered_normal(rng, reps, p, gp, mr, sd_m, icc, shared=sr)
    la, ua = cluster_bounds(a, gp)
    l0, u0 = cluster_bounds(x0, gp)
    lr, ur = cluster_bounds(xr, gp)
    ng_r = ua < 0.05
    ng_m = ~ng_r & ((u0 < 0.02) | (ur < 0.02))
    go = ~ng_r & ~ng_m & (la > 0) & (l0 > 0) & (lr > 0)
    hold = ~(ng_r | ng_m | go)
    return {"GO": go.mean(), "NO_GO_RESOURCE": ng_r.mean(),
            "NO_GO_MECHANISM": ng_m.mean(), "HOLD": hold.mean()}


def max_metric_bias(rng, n=2_000_000):
    """v2.0 的 m = U(F_S) - max(U(F_0), U(F_R)) 在已知方向优势下的期望。"""
    out = []

    def landed(size, p_known):
        known = rng.random(size) < p_known
        return np.where(known, rng.random(size), 0.0)

    for q in (0.1, 0.2, 0.4):
        for p_s in (0.5, 0.6, 0.7):
            u0 = landed(n, 0.5)
            us = np.where(rng.random(n) < q, landed(n, p_s), u0)
            ur = np.where(rng.random(n) < q, landed(n, 0.5), u0)
            out.append({"q": q, "p_known_random": 0.5, "p_known_score": p_s,
                        "E_US_minus_U0": float((us - u0).mean()),
                        "E_US_minus_UR": float((us - ur).mean()),
                        "E_m_v20": float((us - np.maximum(u0, ur)).mean())})
    return out


def main(reps=100_000, seed=20261007):
    rng = np.random.default_rng(seed)
    gp, icc, sd_a, sd_m = 4, 0.10, 0.25, 0.10
    report = {"assumptions": {"parents_per_formula": gp, "contrast_icc": icc,
                              "sd_auc_diff": sd_a, "sd_each_mechanism_contrast": sd_m,
                              "primary_mechanism_correlation": 0.0, "reps": reps}}

    report["v20_reproduction_N300"] = dict(zip(
        ("GO", "NO_GO_total", "HOLD"),
        gate_v20(rng, reps, 300, gp, 0.05, 0.02, sd_a, sd_m, icc)))

    power = []
    for p, rho, sda in ((240, 0.5, sd_a), (300, 0.5, sd_a), (320, 0.5, sd_a),
                        (340, 0.5, sd_a), (400, 0.5, sd_a), (300, 0.3, sd_a),
                        (300, 0.7, sd_a), (300, 0.5, 0.30), (340, 0.5, 0.30)):
        r = gate_v21(rng, reps, p, gp, 0.05, 0.02, 0.02, sda, sd_m, icc, rho)
        power.append({"N": p, "rho_M0_MR": rho, "sd_auc_diff": sda, **r})
    report["v21_target_power"] = power

    scen = []
    for auc, m0, mr in ((0.05, 0.02, 0.02), (0.0, 0.02, 0.02), (0.05, 0.0, 0.0),
                        (0.05, 0.02, 0.0), (0.05, 0.0, 0.02), (0.0, 0.0, 0.0)):
        r = gate_v21(rng, reps, 300, gp, auc, m0, mr, sd_a, sd_m, icc, 0.5)
        scen.append({"auc": auc, "M0": m0, "MR": mr, **r})
    report["v21_scenarios_N300_rho05"] = scen

    report["v20_max_metric_bias"] = max_metric_bias(rng)
    json.dump(report, sys.stdout, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
