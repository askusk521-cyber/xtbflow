"""Event matching up to reactant automorphism, RMSD, per-query metrics, bootstrap, decision."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

VALENCE_E = {1: 1, 6: 4, 7: 5, 8: 6}
DELTAS = tuple(round(0.1 * k, 1) for k in range(1, 11))  # 0.1 ... 1.0 Å


def change_set(b_r: np.ndarray, b_p: np.ndarray) -> frozenset:
    changed = np.triu((b_r > 0) != (b_p > 0), k=1)
    return frozenset((int(i), int(j)) for i, j in np.argwhere(changed))


def be_to_mol(z: np.ndarray, b: np.ndarray, sanitize: bool = True):
    from rdkit import Chem

    order = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    mol = Chem.RWMol()
    n = len(z)
    for i in range(n):
        atom = Chem.Atom(int(z[i]))
        atom.SetFormalCharge(int(VALENCE_E[int(z[i])] - b[i].sum()))  # 行和含对角元
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    for i in range(n):
        for j in range(i + 1, n):
            if b[i, j] > 0:
                mol.AddBond(i, j, order[int(b[i, j])])
    mol = mol.GetMol()
    if sanitize:
        Chem.SanitizeMol(mol)
    else:
        mol.UpdatePropertyCache(strict=False)
    return mol


def is_valid_product(z: np.ndarray, b_r: np.ndarray, b_dec: np.ndarray | None) -> bool:
    if b_dec is None or len(change_set(b_r, b_dec)) == 0:
        return False
    try:
        be_to_mol(z, b_dec, sanitize=True)
    except Exception:
        return False
    return True


def automorphisms(z: np.ndarray, b_r: np.ndarray, max_matches: int = 10000) -> tuple[np.ndarray, bool]:
    """perms[k][q] = image of atom q under automorphism k; second value = cap reached."""
    try:
        mol = be_to_mol(z, b_r, sanitize=True)
        matches = mol.GetSubstructMatches(mol, uniquify=False, useChirality=False, maxMatches=max_matches)
    except Exception:
        matches = ()
    if not matches:
        return np.arange(len(z))[None, :], False
    return np.asarray(matches, dtype=np.int64), len(matches) >= max_matches


def reference_index(ref_pairs: frozenset, perms: np.ndarray) -> dict:
    """{relabelled reference change set: [k, ...]} with relabelled = {(pi(i), pi(j))}."""
    index: dict = defaultdict(list)
    for k, pi in enumerate(perms):
        relabelled = frozenset(tuple(sorted((int(pi[i]), int(pi[j])))) for i, j in ref_pairs)
        index[relabelled].append(k)
    return index


def batched_kabsch_rmsd(mobile: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """mobile [K,N,3], ref [N,3] -> [K] RMSD after the best proper rotation."""
    m = mobile - mobile.mean(axis=1, keepdims=True)
    r = ref - ref.mean(axis=0)
    h = np.einsum("kni,nj->kij", m, r)
    u, s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u) * np.linalg.det(vt))
    trace = s[:, 0] + s[:, 1] + d * s[:, 2]
    msd = ((m ** 2).sum(axis=(1, 2)) + (r ** 2).sum() - 2.0 * trace) / ref.shape[0]
    return np.sqrt(np.clip(msd, 0.0, None))


def evaluate_query(z, b_r, x_r, b_p_ref, x_ts_ref, cand_b_dec: list, cand_x: np.ndarray,
                   perms: np.ndarray) -> dict:
    """One test reaction and its S candidates (cand_b_dec[k] is an int matrix or None)."""
    index = reference_index(change_set(b_r, b_p_ref), perms)
    d_r = np.linalg.norm(x_r[:, None] - x_r[None], axis=-1)
    valid, best_rmsd, consistency, events = [], [], [], set()
    for b_dec, x in zip(cand_b_dec, cand_x):
        ok = is_valid_product(z, b_r, b_dec)
        valid.append(ok)
        if not ok:
            continue
        pairs = change_set(b_r, b_dec)
        events.add(pairs)
        d_c = np.linalg.norm(x[:, None] - x[None], axis=-1)
        agree = [(d_c[i, j] < d_r[i, j]) if b_dec[i, j] > 0 else (d_c[i, j] > d_r[i, j]) for i, j in pairs]
        consistency.append(float(np.mean(agree)))
        ks = index.get(pairs)
        if ks:
            best_rmsd.append(float(batched_kabsch_rmsd(x[perms[ks]], x_ts_ref).min()))
    best = min(best_rmsd) if best_rmsd else float("inf")
    return {
        "valid_frac": float(np.mean(valid)),
        "hit_event": bool(best_rmsd),
        "hit": {str(d): bool(best <= d) for d in DELTAS},
        "best_rmsd": best if best_rmsd else None,
        "consistency": float(np.mean(consistency)) if consistency else None,
        "unique_valid_events": len(events),
    }


def parent_means(per_query: dict[int, float], parent_of: dict[int, str]) -> dict[str, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for q, value in per_query.items():
        groups[parent_of[q]].append(value)
    return {p: float(np.mean(v)) for p, v in groups.items()}


def paired_bootstrap(a: dict[str, float], b: dict[str, float], n_boot: int, seed: int) -> dict:
    if set(a) != set(b):
        raise ValueError("arms must cover the same parents")
    parents = sorted(a)
    va, vb = np.array([a[p] for p in parents]), np.array([b[p] for p in parents])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(parents), size=(n_boot, len(parents)))
    diffs = (vb[idx] - va[idx]).mean(axis=1)
    return {"M_A": float(va.mean()), "M_B": float(vb.mean()), "delta": float((vb - va).mean()),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "n_parents": len(parents)}


def decide(primary: dict, seed_deltas: list[float], valid: dict) -> str:
    low, high = primary["ci95"]
    if low > 0 and sum(d > 0 for d in seed_deltas) >= 2 and valid["ci95"][0] > -0.02:
        return "GO"
    if high < 0.02 or primary["delta"] <= 0:
        return "NO-GO"
    return "INCONCLUSIVE"
