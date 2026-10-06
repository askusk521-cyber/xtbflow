"""Transition1x -> M0 CSR cache.  Building needs h5py + rdkit; loading needs numpy only."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np

SPLITS = {"train": 0, "val": 1, "test": 2}
SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O"}
VALENCE_E = {1: 1, 6: 4, 7: 5, 8: 6}
POS_KEY = "positions"
ENERGY_KEY = "wB97x_6-31G(d).energy"  # 以 4.2 打印出的实际键名为准


class DropReaction(Exception):
    """Raised to exclude one reaction; str(exc) is the reason code."""


@dataclass(frozen=True)
class RawReaction:
    split: str
    formula: str
    rxn: str
    z: np.ndarray      # [N] int64
    x_r: np.ndarray    # [N,3] float64, Å
    x_ts: np.ndarray
    x_p: np.ndarray
    e_r: float         # eV
    e_ts: float
    e_p: float


def iter_split(h5_path: Path, split: str, limit: int | None = None) -> Iterator[RawReaction]:
    import h5py

    count = 0
    with h5py.File(h5_path, "r") as f:
        root = f[split]
        for formula in sorted(root.keys()):
            for rxn in sorted(root[formula].keys()):
                g = root[formula][rxn]
                z = np.asarray(g["atomic_numbers"], dtype=np.int64).reshape(-1)

                def frame(name: str) -> tuple[np.ndarray, float]:
                    sub = g[name]
                    x = np.asarray(sub[POS_KEY], dtype=np.float64)
                    x = x[0] if x.ndim == 3 else x
                    e = float(np.asarray(sub[ENERGY_KEY], dtype=np.float64).reshape(-1)[0])
                    return x, e

                x_r, e_r = frame("reactant")
                x_ts, e_ts = frame("transition_state")
                x_p, e_p = frame("product")
                yield RawReaction(split, formula, rxn, z, x_r, x_ts, x_p, e_r, e_ts, e_p)
                count += 1
                if limit is not None and count >= limit:
                    return


def kabsch_align(mobile: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Rotate (proper rotation only) and translate `mobile` onto `ref`.
    Returns coordinates centred at ref's geometric centre."""
    ref_c = ref - ref.mean(axis=0)
    mob_c = mobile - mobile.mean(axis=0)
    h = mob_c.T @ ref_c
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return mob_c @ rot.T + ref.mean(axis=0)


def perceive_mol(z: np.ndarray, x: np.ndarray):
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    lines = [str(len(z)), ""] + [
        f"{SYMBOL[int(a)]} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for a, p in zip(z, x)
    ]
    mol = Chem.MolFromXYZBlock("\n".join(lines) + "\n")
    if mol is None:
        raise DropReaction("xyz_parse_failed")
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:  # RDKit 不同版本抛不同异常类型
        raise DropReaction("determine_bonds_failed") from None
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        raise DropReaction("radical")
    if sum(a.GetFormalCharge() for a in mol.GetAtoms()) != 0:
        raise DropReaction("net_charge")
    if [a.GetAtomicNum() for a in mol.GetAtoms()] != [int(v) for v in z]:
        raise DropReaction("atom_order_changed")
    return mol


def be_matrix(mol) -> np.ndarray:
    n = mol.GetNumAtoms()
    b = np.zeros((n, n), dtype=np.int64)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        order = bond.GetBondTypeAsDouble()
        if order not in (1.0, 2.0, 3.0):
            raise DropReaction("non_integer_bond_order")
        b[i, j] = b[j, i] = int(order)
    bond_sum = b.sum(axis=1)  # 此时对角线为 0
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        lone = VALENCE_E[atom.GetAtomicNum()] - atom.GetFormalCharge() - int(bond_sum[i])
        if lone < 0 or lone % 2:
            raise DropReaction("bad_lone_electrons")
        b[i, i] = lone
    return b


def _validate_lewis(z: np.ndarray, b: np.ndarray) -> None:
    from mechai.data.events import LewisState

    symbols = tuple(SYMBOL[int(v)] for v in z)
    rows = tuple(tuple(int(v) for v in row) for row in b)
    try:
        LewisState(symbols, rows, 0, 1).validate()
    except ValueError:
        raise DropReaction("lewis_invalid") from None


def connectivity_change(b_r: np.ndarray, b_p: np.ndarray) -> np.ndarray:
    """Upper-triangle pairs (i<j) whose bonded/non-bonded status changes. Returns [K,2] int."""
    changed = (b_r > 0) != (b_p > 0)
    changed = np.triu(changed, k=1)
    return np.argwhere(changed)


def process_reaction(raw: RawReaction) -> dict:
    from rdkit import Chem

    if not {int(v) for v in raw.z} <= set(SYMBOL):
        raise DropReaction("element_out_of_scope")
    n = len(raw.z)
    for x in (raw.x_r, raw.x_ts, raw.x_p):
        if x.shape != (n, 3) or not np.isfinite(x).all():
            raise DropReaction("bad_coordinates")
    mol_r = perceive_mol(raw.z, raw.x_r)
    mol_p = perceive_mol(raw.z, raw.x_p)
    b_r, b_p = be_matrix(mol_r), be_matrix(mol_p)
    _validate_lewis(raw.z, b_r)
    _validate_lewis(raw.z, b_p)
    pairs = connectivity_change(b_r, b_p)
    if len(pairs) == 0:
        raise DropReaction("no_connectivity_change")
    off = ~np.eye(n, dtype=bool)
    order_only = int(np.triu((b_r != b_p) & off & ((b_r > 0) == (b_p > 0)), k=1).sum())
    x_r = raw.x_r - raw.x_r.mean(axis=0)
    try:
        smiles_r = Chem.MolToSmiles(Chem.RemoveHs(mol_r), isomericSmiles=False)
        smiles_p = Chem.MolToSmiles(Chem.RemoveHs(mol_p), isomericSmiles=False)
    except Exception:
        raise DropReaction("smiles_failed") from None
    return {
        "split": SPLITS[raw.split],
        "formula": raw.formula,
        "rxn_key": f"{raw.formula}/{raw.rxn}",
        "z": raw.z.astype(np.int8),
        "x_r": x_r.astype(np.float32),
        "x_ts": kabsch_align(raw.x_ts, x_r).astype(np.float32),
        "x_p": kabsch_align(raw.x_p, x_r).astype(np.float32),
        "b_r": b_r.astype(np.int8),
        "b_p": b_p.astype(np.int8),
        "energies": np.array([raw.e_r, raw.e_ts, raw.e_p], dtype=np.float64),
        "smiles_r": smiles_r,
        "smiles_p": smiles_p,
        "parent_id": hashlib.sha1(smiles_r.encode()).hexdigest()[:12],
        "n_changed": len(pairs),
        "order_only": order_only,
        "any_formal_charge": any(a.GetFormalCharge() != 0 for m in (mol_r, mol_p) for a in m.GetAtoms()),
    }


def write_cache(records: list[dict], path: Path) -> None:
    n_atoms = np.array([len(r["z"]) for r in records], dtype=np.int64)
    atom_off = np.concatenate([[0], np.cumsum(n_atoms)])
    pair_off = np.concatenate([[0], np.cumsum(n_atoms ** 2)])
    cat = lambda key: np.concatenate([r[key].reshape(-1) for r in records])
    np.savez_compressed(
        path,
        n_atoms=n_atoms, atom_off=atom_off, pair_off=pair_off,
        z=cat("z"),
        x_r=cat("x_r").reshape(-1, 3), x_ts=cat("x_ts").reshape(-1, 3), x_p=cat("x_p").reshape(-1, 3),
        b_r=cat("b_r"), b_p=cat("b_p"),
        energies=np.stack([r["energies"] for r in records]),
        split=np.array([r["split"] for r in records], dtype=np.int8),
        parent_id=np.array([r["parent_id"] for r in records]),
        rxn_key=np.array([r["rxn_key"] for r in records]),
        formula=np.array([r["formula"] for r in records]),
        smiles_r=np.array([r["smiles_r"] for r in records]),
        smiles_p=np.array([r["smiles_p"] for r in records]),
    )


class T1xCache:
    """Read-only view of the CSR cache.  No pickle is used."""

    def __init__(self, path: Path):
        data = np.load(path, allow_pickle=False)
        self.arrays = {key: data[key] for key in data.files}
        self.split = self.arrays["split"]
        self.parent_id = self.arrays["parent_id"]

    def __len__(self) -> int:
        return len(self.split)

    def reaction(self, i: int) -> dict:
        a = self.arrays
        s, e = int(a["atom_off"][i]), int(a["atom_off"][i + 1])
        ps, pe = int(a["pair_off"][i]), int(a["pair_off"][i + 1])
        n = e - s
        return {
            "z": a["z"][s:e].astype(np.int64),
            "x_r": a["x_r"][s:e], "x_ts": a["x_ts"][s:e], "x_p": a["x_p"][s:e],
            "b_r": a["b_r"][ps:pe].reshape(n, n).astype(np.float32),
            "b_p": a["b_p"][ps:pe].reshape(n, n).astype(np.float32),
            "parent_id": str(a["parent_id"][i]),
            "index": i,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:  # 分块读取，避免把 6.6 GB 读进内存
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_cache(h5_path: Path, out_path: Path, limit: int | None = None) -> dict:
    records: list[dict] = []
    dropped: Counter[str] = Counter()
    seen = Counter()
    for split in SPLITS:
        for raw in iter_split(h5_path, split, limit):
            seen[split] += 1
            try:
                records.append(process_reaction(raw))
            except DropReaction as exc:
                dropped[f"{split}:{exc}"] += 1
    by_split = {s: [r for r in records if r["split"] == v] for s, v in SPLITS.items()}
    for key in ("formula", "parent_id"):
        sets = {s: {r[key] for r in rs} for s, rs in by_split.items()}
        for a in SPLITS:
            for b in SPLITS:
                if a < b and sets[a] & sets[b]:
                    raise RuntimeError(f"leakage: {key} shared by {a} and {b}")
    write_cache(records, out_path)
    parents = {s: Counter(r["parent_id"] for r in rs) for s, rs in by_split.items()}
    manifest = {
        "schema": "xtbflow-m0-t1x-cache/v1",
        "h5_sha256": _sha256_file(Path(h5_path)) if limit is None else "skipped_for_limit",
        "cache_sha256": hashlib.sha256(Path(out_path).read_bytes()).hexdigest(),
        "seen": dict(seen),
        "kept": {s: len(rs) for s, rs in by_split.items()},
        "dropped": dict(sorted(dropped.items())),
        "parents": {s: len(c) for s, c in parents.items()},
        "reactions_per_parent_max": {s: max(c.values()) if c else 0 for s, c in parents.items()},
        "max_atoms": int(max(len(r["z"]) for r in records)),
        "order_only_reactions": int(sum(r["order_only"] > 0 for r in records)),
        "formal_charge_reactions": int(sum(r["any_formal_charge"] for r in records)),
    }
    return manifest
