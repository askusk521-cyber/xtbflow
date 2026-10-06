"""Pad variable-size reactions into dense batches."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .t1x_data import T1xCache

ELEMENT_INDEX = {1: 1, 6: 2, 7: 3, 8: 4}  # 0 = padding
N_ELEMENT_TYPES = 5


class T1xDataset(Dataset):
    def __init__(self, cache: T1xCache, split: int, limit: int | None = None):
        self.cache = cache
        idx = np.flatnonzero(cache.split == split)
        self.idx = idx if limit is None else idx[:limit]

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, k: int) -> dict:
        return self.cache.reaction(int(self.idx[k]))


def collate(items: list[dict]) -> dict[str, torch.Tensor]:
    bsz = len(items)
    n = max(len(it["z"]) for it in items)
    out = {
        "z": torch.zeros(bsz, n, dtype=torch.long),
        "atom_mask": torch.zeros(bsz, n, dtype=torch.bool),
        "x_r": torch.zeros(bsz, n, 3),
        "x_ts": torch.zeros(bsz, n, 3),
        "x_p": torch.zeros(bsz, n, 3),
        "b_r": torch.zeros(bsz, n, n),
        "b_p": torch.zeros(bsz, n, n),
        "index": torch.tensor([it["index"] for it in items], dtype=torch.long),
    }
    for k, it in enumerate(items):
        m = len(it["z"])
        out["z"][k, :m] = torch.tensor([ELEMENT_INDEX[int(v)] for v in it["z"]])
        out["atom_mask"][k, :m] = True
        for key in ("x_r", "x_ts", "x_p"):
            out[key][k, :m] = torch.from_numpy(np.asarray(it[key], dtype=np.float32))
        for key in ("b_r", "b_p"):
            out[key][k, :m, :m] = torch.from_numpy(np.asarray(it[key], dtype=np.float32))
    return out
