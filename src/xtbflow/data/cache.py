"""Append-only JSONL cache for E/F pairs with conflict detection."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .pairing import EFPair


def load_pair_cache(path: str | Path) -> dict[str, EFPair]:
    """Load all rows and reject malformed or conflicting duplicate identities."""

    target = Path(path)
    if not target.exists():
        return {}
    output: dict[str, EFPair] = {}
    with target.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                pair = EFPair.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid E/F cache row at line {line_number}: {exc}") from exc
            previous = output.get(pair.pair_id)
            if previous is not None and previous.to_dict() != pair.to_dict():
                raise ValueError(f"conflicting duplicate pair_id: {pair.pair_id}")
            output[pair.pair_id] = pair
    return output


def append_pair_cache(path: str | Path, pairs: Iterable[EFPair]) -> tuple[int, int]:
    """Append unseen rows; return ``(written, cache_hits)``.

    Existing identities are never overwritten. An identical row is a cache hit;
    a different row with the same identity fails closed.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = load_pair_cache(target)
    pending: list[EFPair] = []
    hits = 0
    for pair in pairs:
        previous = existing.get(pair.pair_id)
        if previous is not None:
            if previous.to_dict() != pair.to_dict():
                raise ValueError(f"conflicting duplicate pair_id: {pair.pair_id}")
            hits += 1
        else:
            existing[pair.pair_id] = pair
            pending.append(pair)
    if pending:
        with target.open("a", encoding="utf-8") as handle:
            for pair in pending:
                handle.write(json.dumps(pair.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return len(pending), hits

