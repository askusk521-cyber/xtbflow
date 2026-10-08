"""Exact bounded BE edits; no learned or evaluation-derived pruning."""
from itertools import combinations_with_replacement, product

import numpy as np

from xtbflow.v1.data import canonical_event
from xtbflow.v1.proxy import valid_endpoint

VALENCE = {1: 1, 6: 4, 7: 5, 8: 6}
# RDKit strict explicit valence for CHNO at q=-1,0,+1 (isoelectronic rule).
# Carbon ions use the documented special-case maximum of three.
MAX_BONDS = {1: {-1: 1, 0: 1, 1: 1}, 6: {-1: 3, 0: 4, 1: 3},
             7: {-1: 2, 0: 3, 1: 4}, 8: {-1: 1, 0: 2, 1: 3}}


def bounded_formations(available, counts, capacity, number, start=0):
    """Enumerate sorted bond multisets, pruning saturated atoms before descent."""
    if number == 0:
        yield (), counts
        return
    for index in range(start, len(available)):
        i, j = available[index]
        if counts[i] >= capacity[i] or counts[j] >= capacity[j]:
            continue
        updated = counts.copy()
        updated[i] += 1
        updated[j] += 1
        for tail, degrees in bounded_formations(available, updated, capacity, number-1, index):
            yield ((i, j),) + tail, degrees


def enumerate_events(z, br, perms, size=2, stop_after=None):
    """Yield unique (channel_id, BE) within the frozen bNfN domain.

    A bond is never both broken and formed: edits denote net changes.
    Capacity pruning follows q >= -1 and the minimum allowed diagonal;
    hydrogen additionally obeys the explicit LewisState two-electron bound.
    stop_after is only for lower-bound feasibility audits, not final coverage.
    """
    z = np.asarray(z)
    br = np.asarray(br, dtype=int)
    perms = np.asarray(perms, dtype=int)
    n = len(z)
    valence = np.array([VALENCE[int(v)] for v in z])
    diag = np.diag(br)
    capacity = valence + 1 - np.maximum(0, diag - 2)
    capacity = np.minimum(capacity, np.array([max(MAX_BONDS[int(v)].values()) for v in z]))
    capacity[z == 1] = np.minimum(capacity[z == 1], 1)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    bonds = [(i, j) for i, j in pairs if br[i, j] > 0]
    seen = set()
    for nb in range(size + 1):
        for broken in combinations_with_replacement(bonds, nb):
            bp = br.copy()
            for i, j in broken:
                bp[i, j] -= 1
                bp[j, i] -= 1
            if np.any(bp < 0):
                continue
            counts = bp.sum(axis=1) - diag
            if np.any(counts > capacity):
                continue
            banned = set(broken)
            available = [(i, j) for i, j in pairs if (i, j) not in banned
                         and counts[i] < capacity[i] and counts[j] < capacity[j]]
            for nf in range(size + 1):
                for formed, degrees in bounded_formations(available, counts, capacity, nf):
                    if not broken and not formed:
                        continue
                    touched = {v for pair in broken + formed for v in pair}
                    options = []
                    for i in range(n):
                        changes = (-2, 0, 2) if i in touched else (0,)
                        options.append([d for d in changes if diag[i] + d >= 0
                                        and abs(valence[i] - degrees[i] - diag[i] - d) <= 1
                                        and degrees[i] <= MAX_BONDS[int(z[i])][int(valence[i] - degrees[i] - diag[i] - d)]
                                        and (z[i] != 1 or diag[i] + d + 2 * degrees[i] <= 2)])
                    if any(not op for op in options):
                        continue
                    for changes in product(*options):
                        if sum(changes) + 2 * (nf - nb) != 0:
                            continue
                        candidate = bp.copy()
                        for i, j in formed:
                            candidate[i, j] += 1
                            candidate[j, i] += 1
                        candidate[np.diag_indices(n)] += changes
                        if not valid_endpoint(z, br, candidate):
                            continue
                        channel = canonical_event(br, candidate, perms)
                        if channel in seen:
                            continue
                        seen.add(channel)
                        yield channel, candidate
                        if stop_after is not None and len(seen) >= stop_after:
                            return
