"""Exact bounded BE edits; no learned or evaluation-derived pruning."""
from itertools import combinations_with_replacement, product

import numpy as np

from xtbflow.v1.data import canonical_event
from xtbflow.v1.proxy import valid_endpoint

VALENCE = {1: 1, 6: 4, 7: 5, 8: 6}
# RDKit strict explicit valence for CHNO at q=-1,0,+1 (isoelectronic rule).
# Positive carbon permits five in RDKit 2024.03; nonnegative BE diagonals
# further limit its feasible degree. Bounds are tested against this sanitizer.
MAX_BONDS = {1: {-1: 1, 0: 1, 1: 1}, 6: {-1: 3, 0: 4, 1: 5},
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


def minimizing_permutation(br, bp, perms):
    """Select an exact lexicographic minimizer; canonical_event still makes the ID.

    Filtering tied rows columnwise avoids constructing thousands of Python
    tuples for high-symmetry reactants. It changes neither quotient nor hash.
    """
    if len(perms) < 16:
        return perms
    delta = np.asarray(bp, dtype=np.int64) - np.asarray(br, dtype=np.int64)
    triangle = np.triu_indices(len(delta))
    values = delta[perms[:, :, None], perms[:, None, :]][:, triangle[0], triangle[1]]
    active = np.arange(len(perms))
    for column in range(values.shape[1]):
        entries = values[active, column]
        active = active[entries == entries.min()]
        if len(active) == 1:
            break
    return perms[active[:1]]


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
    option_table = {}
    for i in range(n):
        for degree in range(int(capacity[i]) + 1):
            for touched in (False, True):
                choices = (-2, 0, 2) if touched else (0,)
                option_table[i, degree, touched] = tuple(d for d in choices
                    if diag[i] + d >= 0
                    and abs(valence[i] - degree - diag[i] - d) <= 1
                    and degree <= MAX_BONDS[int(z[i])][int(valence[i] - degree - diag[i] - d)]
                    and (z[i] != 1 or diag[i] + d + 2 * degree <= 2))
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    bonds = [(i, j) for i, j in pairs if br[i, j] > 0]
    seen = set()
    for nb in range(size + 1):
        broken_orbits = set()
        for broken in combinations_with_replacement(bonds, nb):
            if broken in broken_orbits:
                continue
            # Reactant automorphisms preserve the entire enumeration domain.
            # Enumerate formations from one representative broken multiset;
            # every omitted product has the same canonical_event as a retained one.
            for permutation in perms:
                mapped = tuple(sorted(tuple(sorted((int(permutation[i]), int(permutation[j]))))
                                      for i, j in broken))
                broken_orbits.add(mapped)
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
                    options = [option_table[i, int(degrees[i]), i in touched] for i in range(n)]
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
                        channel = canonical_event(br, candidate, minimizing_permutation(br, candidate, perms))
                        if channel in seen:
                            continue
                        seen.add(channel)
                        yield channel, candidate
                        if stop_after is not None and len(seen) >= stop_after:
                            return
