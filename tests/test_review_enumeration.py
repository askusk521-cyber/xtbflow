"""Tiny exhaustive oracle checks, independent of catalogue evaluation data."""
from itertools import product

import numpy as np

from xtbflow.v1.data import canonical_event
from xtbflow.v1.enumeration import enumerate_events, bounded_formations, minimizing_permutation
from xtbflow.v1.proxy import valid_endpoint


def test_vectorized_lexicographic_minimizer():
    from itertools import permutations
    rng = np.random.default_rng(1729)
    perms = np.array(list(permutations(range(5))))
    for _ in range(20):
        br = rng.integers(0, 4, (5, 5))
        bp = rng.integers(0, 4, (5, 5))
        assert canonical_event(br, bp, perms) == canonical_event(br, bp, minimizing_permutation(br, bp, perms))


def test_break_orbit_pruning_preserves_canonical_products():
    from itertools import permutations
    z = np.array([6, 1, 1, 1, 1])
    br = np.zeros((5, 5), dtype=int)
    br[0, 1:] = 1
    br[1:, 0] = 1
    perms = np.array([(0,) + p for p in permutations((1, 2, 3, 4))])
    identity = np.arange(5)[None, :]
    labelled = enumerate_events(z, br, identity, 2)
    expected = {canonical_event(br, b, perms) for _, b in labelled}
    actual = {c for c, _ in enumerate_events(z, br, perms, 2)}
    assert actual == expected


def test_capacity_pruned_multisets():
    from itertools import combinations_with_replacement
    pairs = [(0, 1), (0, 2), (1, 2)]
    counts = np.array([0, 0, 1])
    capacity = np.array([2, 2, 2])
    for number in range(4):
        expected = set()
        for edits in combinations_with_replacement(pairs, number):
            deg = counts.copy()
            for i, j in edits:
                deg[i] += 1
                deg[j] += 1
            if np.all(deg <= capacity):
                expected.add(edits)
        actual = {edits for edits, _ in bounded_formations(pairs, counts, capacity, number)}
        assert actual == expected


def test_exhaustive_water_domain():
    z = np.array([8, 1, 1])
    br = np.array([[4, 1, 1], [1, 0, 0], [1, 0, 0]])
    perms = np.array([[0, 1, 2], [0, 2, 1]])
    expected = set()
    pairs = [(0, 1), (0, 2), (1, 2)]
    for bonds in product(range(3), repeat=3):
        for ds in product((-2, 0, 2), repeat=3):
            bp = br.copy()
            bp[np.diag_indices(3)] += ds
            for (i, j), value in zip(pairs, bonds):
                bp[i, j] = bp[j, i] = value
            delta = bp - br
            off = delta.copy()
            np.fill_diagonal(off, 0)
            upper = off[np.triu_indices(3, 1)]
            if -upper[upper < 0].sum() > 2 or upper[upper > 0].sum() > 2:
                continue
            if any(ds[i] and not np.any(off[i]) for i in range(3)):
                continue
            if np.any(np.abs(np.array([6, 1, 1]) - bp.sum(axis=1)) > 1):
                continue
            if valid_endpoint(z, br, bp):
                expected.add(canonical_event(br, bp, perms))
    actual = list(enumerate_events(z, br, perms, 2))
    assert {channel for channel, _ in actual} == expected
    assert len(actual) == len(expected)
