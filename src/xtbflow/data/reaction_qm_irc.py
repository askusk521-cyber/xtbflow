"""IRC evidence for Reaction-QM records (``B3LYPD3_TZVP_IRC.h5``).

The IRC asset adds two things the main HDF5 lacks:

* **forces** along the reaction path, so an ``energy_force`` task can exist;
* a **path that leaves the TS in both directions**, so we can test, from
  geometry alone, whether the TS really connects the reactant and product
  graphs that the event label claims.

Design notes (why this module is shaped the way it is)
------------------------------------------------------
* IRC frames are stored in the TS atom order, so an IRC end frame is already in
  *global* map order.  Using it as the reactant/product geometry therefore needs
  no endpoint local->global mapping and sidesteps the symmetry ambiguity that
  blocks most records in the main funnel.  It is a *path-derived* endpoint
  (``irc_end_frame``), not an independently optimised minimum; the origin is
  recorded so downstream code cannot mistake it for an independent seed.
* Units are not assumed.  Energy/force consistency is *measured* from the work
  identity ``E[i+1]-E[i] ~ -0.5 (F[i]+F[i+1]) . (x[i+1]-x[i])`` and the TS frame
  is compared with the main HDF5; both are reported as distributions.
* Edge lengths at the IRC ends are used only to check the *formed/broken sigma
  bonds named by the event*, never to build labels.  Ends are often not fully
  relaxed, so the primary test is directional (did each named edge move the way
  the event says) and the absolute bonded-state test is a stricter tier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

IRC_SCHEMA = "xtbflow-reaction-qm-irc-evidence/v1"
HARTREE_TO_EV = 27.211386245988  # CODATA 2018

# Cordero et al. 2008 single-bond covalent radii (angstrom) for the CHNOS pool.
COVALENT_RADII = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 16: 1.05}
BOND_SCALE = 1.25  # same covalent-radius factor the DA adapter uses for connectivity

# A-priori gates (chosen before looking at real IRC data).
TS_COORD_TOLERANCE = 1e-3  # angstrom, max per-atom deviation of IRC frame 0 from the main-HDF5 TS *after rigid alignment*
TS_ENERGY_TOLERANCE_HARTREE = 1e-3
MIN_FRAMES = 3
# The aggregate work-identity slope must lie in this window for energies and
# forces to be called mutually consistent.
UNIT_SLOPE_WINDOW = (0.9, 1.1)

ORIENT_CONNECTS = "connects_declared_reactant_and_product"
ORIENT_CONTRADICTS = "contradicts_declared_event"
ORIENT_NOT_TESTABLE = "not_testable_no_sigma_bond_change"
ORIENT_AMBIGUOUS = "ambiguous_both_orientations_fit"
ORIENT_NO_BRANCHES = "no_two_branches"
ORIENT_NO_CHANGE = "named_edges_do_not_change_along_path"

# Directional test (the primary criterion): an edge counts as moving the way the
# event says only if its length changes by more than this between the two ends.
# Chosen a priori; IRC ends are often not fully relaxed, so the absolute
# bonded/non-bonded state is reported separately as a stricter tier.
DISTANCE_DEADBAND = 0.05  # angstrom
MIN_EDGE_FRACTION = 0.5  # share of named edges that must move clearly the right way


@dataclass(frozen=True)
class IrcInput:
    atomic_numbers: tuple[int, ...]
    coordinates: np.ndarray  # (steps, atoms, 3)
    energies: np.ndarray  # (steps,)
    forces: np.ndarray  # (steps, atoms, 3)


@dataclass
class IrcResult:
    record_id: str
    present: bool = False
    identity_ok: bool = False
    ts_frame_matches: bool = False
    n_frames: int = 0
    branch_boundary: int | None = None
    continuity_ok: bool = False
    orientation: str = ORIENT_NO_BRANCHES
    reactant_end_frame: int | None = None
    product_end_frame: int | None = None
    strict_bond_state_match: bool = False
    edges_correct: int = 0
    edges_wrong: int = 0
    edges_flat: int = 0
    ts_coord_max_dev: float | None = None
    ts_energy_dev_hartree: float | None = None
    work_slope: float | None = None
    work_correlation: float | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def energy_force_ok(self) -> bool:
        """Frames are usable as E/F points: identity and finiteness verified and
        the TS frame reproduces the main HDF5.  Unit consistency is an aggregate
        property decided in the report, not per record."""

        return self.present and self.identity_ok and self.ts_frame_matches

    @property
    def path_pairing_ok(self) -> bool:
        return self.energy_force_ok and self.orientation == ORIENT_CONNECTS


# --------------------------------------------------------------------------
# Rigid alignment
# --------------------------------------------------------------------------


def aligned_max_deviation(a: np.ndarray, b: np.ndarray) -> float:
    """Largest per-atom distance between ``a`` and ``b`` after the optimal proper
    rotation and translation (Kabsch, reflections excluded).

    IRC runs are stored in their own Cartesian frame, so the same TS can appear
    rotated/translated relative to the main HDF5.  Comparing raw coordinates would
    reject those records although they are identical geometries; comparing after
    alignment still rejects a permuted atom order or a different structure.
    """

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        return float("inf")
    a0, b0 = a - a.mean(axis=0), b - b.mean(axis=0)
    u, _, vt = np.linalg.svd(a0.T @ b0)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d if d != 0 else 1.0]) @ vt
    return float(np.max(np.linalg.norm(a0 @ rot - b0, axis=1)))

# --------------------------------------------------------------------------
# Path structure
# --------------------------------------------------------------------------


def find_branch_boundary(energies: Sequence[float]) -> int | None:
    """Index where the second IRC branch starts, or ``None``.

    The source concatenates the two IRC directions as ``[TS, branch A, branch
    B]``, each branch descending from the TS.  The upstream example script marks
    the switch where the energy jumps back up by more than the depth already
    below the TS; this is the same rule, written so that it cannot trigger on
    optimisation noise near a minimum (the jump must also exceed the depth).
    """

    e = np.asarray(energies, dtype=float)
    if e.size < MIN_FRAMES:
        return None
    ts = e[0]
    for i in range(1, e.size):
        if (ts - e[i]) < (e[i] - e[i - 1]):
            return i
    return None


def _continuity(coordinates: np.ndarray, boundary: int) -> bool:
    """Branch B must restart next to the TS, i.e. closer to frame 0 than the
    last frame of branch A is.  A stored order that merely *looks* like two
    branches in energy but is geometrically continuous is rejected."""

    start = coordinates[0]
    restart = float(np.linalg.norm(coordinates[boundary] - start))
    end_of_a = float(np.linalg.norm(coordinates[boundary - 1] - start))
    return restart < end_of_a


# --------------------------------------------------------------------------
# Distance connectivity (heuristic, used only to test named sigma-bond edits)
# --------------------------------------------------------------------------


def bonded(atomic_numbers: Sequence[int], xyz: np.ndarray, i: int, j: int) -> bool:
    """1-based global ids ``i``/``j`` are bonded in this frame (distance test)."""

    zi, zj = atomic_numbers[i - 1], atomic_numbers[j - 1]
    cutoff = BOND_SCALE * (COVALENT_RADII[zi] + COVALENT_RADII[zj])
    return float(np.linalg.norm(xyz[i - 1] - xyz[j - 1])) < cutoff


def _score(
    z: Sequence[int],
    xyz: np.ndarray,
    formed: Sequence[tuple[int, int]],
    broken: Sequence[tuple[int, int]],
    *,
    as_reactant: bool,
) -> float:
    """Fraction of named sigma-bond edits whose bonded/non-bonded state matches
    the hypothesis that this frame is the reactant (or the product) end."""

    total = len(formed) + len(broken)
    ok = 0
    for i, j in broken:  # present in the reactant, absent in the product
        ok += bonded(z, xyz, i, j) == as_reactant
    for i, j in formed:  # absent in the reactant, present in the product
        ok += bonded(z, xyz, i, j) != as_reactant
    return ok / total


def _edge_distance(xyz: np.ndarray, edge: tuple[int, int]) -> float:
    return float(np.linalg.norm(xyz[edge[0] - 1] - xyz[edge[1] - 1]))


def orient_branches(
    z: Sequence[int],
    end_a: np.ndarray,
    end_b: np.ndarray,
    formed: Sequence[tuple[int, int]],
    broken: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Decide which branch end is the reactant, directionally.

    For every formed edge the distance must be *shorter* at the product end, and
    for every broken edge *longer*, by more than ``DISTANCE_DEADBAND``.  The
    orientation is accepted only if no edge moves the wrong way and at least
    ``MIN_EDGE_FRACTION`` of the named edges move clearly the right way.  This
    tolerates IRC ends that stop short of a fully relaxed product, which a
    bonded/non-bonded threshold would call a contradiction.

    ``strict`` additionally reports whether both ends reproduce the absolute
    bonded/non-bonded state of every named edge (covalent-radius heuristic).
    Only formed/broken sigma bonds can discriminate; a pure bond-order change
    leaves both ends bonded.
    """

    edges = [(e, "formed") for e in formed] + [(e, "broken") for e in broken]
    result: dict[str, Any] = {
        "orientation": ORIENT_NOT_TESTABLE, "reactant_branch": None, "strict": False,
        "correct": 0, "wrong": 0, "flat": 0,
    }
    if not edges:
        return result
    # Positive "toward_b" means the edge looks like it changes from A to B in the
    # direction "A is reactant, B is product".
    correct_ab = wrong_ab = flat = 0
    for edge, kind in edges:
        delta = _edge_distance(end_b, edge) - _edge_distance(end_a, edge)  # B minus A
        if abs(delta) <= DISTANCE_DEADBAND:
            flat += 1
        elif (kind == "formed") == (delta < 0):  # formed shortens toward B, broken lengthens toward B
            correct_ab += 1
        else:
            wrong_ab += 1
    total = len(edges)
    # Reversing the orientation turns every correct edge into a wrong one.
    min_edges = max(1, int(np.ceil(MIN_EDGE_FRACTION * total)))
    if correct_ab + wrong_ab == 0:
        result.update(orientation=ORIENT_NO_CHANGE, flat=flat)
        return result
    if wrong_ab == 0 and correct_ab >= min_edges:
        branch, correct, wrong = "A", correct_ab, wrong_ab
    elif correct_ab == 0 and wrong_ab >= min_edges:
        branch, correct, wrong = "B", wrong_ab, correct_ab
    else:
        result.update(orientation=ORIENT_CONTRADICTS, correct=max(correct_ab, wrong_ab), wrong=min(correct_ab, wrong_ab), flat=flat)
        return result
    reactant_end, product_end = (end_a, end_b) if branch == "A" else (end_b, end_a)
    strict = _score(z, reactant_end, formed, broken, as_reactant=True) == 1.0 and _score(
        z, product_end, formed, broken, as_reactant=False
    ) == 1.0
    result.update(
        orientation=ORIENT_CONNECTS, reactant_branch=branch, strict=bool(strict), correct=correct, wrong=wrong, flat=flat
    )
    return result

# --------------------------------------------------------------------------
# Energy / force consistency
# --------------------------------------------------------------------------


def work_identity(
    coordinates: np.ndarray, energies: np.ndarray, forces: np.ndarray, *, boundary: int | None
) -> tuple[float | None, float | None]:
    """Least-squares slope of ``dE`` against the trapezoid work of the forces.

    Slope ~ +1 means energies and forces are mutually consistent *in whatever
    units they are stored* (no conversion factor hiding in between); ~ -1 means
    ``forces`` are actually gradients.  Segments that straddle the branch
    boundary are skipped because frames there are not adjacent on the path.
    Returns ``(slope, correlation)`` or ``(None, None)`` if undetermined.
    """

    n = energies.shape[0]
    d_e: list[float] = []
    work: list[float] = []
    for i in range(n - 1):
        if boundary is not None and i + 1 == boundary:
            continue  # frame boundary-1 -> boundary is the jump back to the TS
        dx = coordinates[i + 1] - coordinates[i]
        w = -0.5 * float(np.sum((forces[i] + forces[i + 1]) * dx))
        d_e.append(float(energies[i + 1] - energies[i]))
        work.append(w)
    de, w = np.asarray(d_e), np.asarray(work)
    denom = float(np.dot(w, w))
    if w.size < 2 or denom <= 0.0 or not np.isfinite(denom):
        return None, None
    slope = float(np.dot(de, w) / denom)
    corr = float(np.corrcoef(de, w)[0, 1]) if np.std(de) > 0 and np.std(w) > 0 else None
    return slope, corr


# --------------------------------------------------------------------------
# Per-record analysis
# --------------------------------------------------------------------------


def analyse_irc(
    record_id: str,
    irc: IrcInput | None,
    *,
    ts_atomic_numbers: Sequence[int],
    ts_coordinates: np.ndarray,
    ts_energy_hartree: float,
    formed: Sequence[tuple[int, int]],
    broken: Sequence[tuple[int, int]],
) -> IrcResult:
    out = IrcResult(record_id=record_id)
    if irc is None:
        out.reasons.append("irc_record_absent")
        return out
    out.present = True
    z = tuple(int(v) for v in irc.atomic_numbers)
    c, e, f = irc.coordinates, irc.energies, irc.forces
    n_atoms = len(z)
    steps = int(e.shape[0]) if e.ndim == 1 else 0
    out.n_frames = steps
    if (
        c.ndim != 3
        or c.shape != (steps, n_atoms, 3)
        or f.shape != c.shape
        or steps < MIN_FRAMES
        or not (np.isfinite(c).all() and np.isfinite(e).all() and np.isfinite(f).all())
    ):
        out.reasons.append("irc_arrays_invalid")
        return out
    if z != tuple(int(v) for v in ts_atomic_numbers):
        out.reasons.append("irc_atom_order_differs_from_ts")
        return out
    out.identity_ok = True

    dev = aligned_max_deviation(c[0], ts_coordinates)
    out.ts_coord_max_dev = dev
    # IRC energies are stored in eV (per the upstream example); compare in Hartree.
    out.ts_energy_dev_hartree = abs(float(e[0]) / HARTREE_TO_EV - float(ts_energy_hartree))
    out.ts_frame_matches = dev <= TS_COORD_TOLERANCE and out.ts_energy_dev_hartree <= TS_ENERGY_TOLERANCE_HARTREE
    if not out.ts_frame_matches:
        out.reasons.append("irc_ts_frame_differs_from_main_hdf5")

    boundary = find_branch_boundary(e)
    out.branch_boundary = boundary
    out.work_slope, out.work_correlation = work_identity(c, e, f, boundary=boundary)
    if boundary is None or boundary < 2 or steps - boundary < 1:
        out.reasons.append("irc_second_branch_not_found")
        return out
    out.continuity_ok = _continuity(c, boundary)
    if not out.continuity_ok:
        out.reasons.append("irc_branch_not_continuous_with_ts")
        out.orientation = ORIENT_NO_BRANCHES
        return out
    end_a, end_b = boundary - 1, steps - 1
    verdict = orient_branches(z, c[end_a], c[end_b], formed, broken)
    out.orientation, reactant_branch = verdict["orientation"], verdict["reactant_branch"]
    out.strict_bond_state_match = verdict["strict"]
    out.edges_correct, out.edges_wrong, out.edges_flat = verdict["correct"], verdict["wrong"], verdict["flat"]
    if reactant_branch == "A":
        out.reactant_end_frame, out.product_end_frame = end_a, end_b
    elif reactant_branch == "B":
        out.reactant_end_frame, out.product_end_frame = end_b, end_a
    if out.orientation != ORIENT_CONNECTS:
        out.reasons.append(f"irc_{out.orientation}")
    return out


def result_row(result: IrcResult) -> dict[str, Any]:
    return {
        "record_id": result.record_id,
        "irc_present": result.present,
        "irc_identity_ok": result.identity_ok,
        "irc_ts_frame_matches": result.ts_frame_matches,
        "irc_n_frames": result.n_frames,
        "irc_branch_boundary": result.branch_boundary,
        "irc_continuity_ok": result.continuity_ok,
        "irc_orientation": result.orientation,
        "irc_reactant_end_frame": result.reactant_end_frame,
        "irc_product_end_frame": result.product_end_frame,
        "irc_strict_bond_state_match": result.strict_bond_state_match,
        "irc_edges_correct": result.edges_correct,
        "irc_edges_wrong": result.edges_wrong,
        "irc_edges_flat": result.edges_flat,
        "irc_ts_coord_max_dev": result.ts_coord_max_dev,
        "irc_ts_energy_dev_hartree": result.ts_energy_dev_hartree,
        "irc_work_slope": result.work_slope,
        "irc_work_correlation": result.work_correlation,
        "irc_energy_force_ok": result.energy_force_ok,
        "irc_path_pairing_ok": result.path_pairing_ok,
        "irc_reasons": list(result.reasons),
        # Geometry origin must travel with any label built from these frames.
        "reactant_geometry_origin": "irc_end_frame" if result.reactant_end_frame is not None else None,
    }


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"n": 0, "p05": None, "p50": None, "p95": None}
    arr = np.sort(np.asarray(values, dtype=float))
    pick = lambda q: float(arr[min(arr.size - 1, int(q * arr.size))])  # noqa: E731
    return {"n": int(arr.size), "p05": pick(0.05), "p50": pick(0.5), "p95": pick(0.95)}


def build_irc_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    groups: Mapping[str, str],
    event_ok_ids: set[str],
    base_tasks: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    """Aggregate IRC rows.

    ``event_ok_ids`` are records that already carry a trusted event label from
    the main funnel; ``base_tasks`` are that funnel's counts, kept alongside so
    the IRC gain is read against the right baseline.
    """

    def counts(pick) -> dict[str, int]:
        chosen = [r for r in rows if pick(r)]
        return {
            "records": len(chosen),
            "parent_groups": len({groups[r["record_id"]] for r in chosen if r["record_id"] in groups}),
        }

    slopes = [r["irc_work_slope"] for r in rows if r["irc_work_slope"] is not None and r["irc_identity_ok"]]
    median_slope = float(np.median(slopes)) if slopes else None
    lo, hi = UNIT_SLOPE_WINDOW
    units_consistent = median_slope is not None and lo <= median_slope <= hi
    in_window = sum(1 for s in slopes if lo <= s <= hi)

    ef = counts(lambda r: r["irc_energy_force_ok"])
    ef_frames = int(sum(r["irc_n_frames"] for r in rows if r["irc_energy_force_ok"]))
    pairing = counts(lambda r: r["irc_path_pairing_ok"])
    path_paired_joint = counts(lambda r: r["irc_path_pairing_ok"] and r["record_id"] in event_ok_ids)
    orientation = {}
    for r in rows:
        if r["irc_energy_force_ok"]:
            orientation[r["irc_orientation"]] = orientation.get(r["irc_orientation"], 0) + 1
    reason_counts: dict[str, int] = {}
    for r in rows:
        for reason in r["irc_reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return {
        "schema": IRC_SCHEMA,
        "records_examined": len(rows),
        "irc_present": counts(lambda r: r["irc_present"]),
        "irc_identity_ok": counts(lambda r: r["irc_identity_ok"]),
        "energy_force": {
            **ef,
            "frames": ef_frames,
            "unit_consistency": {
                "criterion": "median least-squares slope of dE vs trapezoid force work within window",
                "window": list(UNIT_SLOPE_WINDOW),
                "median_slope": median_slope,
                "records_in_window": in_window,
                "records_with_slope": len(slopes),
                "mutually_consistent": bool(units_consistent),
            },
        },
        "ts_frame_agreement_with_main_hdf5": {
            "coordinate_max_dev_angstrom": _quantiles([r["irc_ts_coord_max_dev"] for r in rows if r["irc_ts_coord_max_dev"] is not None]),
            "energy_dev_hartree": _quantiles([r["irc_ts_energy_dev_hartree"] for r in rows if r["irc_ts_energy_dev_hartree"] is not None]),
        },
        "orientation_among_energy_force_ok": orientation,
        "step_pairing_by_irc": {
            "definition": "every formed/broken sigma bond named by the event changes length the stated way between the two IRC ends (none the wrong way, at least half clearly), so exactly one end is the reactant",
            **pairing,
            "strict_tier": {
                "definition": "additionally both ends reproduce the absolute bonded/non-bonded state of every named edge",
                **counts(lambda r: r["irc_path_pairing_ok"] and r["irc_strict_bond_state_match"]),
            },
        },
        "path_paired_joint": {
            "definition": "event_only (main funnel) AND step pairing by IRC; reactant/product geometry taken from IRC end frames in global atom order",
            "reactant_geometry_origin": "irc_end_frame",
            **path_paired_joint,
        },
        "baseline_main_funnel": dict(base_tasks),
        "irc_reasons": dict(sorted(reason_counts.items(), key=lambda kv: -kv[1])),
        "claim_limits": [
            "IRC end frames are path-derived endpoints, not independently optimised minima or independent reactant seeds.",
            "Pairing evidence is geometric and restricted to the sigma bonds named by the event: edge lengths at the two IRC ends (directional tier) and a covalent-radius bonded/non-bonded test (strict tier).",
            "Unit consistency is measured from the work identity; absolute units are taken from the dataset documentation (eV, eV/angstrom) and cross-checked against the main HDF5 TS energy.",
            "This is development-level evidence; no record is promoted to a Track-B split here.",
        ],
    }
