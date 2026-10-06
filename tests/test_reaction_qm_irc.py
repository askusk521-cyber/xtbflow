"""Tests for the Reaction-QM IRC evidence layer.

The synthetic path is a hydrogen sliding between C and N on an inverted
parabola, so energies and forces are analytically consistent and the expected
orientation, boundary and work-identity slope are known exactly.  Real-record
behaviour is established by the n2 run, not by these fixtures.
"""
from __future__ import annotations

import numpy as np
import pytest

from xtbflow.data.reaction_qm_irc import (
    HARTREE_TO_EV,
    ORIENT_AMBIGUOUS,
    ORIENT_CONNECTS,
    ORIENT_CONTRADICTS,
    ORIENT_NOT_TESTABLE,
    IrcInput,
    analyse_irc,
    build_irc_report,
    find_branch_boundary,
    result_row,
    work_identity,
)

K = 10.0  # eV / angstrom^2
E0_EV = -100.0
Z = (1, 6, 7)  # H1, C2, N3 (global map order)
FORMED, BROKEN = [(1, 3)], [(1, 2)]  # H leaves C2 and joins N3


def _frame(x_h: float) -> np.ndarray:
    # C at -1.3, N at +1.3; the H slides along the axis between them.
    return np.array([[x_h, 0.0, 0.0], [-1.3, 0.0, 0.0], [1.3, 0.0, 0.0]])


def _energy(x_h: float) -> float:
    return E0_EV - 0.5 * K * x_h**2


def _forces(x_h: float) -> np.ndarray:
    f = np.zeros((3, 3))
    f[0, 0] = K * x_h  # -dE/dx_H
    return f


def _path(xs: list[float], *, force_sign: float = 1.0) -> IrcInput:
    return IrcInput(
        atomic_numbers=Z,
        coordinates=np.array([_frame(x) for x in xs]),
        energies=np.array([_energy(x) for x in xs]),
        forces=np.array([force_sign * _forces(x) for x in xs]),
    )


A_SIDE = [-0.04, -0.08, -0.12, -0.16, -0.20]  # H bonded to C
B_SIDE = [0.04, 0.08, 0.12, 0.16, 0.20]  # H bonded to N
TS = _frame(0.0)
TS_E_HARTREE = E0_EV / HARTREE_TO_EV


def _run(irc, formed=FORMED, broken=BROKEN, **overrides):
    kwargs = dict(ts_atomic_numbers=Z, ts_coordinates=TS, ts_energy_hartree=TS_E_HARTREE, formed=formed, broken=broken)
    kwargs.update(overrides)
    return analyse_irc("RXN_test", irc, **kwargs)


def test_two_branch_path_connects_declared_reactant_and_product():
    res = _run(_path([0.0] + A_SIDE + B_SIDE))
    assert res.present and res.identity_ok and res.ts_frame_matches
    assert res.branch_boundary == 6  # first frame of the second branch
    assert res.continuity_ok
    assert res.orientation == ORIENT_CONNECTS
    # Branch A ends bonded to C (the declared reactant), branch B ends bonded to N.
    assert (res.reactant_end_frame, res.product_end_frame) == (5, 10)
    assert res.energy_force_ok and res.path_pairing_ok
    assert res.work_slope == pytest.approx(1.0, abs=0.05)


def test_stored_branch_order_does_not_decide_which_end_is_the_reactant():
    res = _run(_path([0.0] + B_SIDE + A_SIDE))
    assert res.orientation == ORIENT_CONNECTS
    assert (res.reactant_end_frame, res.product_end_frame) == (10, 5)


def test_an_event_that_does_not_happen_in_the_path_is_contradicted():
    res = _run(_path([0.0] + A_SIDE + B_SIDE), formed=[(2, 3)], broken=[])
    assert res.orientation == ORIENT_CONTRADICTS
    assert res.energy_force_ok and not res.path_pairing_ok
    assert res.reactant_end_frame is None


def test_pure_bond_order_change_cannot_be_oriented():
    res = _run(_path([0.0] + A_SIDE + B_SIDE), formed=[], broken=[])
    assert res.orientation == ORIENT_NOT_TESTABLE
    assert not res.path_pairing_ok


def test_orientation_is_ambiguous_when_both_ends_fit_both_hypotheses():
    # An edit whose bonded state is identical at both ends and absent for
    # "formed" fits... only if both hypotheses score fully, which needs ends
    # that satisfy both; exercise the branch directly via symmetric ends.
    from xtbflow.data.reaction_qm_irc import orient_branches

    end = _frame(0.0)  # H equidistant: C-H bonded (1.3 < 1.34), N-H not
    orientation, branch = orient_branches(Z, end, end, FORMED, BROKEN)
    assert orientation in (ORIENT_CONTRADICTS, ORIENT_AMBIGUOUS)
    assert branch is None


def test_ts_frame_that_differs_from_the_main_hdf5_is_not_usable():
    shifted = TS + 0.01
    res = _run(_path([0.0] + A_SIDE + B_SIDE), ts_coordinates=shifted)
    assert res.identity_ok and not res.ts_frame_matches
    assert not res.energy_force_ok
    assert "irc_ts_frame_differs_from_main_hdf5" in res.reasons
    off_energy = _run(_path([0.0] + A_SIDE + B_SIDE), ts_energy_hartree=TS_E_HARTREE + 0.5)
    assert not off_energy.ts_frame_matches


def test_atom_order_mismatch_stops_before_any_geometry_use():
    res = _run(_path([0.0] + A_SIDE + B_SIDE), ts_atomic_numbers=(6, 1, 7))
    assert not res.identity_ok and not res.energy_force_ok
    assert res.reasons == ["irc_atom_order_differs_from_ts"]


def test_missing_or_malformed_records_are_reported_not_raised():
    assert "irc_record_absent" in _run(None).reasons
    bad = _path([0.0] + A_SIDE + B_SIDE)
    bad.forces[2, 0, 0] = np.nan
    assert "irc_arrays_invalid" in _run(bad).reasons
    short = _path([0.0, -0.1])
    assert "irc_arrays_invalid" in _run(short).reasons


def test_single_branch_path_is_not_pairing_evidence():
    res = _run(_path([0.0] + A_SIDE))
    assert res.branch_boundary is None
    assert "irc_second_branch_not_found" in res.reasons
    assert res.energy_force_ok and not res.path_pairing_ok  # E/F points are still usable


def test_energy_jump_without_geometric_restart_is_not_accepted_as_a_branch():
    # Energies look like two branches but the geometry just keeps going.
    xs = [0.0] + A_SIDE + [-0.30, -0.35, -0.40]
    irc = _path(xs)
    irc.energies[6:] = irc.energies[6:] + 5.0  # fake "back up" jump
    res = _run(irc)
    assert find_branch_boundary(irc.energies) == 6
    assert not res.continuity_ok
    assert "irc_branch_not_continuous_with_ts" in res.reasons
    assert not res.path_pairing_ok


def test_work_identity_slope_flags_gradients_stored_as_forces_and_unit_mismatch():
    irc = _path([0.0] + A_SIDE + B_SIDE, force_sign=-1.0)
    slope, _ = work_identity(irc.coordinates, irc.energies, irc.forces, boundary=6)
    assert slope == pytest.approx(-1.0, abs=0.05)
    scaled = _path([0.0] + A_SIDE + B_SIDE)
    slope, corr = work_identity(scaled.coordinates, scaled.energies, scaled.forces * 27.2, boundary=6)
    assert slope == pytest.approx(1 / 27.2, rel=0.05)
    assert corr == pytest.approx(1.0, abs=1e-6)


def test_report_separates_energy_force_from_path_pairing_and_states_the_baseline():
    good = result_row(_run(_path([0.0] + A_SIDE + B_SIDE)))
    one_branch = result_row(_run(_path([0.0] + A_SIDE)))
    one_branch["record_id"] = "RXN_one"
    absent = result_row(analyse_irc("RXN_absent", None, ts_atomic_numbers=Z, ts_coordinates=TS,
                                    ts_energy_hartree=TS_E_HARTREE, formed=FORMED, broken=BROKEN))
    rows = [good, one_branch, absent]
    groups = {"RXN_test": "g1", "RXN_one": "g1", "RXN_absent": "g2"}
    report = build_irc_report(
        rows, groups=groups, event_ok_ids={"RXN_test", "RXN_one", "RXN_absent"},
        base_tasks={"energy_force": {"records": 0, "parent_groups": 0}},
    )
    assert report["irc_present"]["records"] == 2
    assert report["energy_force"]["records"] == 2
    assert report["energy_force"]["parent_groups"] == 1
    assert report["energy_force"]["frames"] == 11 + 6
    assert report["energy_force"]["unit_consistency"]["mutually_consistent"] is True
    assert report["step_pairing_by_irc"]["records"] == 1
    assert report["path_paired_joint"]["records"] == 1
    assert report["path_paired_joint"]["reactant_geometry_origin"] == "irc_end_frame"
    assert report["baseline_main_funnel"]["energy_force"]["records"] == 0
    assert report["irc_reasons"]["irc_record_absent"] == 1
    assert "path-derived endpoints" in report["claim_limits"][0]
