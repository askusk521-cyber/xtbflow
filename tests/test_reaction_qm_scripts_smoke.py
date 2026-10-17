"""End-to-end smoke test of the two Reaction-QM funnel scripts on a tiny,
synthetic HDF5 pair that mimics the real layout (chunk groups in the main file,
flat records in the IRC file).  It guards the file-reading paths; chemistry is
covered by the pure-function tests.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("rdkit")

ROOT = Path(__file__).resolve().parents[1]
K, E0_EV, HARTREE_TO_EV = 10.0, -100.0, 27.211386245988


def _species(group, name, smiles, z, coords, charge=0, mult=1, ehg=(-1.0, -0.9, -0.95)):
    sp = group.create_group(name)
    sp.create_dataset("smiles", data=smiles, dtype=h5py.string_dtype())
    sp.create_dataset("atomic_numbers", data=np.asarray(z, dtype=np.int8))
    sp.create_dataset("charge", data=np.int8(charge))
    sp.create_dataset("multiplicity", data=np.int8(mult))
    sp.create_dataset("EHG", data=np.asarray(ehg, dtype=np.float64))
    sp.create_dataset("coordinates", data=np.asarray(coords, dtype=np.float32))


def _frame(x_h):
    return np.array([[x_h, 0.0, 0.0], [-1.3, 0.0, 0.0], [1.3, 0.0, 0.0]])


def _write_main(path: Path) -> None:
    ts_smiles = "[H:1][C:2]#[N:3]>>[C:2]#[N:3][H:1]"
    with h5py.File(path, "w") as f:
        chunk = f.create_group("B3LYPD3_TZVP_1_10000")
        for rid in ("RXN_0000000001", "RXN_0000000002"):
            g = chunk.create_group(rid)
            _species(g, "R0", "[H:1][C:2]#[N:3]", [1, 6, 7], _frame(-0.2))
            _species(g, "P0", "[C:1]#[N:2][H:3]", [6, 7, 1], _frame(0.2)[[1, 2, 0]])
            _species(g, "TS", ts_smiles, [1, 6, 7], _frame(0.0), ehg=(E0_EV / HARTREE_TO_EV, -0.9, -0.95))
        # An out-of-scope (chlorine) record that must stop at the scope gate.
        g = chunk.create_group("RXN_0000000003")
        _species(g, "R0", "[Cl:1][H:2]", [17, 1], np.zeros((2, 3)))
        _species(g, "P0", "[H:1][Cl:2]", [1, 17], np.zeros((2, 3)))
        _species(g, "TS", "[Cl:1][H:2]>>[H:2][Cl:1]", [17, 1], np.zeros((2, 3)))


def _write_irc(path: Path) -> None:
    xs = [0.0, -0.04, -0.08, -0.12, -0.16, -0.20, 0.04, 0.08, 0.12, 0.16, 0.20]
    with h5py.File(path, "w") as f:
        g = f.create_group("RXN_0000000001")  # RXN_0000000002 deliberately absent
        g.create_dataset("atomic_numbers", data=np.array([1, 6, 7], dtype=np.int8))
        g.create_dataset("coordinates", data=np.array([_frame(x) for x in xs], dtype=np.float32))
        g.create_dataset("energies", data=np.array([E0_EV - 0.5 * K * x * x for x in xs]))
        forces = np.zeros((len(xs), 3, 3))
        forces[:, 0, 0] = [K * x for x in xs]
        g.create_dataset("forces", data=forces)


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}{os.pathsep}{ROOT / 'vendor' / 'mechai_reusable'}")
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], env=env, capture_output=True, text=True)


def test_funnel_and_irc_scripts_run_end_to_end_on_a_synthetic_pair(tmp_path):
    main_h5, irc_h5 = tmp_path / "main.h5", tmp_path / "irc.h5"
    _write_main(main_h5)
    _write_irc(irc_h5)

    funnel_dir = tmp_path / "funnel"
    done = _run("build_reaction_qm_task_funnel.py", "--h5", str(main_h5), "--output-dir", str(funnel_dir), "--workers", "2")
    assert done.returncode == 0, done.stderr
    report = json.loads((funnel_dir / "funnel_report.json").read_text(encoding="utf-8"))
    stages = {s["stage"]: s["records"] for s in report["funnel"]}
    assert stages == {"source_records": 3, "readable": 3, "in_scope": 2, "identity_verifiable": 2}
    assert report["tasks"]["paired_joint"]["records"] == 2
    assert report["tasks"]["energy_force"]["records"] == 0
    assert report["exclusion_reasons"]["out_of_scope_element"]["first_blocking_records"] == 1

    # Refuses to overwrite a populated output directory.
    again = _run("build_reaction_qm_task_funnel.py", "--h5", str(main_h5), "--output-dir", str(funnel_dir))
    assert again.returncode != 0

    irc_dir = tmp_path / "irc"
    done = _run(
        "build_reaction_qm_irc_evidence.py", "--h5", str(main_h5), "--irc-h5", str(irc_h5),
        "--funnel-manifest", str(funnel_dir / "manifest.jsonl"), "--output-dir", str(irc_dir), "--workers", "2",
    )
    assert done.returncode == 0, done.stderr
    irc_report = json.loads((irc_dir / "irc_report.json").read_text(encoding="utf-8"))
    assert irc_report["records_examined"] == 2
    assert irc_report["irc_present"]["records"] == 1
    assert irc_report["energy_force"]["records"] == 1
    assert irc_report["energy_force"]["frames"] == 11
    assert irc_report["energy_force"]["unit_consistency"]["mutually_consistent"] is True
    assert irc_report["step_pairing_by_irc"]["records"] == 1
    assert irc_report["path_paired_joint"]["records"] == 1
    assert irc_report["irc_reasons"] == {"irc_record_absent": 1}
    assert irc_report["baseline_main_funnel"]["paired_joint"]["records"] == 2
    ids = (irc_dir / "task_ids" / "path_paired_joint.txt").read_text(encoding="utf-8").split()
    assert ids == ["RXN_0000000001"]
