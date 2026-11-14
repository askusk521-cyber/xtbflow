"""No native QC: test that batch task budgets retain failed work."""
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from qc_campaign_task import Ledger
from gqc_tautomer_pilot import BudgetExceeded


def test_failed_operation_uses_one_slot(tmp_path):
    ledger = Ledger(tmp_path, {"calls": 1, "minutes": 2})
    with pytest.raises(ZeroDivisionError):
        ledger.call("failure", lambda: 1 / 0)
    with pytest.raises(BudgetExceeded):
        ledger.call("extra", lambda: pytest.fail("operation cap ignored"))
    assert ledger.summary()["started"] == 1
    assert ledger.summary()["failed"] == 1
    assert ledger.summary()["unsettled"] == 0


def test_success_is_journaled_separately(tmp_path):
    ledger = Ledger(tmp_path, {"calls": 2, "minutes": 2})
    assert ledger.call("point", lambda: -7.0) == -7.0
    assert ledger.summary()["completed"] == 1
    assert len((tmp_path / "operations.jsonl").read_text().splitlines()) == 2


def test_tautomer_graph_preserves_atoms_and_moves_one_explicit_proton():
    from prepare_qc_paths import tautomer_graph, canonical
    r, p, (donor, acceptor, h) = tautomer_graph()
    assert canonical(r) == "C=CCC=O"
    assert canonical(p) == "C=CC=CO"
    assert r.GetBondBetweenAtoms(donor, h) is not None
    assert p.GetBondBetweenAtoms(donor, h) is None
    assert p.GetBondBetweenAtoms(acceptor, h) is not None
    assert [a.GetAtomMapNum() for a in r.GetAtoms()] == [a.GetAtomMapNum() for a in p.GetAtoms()]


def test_da_opening_and_michael_conserve_composition_and_mapping():
    from prepare_qc_paths import da_graph, opening_graph, michael_graph
    from rdkit.Chem import rdMolDescriptors
    r, p, alkene = da_graph("Cc1ccc(N)o1", "O=C1C=CC(=O)N1")
    michael = michael_graph(r, alkene)
    assert len({rdMolDescriptors.CalcMolFormula(m) for m in (r, p, michael)}) == 1
    assert [a.GetAtomMapNum() for a in r.GetAtoms()] == [a.GetAtomMapNum() for a in michael.GetAtoms()]
    r, adduct, _ = da_graph("Cc1ccc(O)o1", "C=C")
    opened, (donor, acceptor, h) = opening_graph(adduct)
    assert rdMolDescriptors.CalcMolFormula(adduct) == rdMolDescriptors.CalcMolFormula(opened)
    assert opened.GetBondBetweenAtoms(donor, h) is None
    assert opened.GetBondBetweenAtoms(acceptor, h) is not None


def test_missing_numerical_gate_blocks_without_quantum_work(tmp_path):
    from qc_campaign_task import numerical_gate
    report = {}
    assert not numerical_gate({"gate": "wave1/missing"}, {}, tmp_path / "results-wave2" / "opt", report)
    assert report["status"] == "blocked_numerical_gate"


def test_upstream_manifest_mismatch_cannot_unlock_path(tmp_path):
    import json
    from qc_campaign_paths import load_upstream
    upstream = tmp_path / "opt_input"
    upstream.mkdir()
    (upstream / "report.json").write_text(json.dumps({"manifest_sha256": "other", "status": "optimized_identity_passed"}))
    out = tmp_path / "path"
    out.mkdir()
    ledger = Ledger(out, {"calls": 5, "minutes": 3})
    with pytest.raises(ValueError, match="manifest"):
        load_upstream({"_manifest_sha256": "expected"}, ledger, "opt_input", {"optimized_identity_passed"})
    assert ledger.summary()["started"] == 0
