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
