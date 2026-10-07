import copy
from pathlib import Path
import runpy

import pytest


audit_ledger=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/v1a_stream_evaluate.py'))['audit_ledger']


def test_reject_cost_tampering_and_missing_attempts():
    row=dict(operations=[dict(operation='f',count=50,cost_units=50,cumulative_units=50)],
             spent_units=50,cap_units=200,n_proposals=1,
             attempts=[dict(attempt_id='a',candidate_id='c',completion_units=50)],
             candidates=[dict(candidate_id='c',completion_units=50)])
    audit_ledger(row,{'f':1})
    for key in ('cost_units','cumulative_units'):
        bad=copy.deepcopy(row);bad['operations'][0][key]=49
        with pytest.raises(ValueError,match='ledger'):audit_ledger(bad,{'f':1})
    bad=copy.deepcopy(row);bad['attempts']=[]
    with pytest.raises(ValueError,match='missing'):audit_ledger(bad,{'f':1})
    bad=copy.deepcopy(row);bad['cap_units']=49
    with pytest.raises(ValueError,match='overspend'):audit_ledger(bad,{'f':1})
