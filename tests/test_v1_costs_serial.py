import pytest
from xtbflow.v1.costs import Ledger,prefixes
from xtbflow.v1.serial_search import promote,schedule


def test_failed_and_pruned_work_counts_before_completions():
    ledger=Ledger({'g':.6,'h':.4,'X':.2,'X_back':.3},cap=200)
    for i in range(4):
        assert ledger.charge(f'p{i}','g',50)
    assert ledger.spent==120
    # Invalid p0 retains its proposal cost; p1 is pruned after partial geometry.
    assert ledger.charge('p1','h',20)
    assert ledger.charge('p2','h',50)
    a=ledger.complete('p2','c2','VALID')
    assert ledger.charge('p3','h',50)
    b=ledger.complete('p3','c3','DECODE_FAILED')
    assert a['completion_units']==148 and b['completion_units']==168
    assert sum(r['cost_units'] for r in ledger.operations)==ledger.spent
    assert len(prefixes([a,b])['4'])==2
    assert not ledger.charge('tail','g',100)
    assert ledger.spent==168


def test_step_budget_no_tail_overspend_and_gradient_charges():
    ledger=Ledger({'f':1.,'X':.2,'X_back':.4},cap=5.6)
    assert ledger.step('p','f',True)
    assert ledger.step('p','f',True)
    assert ledger.spent==pytest.approx(5.6)
    assert not ledger.step('p','f',True)
    assert len(ledger.operations)==6


def test_promotion_is_distinct_deterministic_and_independent_of_input_order():
    ids=list(range(8));scores={i:float(i) for i in ids}
    keys=dict(query='q',training_seed=0,round_id=1,stage=0)
    a=promote(ids,scores,4,1,**keys)
    assert len(a)==5 and set(range(4))<=set(a)
    assert a==promote(ids[::-1],scores,4,1,**keys)
    assert schedule(0)['proposals']==4 and schedule(1)['proposals']==8
    assert promote([1],{1:None},4,1,**keys)==[1]
