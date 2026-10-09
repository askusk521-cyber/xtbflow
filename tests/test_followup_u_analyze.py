import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('u_analyze',Path(__file__).resolve().parents[1]/'scripts/followup_u_analyze.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_reading_boundaries():
    assert m.reading(.7,.7)=='U_IS_REACTION_MODE'
    assert m.reading(.299,.8)=='U_NOT_REACTION_MODE'
    assert m.reading(.9,.499)=='U_NOT_REACTION_MODE'
    assert m.reading(.3,.5)=='U_PARTIAL'


def test_wilson_and_record_gate():
    lo,hi=m.wilson(50,100)
    assert lo==pytest.approx(.4038315303659956)
    assert hi==pytest.approx(.5961684696340044)
    with pytest.raises(ValueError):m.summarize([])


def test_parent_average_not_record_weighted():
    rows=[]
    for i in range(558):
        parent='a' if i<500 else 'b'
        rows.append(dict(reference_id=str(i),parent_id=parent,split_group=parent,status='ok',gradient_calls=6,
                         metrics=dict(event_size=2,overlap=.2 if parent=='a' else .8,c_u=-1,
                                      imaginary_count=1,best_neg_overlap=.8)))
    out=m.summarize(rows)
    assert out['cluster_intervals']['overlap']['estimate']==pytest.approx(.5)
    assert out['negative_c_u_count']==558
    assert out['reading']=='U_NOT_REACTION_MODE'
    assert out['gradient_calls']==3348
