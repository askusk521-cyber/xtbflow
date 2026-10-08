import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('review_analysis',Path(__file__).parents[1]/'scripts/review_open_world_analyze.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_wilson_boundaries():
    assert m.wilson(0,0) is None
    assert abs(m.wilson(0,20)['ci_two95'][0])<1e-12
    assert abs(m.wilson(20,20)['ci_two95'][1]-1)<1e-12
    interval=m.wilson(17,20)
    assert interval['rate']==.85
    assert interval['ci_two95'][0]<.85<interval['ci_two95'][1]
