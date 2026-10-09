import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('analysis',Path(__file__).parents[1]/'scripts/followup_thermo_analyze.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_frozen_sort_orders():
    channels=['a','b','c','d']
    scores=m.score_orders(channels,np.array([3.,0.,2.,1.]),[2.,None,5.,1.],2.)
    assert np.argsort(scores['S_N2']).tolist()==[1,3,2,0]
    assert np.argsort(scores['S_thermo']).tolist()==[3,0,2,1]
    assert np.argsort(scores['S_N_thermo']).tolist()==[3,0,1,2]


def test_tie_is_channel_sha():
    channels=['a','b','c']
    scores=m.score_orders(channels,np.zeros(3),[1.,1.,1.],1.)
    expected=sorted(range(3),key=lambda i:m.sha256(channels[i].encode()).hexdigest())
    assert all(np.argsort(rank).tolist()==expected for rank in scores.values())


def test_reading_boundaries():
    assert m.reading({'summary':{'ci_two95':[0.,1.]}})=='INCONCLUSIVE'
    assert m.reading({'summary':{'ci_two95':[.1,1.]}})=='ENUMERATION_BEATS_GENERATION'
    assert m.reading({'summary':{'ci_two95':[-1.,-.1]}})=='GENERATION_BEATS_ENUMERATION'
