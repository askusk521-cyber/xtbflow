import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('smc_analysis',Path(__file__).resolve().parents[1]/'scripts/followup_smc_analyze.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_frozen_readings_all_cases():
    assert m.reading([.1,.2],[.01,.3])=='IN_GENERATION_SELECTION_HELPS'
    assert m.reading([.1,.2],[-.1,.3])=='NOT_PHYSICS_SPECIFIC'
    assert m.reading([0.,.2],[.1,.3])=='NO_GAIN_OVER_POSTHOC'
    assert m.reading([-.2,-.1],[.1,.3])=='SELECTION_HURTS'
    assert m.reading([.1,.2],[-.3,-.1])=='UNCLASSIFIED_BY_FROZEN_TABLE'


def test_closure_recommendation_boundaries():
    assert m.recommendation({'aimnet_0.7':'NO_GAIN_OVER_POSTHOC'})=='LINE_CLOSURE_RECOMMENDED'
    assert m.recommendation({'aimnet_0.7':'SELECTION_HURTS','gxtb_0.9':'IN_GENERATION_SELECTION_HELPS'})=='NEEDS_HELDOUT_CONFIRMATION'
    assert m.recommendation({'aimnet_0.7':'NOT_PHYSICS_SPECIFIC'})=='NO_FROZEN_RECOMMENDATION'
