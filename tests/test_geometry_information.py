import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from xtbflow.v1.geometry_information import (commit_index,event_table,first_informative,
                                             predicted_endpoints,ranking_metrics,
                                             within_parent_rho,xtb_energy_kcal)

SCRIPT=Path(__file__).resolve().parents[1]/'scripts'/'explore_geometry_information.py'


def load_script():
    spec=importlib.util.spec_from_file_location('explore_geometry_information',SCRIPT)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_ranking_metrics_orders_ties_and_degenerate_parents():
    m=ranking_metrics([1.,5.,9.],[10.,20.,30.])
    assert m['rho']==pytest.approx(1.) and m['top1']==1. and m['concordance']==1.
    m=ranking_metrics([1.,5.,9.],[30.,20.,10.])
    assert m['rho']==pytest.approx(-1.) and m['top1']==0. and m['concordance']==0.
    # Tied lowest prediction shares the top-1 hit; tied pairs count one half.
    m=ranking_metrics([1.,5.,9.],[2.,2.,7.])
    assert m['top1']==.5 and m['concordance']==pytest.approx((.5+1+1)/3)
    assert ranking_metrics([3.,3.],[1.,2.]) is None
    assert ranking_metrics([1.,4.],[7.,7.])['rho']==0.
    # Pairs closer than the gap are not scored.
    assert ranking_metrics([1.,2.],[2.,1.])['concordance'] is None


def test_event_table_targets_channel_minimum_and_pools_predictions():
    rows=[dict(channel_id='a',catalog_barrier_kcal=10.,p=3.),dict(channel_id='a',catalog_barrier_kcal=12.,p=1.),
          dict(channel_id='b',catalog_barrier_kcal=5.,p=4.)]
    ch,t,p=event_table(rows,'p','min')
    assert ch==['a','b'] and t.tolist()==[10.,5.] and p.tolist()==[1.,4.]
    assert event_table(rows,'p','mean')[2].tolist()==[2.,4.]


def test_predicted_endpoints_recover_linear_flow_target():
    clock=np.linspace(0,1,11);x0=torch.randn(4,3);x1=torch.randn(4,3)
    trace=torch.stack([(1-t)*x0+t*x1 for t in clock])
    hat=predicted_endpoints(trace,clock)
    assert torch.allclose(hat,x1.expand_as(hat),atol=1e-5)
    with pytest.raises(ValueError):predicted_endpoints(trace,clock[:-1])


def test_commit_index():
    assert commit_index(['a','b','a','a']) == 2
    assert commit_index(['a','a','a']) == 0
    assert commit_index(['a',None,'b']) == 2
    assert commit_index(['a','a',None]) is None


def test_within_parent_rho_and_first_informative():
    pred=[1,2,3,4,1,1,1,1,5,6];target=[1,2,3,4,9,8,7,6,1,2];parents=list('aaaabbbbcc')
    rho=within_parent_rho(pred,target,parents)
    assert rho['a']==pytest.approx(1.) and rho['b']==0. and 'c' not in rho
    assert within_parent_rho([1,2,3,4],[5,5,5,5],list('aaaa'))=={}
    assert first_informative([0,.5,1],[0.,.3,.5],.1)==.5
    assert first_informative([0,.5,1],[0.,.3,.5],-.1) is None


def test_xtb_skips_collapsed_geometry_without_calling_tblite():
    cfg=dict(min_distance_angstrom=.5,charge=0,uhf=0,accuracy=1.,max_iterations=250,electronic_temperature_kelvin=300.)
    e,status=xtb_energy_kcal([1,1],[[0,0,0],[0,0,.1]],cfg)
    assert np.isnan(e) and status=='collapsed'
    assert xtb_energy_kcal([1,1],[[0,0,0],[0,0,np.nan]],cfg)[1]=='nonfinite'


def synthetic_a(rng):
    rows=[]
    for p in range(6):
        for c in range(3):
            barrier=10.+7*c+rng.normal()
            for r in range(2):
                rows.append(dict(parent_id=f'p{p}',formula=f'f{p%3}',channel_id=f'c{c}',
                                 catalog_barrier_kcal=barrier+r,oracle=barrier+r,E_frozen=barrier+rng.normal(0,5),
                                 X_frozen=barrier+rng.normal(0,1),N_clean=rng.normal(),G_clean=barrier,
                                 xtb_ts=barrier*1.2,xtb_status='ok'))
    return rows


def test_analysis_a_on_synthetic_rows():
    s=load_script();cfg=s.json.loads((SCRIPT.parents[1]/'configs'/'explore'/'geometry_information.json').read_text())
    out=s.analyze_a(synthetic_a(np.random.default_rng(0)),cfg['check_a_information_bound'])
    assert out['min']['predictors']['oracle']['rho']['estimate']==pytest.approx(1.)
    assert out['min']['comparisons']['G_clean-N_clean']['rho']['estimate']>0
    assert set(out['reading'])=={'X_frozen-E_frozen','G_clean-N_clean','xtb_ts-E_frozen','xtb_ts-X_frozen'}


def test_analysis_b_on_synthetic_trajectories():
    s=load_script();cfg=s.json.loads((SCRIPT.parents[1]/'configs'/'explore'/'geometry_information.json').read_text())
    rng=np.random.default_rng(1);rows=[]
    for p in range(6):
        for j in range(8):
            barrier=float(rng.normal(30,8));step=int(rng.integers(5,40))
            events=['x']*step+[f'e{j%3}']*(51-step)
            noise=np.linspace(20,1,11)
            rows.append(dict(parent_id=f'p{p}',formula=f'f{p%3}',commit_step=commit_index(events),events=events,
                             proxy_status='PROXY_MATCH',hits_best_event=j%3==0,event_barrier_kcal=barrier,
                             xtb_delta_kcal=(barrier+rng.normal(0,1,11)*noise).tolist(),xtb_status=['ok']*11,
                             hx_mean_kcal=(barrier+rng.normal(0,3,11)).tolist()))
    out=s.analyze_b(rows,cfg['check_b_time_window'])
    assert out['n_valid_final']==48 and 0<out['commit']['median_t']<1
    assert out['signals']['hX_vs_final_event_barrier']['t_geo']==0.
    assert out['signals']['xtb_vs_final_xtb']['mean'][-1]==pytest.approx(1.)
    assert out['signals']['xtb_vs_final_event_barrier']['reading'] in ('WINDOW_EXISTS','NO_USABLE_WINDOW')
