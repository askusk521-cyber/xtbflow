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
                             bond_change_commit_step=max(0,commit_index(events)-3),
                             proxy_status='PROXY_MATCH',hits_best_event=j%3==0,event_barrier_kcal=barrier,
                             xtb_delta_kcal=(barrier+rng.normal(0,1,11)*noise).tolist(),xtb_status=['ok']*11,
                             hx_mean_kcal=(barrier+rng.normal(0,3,11)).tolist()))
    out=s.analyze_b(rows,cfg['check_b_time_window'])
    assert out['n_valid_final']==48 and 0<out['commit']['median_t']<1
    assert out['commit_bond_change_sensitivity']['median_t']<out['commit']['median_t']
    assert out['self_correction']['changed_after_half']>=0
    assert out['signals']['hX_vs_final_event_barrier']['t_geo']==0.
    assert out['signals']['xtb_vs_final_xtb']['mean'][-1]==pytest.approx(1.)
    assert out['signals']['xtb_vs_final_event_barrier']['reading'] in ('WINDOW_EXISTS','NO_USABLE_WINDOW')


def synthetic_runs(rng,shift):
    rows=[]
    for p in range(6):
        for seed in range(2):
            for j in range(8):
                barrier=float(rng.normal(30,8));step=int(rng.integers(5,40))+shift
                events=['x']*step+[f'e{j%3}']*(51-step)
                noise=np.linspace(20,1,11)
                rows.append(dict(parent_id=f'p{p}',formula=f'f{p%3}',seed=seed,proposal=j,events=events,
                                 commit_step=commit_index(events),bond_change_commit_step=commit_index(events),
                                 proxy_status='PROXY_MATCH' if j%2 else 'OUTSIDE_CATALOGUE_EVENT',
                                 hits_best_event=j%3==0,hits_best_reference=j%4==0,event_barrier_kcal=barrier,
                                 xtb_delta_kcal=(barrier+rng.normal(0,1,11)*noise).tolist(),xtb_status=['ok']*11,
                                 hx_mean_kcal=(barrier+rng.normal(0,3,11)).tolist()))
    return rows


def test_analyze_clock_reads_window_quality_and_reproduction(tmp_path):
    import json
    s=load_script();rng=np.random.default_rng(2)
    sync=synthetic_runs(rng,0);lead=synthetic_runs(rng,8)
    files={}
    for name,rows in (('sync',sync),('geometry_lead2',lead)):
        files[name]=tmp_path/f'{name}.jsonl'
        files[name].write_text(''.join(json.dumps(r)+'\n' for r in rows))
    out=tmp_path/'clock.json'
    cfg=SCRIPT.parents[1]/'configs'/'explore'/'geometry_lead.json'
    import argparse,os
    cwd=os.getcwd();os.chdir(SCRIPT.parents[1])
    try:
        s.cmd_analyze_clock(argparse.Namespace(config=cfg,run=[[k,str(v)] for k,v in files.items()],
                                               reference_sync=files['sync'],out=out))
    finally:os.chdir(cwd)
    res=json.loads(out.read_text())
    assert res['reproduction_vs_check_b']=={'n':96,'final_event':1.,'proxy_status':1.,'commit_step':1.}
    lead=res['paths']['geometry_lead2']
    assert lead['reading'] in ('GEOMETRY_LEAD_OPENS_WINDOW','WINDOW_WITH_QUALITY_LOSS','NO_USABLE_WINDOW')
    assert set(lead['quality_vs_sync'])=={'valid','proxy_match','best_event','best_reference'}
    assert res['paths']['sync']['time_window']['commit']['median_t']<lead['time_window']['commit']['median_t']


def test_analyze_physics_pairs_arms_and_reads():
    s=load_script();cfg=s.json.loads((SCRIPT.parents[1]/'configs'/'explore'/'physics_guidance.json').read_text())
    rng=np.random.default_rng(3)

    def arm(bump,changed):
        rows=[]
        for p in range(8):
            for seed in range(2):
                for j in range(6):
                    u=float(np.clip(.3+bump*(j%2)+rng.normal(0,.02),0,1))
                    ev='e1' if (changed and j%2) else 'e0'
                    rows.append(dict(parent_id=f'p{p}',formula=f'f{p%4}',seed=seed,proposal=j,events=[None,ev],
                                     event_utility=u,hits_best_event=u>.5,hits_best_reference=False,
                                     proxy_status='PROXY_MATCH',event_barrier_kcal=30.-10*u,
                                     guidance=dict(applied_steps=5,displacement_rms_sum=.1,force_failures=0,
                                                   mean_abs_cos_u_force=.4)))
        return rows
    out=s.analyze_physics({'none':arm(0,False),'saddle_0.3':arm(.3,True),'random_0.3':arm(0,False),
                           'descent_0.3':arm(-.2,True)},cfg)
    assert out['readings']['saddle_0.3']=='PHYSICS_REWRITES_TOWARD_LOW_BARRIER'
    assert out['readings']['descent_0.3']=='PHYSICS_HURTS'
    assert out['arms']['saddle_0.3']['changed_vs_none']['estimate']==pytest.approx(.5)
    assert out['arms']['saddle_0.3']['barrier_shift_vs_none_kcal']['estimate']<0
    with pytest.raises(ValueError):
        s.analyze_physics({'none':arm(0,False),'saddle_0.3':arm(.3,True)[:-1]},cfg)


def test_force_job_rejects_collapsed_geometry_without_tblite():
    s=load_script()
    cfg=dict(min_distance_angstrom=.5,charge=0,uhf=0,accuracy=1.,max_iterations=250,electronic_temperature_kelvin=300.)
    assert s._force_job(([1,1],[[0,0,0],[0,0,.1]],cfg))==(None,'collapsed')
