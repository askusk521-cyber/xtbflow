import json

import numpy as np
import pytest

from xtbflow.v1.formal import (AMPLITUDES,ARMS,SEEDS,build_freeze,evaluation_parents,formal_analysis,
                               load_freeze,rarity_stratum,resolve_queries,v1b_export)

N=210


def plan_and_split():
    parents=[f'p{i:03d}' for i in range(N)];groups=[f'F{i//3}' for i in range(N)]
    formulas=sorted(set(groups))
    split=dict(split_hash='h',train=dict(parent_ids=['t0']),development=dict(parent_ids=['d0']),
               screen_reserve=dict(parent_ids=parents+['extra']))
    plan=dict(status='POWER_PLAN_CANDIDATE_PASS',screen_results_used_for_planning=False,split_hash='h',
              planned_n_parents=N,planning_guard=dict(sd_multiplier=1.,icc_increment=0.),
              candidate_inventory=dict(parent_ids=parents,formula_groups=formulas,sizes=[3]*len(formulas)),
              scenarios=[dict(label='target',GO=.81)],nuisance={})
    rows=[dict(query_id=f'{p}_anchor_x',atomic_numbers=[1]) for p in parents+['extra']]
    parent_of={r['query_id']:r['query_id'].split('_anchor')[0] for r in rows}
    return plan,split,rows,parent_of,dict(zip(parents,groups))


def make_freeze(tmp_path):
    plan,split,rows,parent_of,group=plan_and_split()
    freeze,kept=build_freeze(power_plan=plan,split=split,reserve_rows=rows,parent_of=parent_of,
                             config=dict(path='sync',alpha_x=.3,alpha_b=.25,guidance_start=.5,guidance_stop=.95,
                                         a2_times=[.4,.7],cap_units=3200.,t_star=.5),
                             bindings=dict(cost_table_sha256='c'),decision=dict(text='owner decision'))
    path=tmp_path/'freeze.json';path.write_text(json.dumps(freeze))
    q=tmp_path/'screen_queries.jsonl';q.write_text(''.join(json.dumps(r)+'\n' for r in kept))
    return freeze,path,q,group


def test_freeze_keeps_only_planned_parents_and_detects_tampering(tmp_path):
    freeze,path,q,_=make_freeze(tmp_path)
    assert freeze['n_parents']==N and 'extra' not in freeze['parent_ids']
    rows,name,loaded=resolve_queries(q,path)
    assert name=='screen' and len(rows)==N and loaded['freeze_sha256']==freeze['freeze_sha256']
    altered=dict(freeze,n_parents=N+1);path.write_text(json.dumps(altered))
    with pytest.raises(ValueError):load_freeze(path)


def test_freeze_rejects_hold_plan_and_overlap():
    plan,split,rows,parent_of,_=plan_and_split()
    kwargs=dict(split=split,reserve_rows=rows,parent_of=parent_of,config={},bindings={},decision={})
    with pytest.raises(ValueError):build_freeze(power_plan=dict(plan,status='HOLD_POWER'),**kwargs)
    split['development']['parent_ids']=['p000']
    with pytest.raises(ValueError):build_freeze(power_plan=plan,**kwargs)


def test_screen_queries_require_freeze_and_matching_inventory(tmp_path):
    _,path,q,_=make_freeze(tmp_path)
    with pytest.raises(ValueError):resolve_queries(q)
    q.write_text(q.read_text().splitlines()[0]+'\n')
    with pytest.raises(ValueError):resolve_queries(q,path)


def test_evaluator_refuses_unbound_screen_runs(tmp_path):
    freeze,path,_,_=make_freeze(tmp_path)
    split=dict(split_hash='h',development=dict(parent_ids=['d0']))
    assert evaluation_parents(split,{},None)[0]=={'d0'}
    with pytest.raises(ValueError):evaluation_parents(split,dict(split_name='screen'),None)
    with pytest.raises(ValueError):evaluation_parents(split,dict(split_name='screen',freeze_sha256='x'),path)
    ids,name=evaluation_parents(split,dict(split_name='screen',freeze_sha256=freeze['freeze_sha256']),path)
    assert len(ids)==N and name=='screen_queries.jsonl'


def test_rarity_strata_boundaries():
    assert rarity_stratum(0)=='unseen_in_pilot' and rarity_stratum(4/128)=='rare'
    assert rarity_stratum(5/128)=='intermediate' and rarity_stratum(16/128)=='intermediate'
    assert rarity_stratum(17/128)=='common'


def reports(freeze,group,*,auc_gain,m0,mr,rng):
    base=dict(freeze_sha256=freeze['freeze_sha256'],split_hash='h')
    eff={};pulses={}
    for s in SEEDS:
        summaries={}
        for arm in ARMS:
            rows=[]
            for p in freeze['parent_ids']:
                y=int(rng.random()<(.3+(auc_gain if arm=='B1' else 0)))
                hits=[0,y,y,1,1]
                rows.append(dict(parent_id=p,split_group=group[p],query_id=p+'_anchor_x',hits=hits,event_hits=hits,
                                 any_reference_hits=hits,auc=float(np.dot(hits,[1,2,2,2,1])/8),
                                 event_auc=float(np.dot(hits,[1,2,2,2,1])/8),completed_candidates=[1,2,4,8,16],
                                 valid_candidates=[1,2,4,8,16],unique_events=[1,1,2,2,3],n_proposals=20,
                                 best_event_count=1,
                                 v1b_budget16=dict(candidate_ids=['c'],attempt_ids=['a'],best_event_candidate_ids=[],
                                                   proxy_best_reference_hit=bool(y))))
            summaries[arm]=dict(parent_rows=rows)
        eff[s]=dict(base,training_seed=s,summaries=summaries,attempt_status_counts={},source_sha256='e',
                    config=dict(freeze['config'],namespace='efficiency'))
        prow=[dict(parent_id=p,split_group=group[p],M0=m0+rng.normal(0,.05),MR=mr+rng.normal(0,.05))
              for p in freeze['parent_ids']]
        amps={str(a):dict(parent_rows=[dict(parent_id=p,valid_event_change=.1,random_valid_event_change=.1,
                                             coverage=1.,actual_rms=a) for p in freeze['parent_ids']])
              for a in AMPLITUDES}
        pulses[s]=dict(base,training_seed=s,path='sync',source_sha256='w',
                       windows=[dict(requested_t=.5,parent_rows=prow,amplitudes=amps,transitions={})])
    ctrl=dict(base,parent_rows=[dict(parent_id=p,split_group=group[p]) for p in freeze['parent_ids']],
              summary={},source_sha256='c')
    rar=dict(base,training_seed=0,config=dict(namespace='rarity_pilot',cap_units=6400.),source_sha256='r',
             summaries=dict(B0=dict(parent_rows=[dict(parent_id=p,split_group=group[p],n_proposals=128,
                                                       best_event_count=int(rng.integers(0,30)))
                                                  for p in freeze['parent_ids']])))
    return eff,pulses,ctrl,rar


def test_formal_analysis_go_and_mechanism_stop(tmp_path):
    freeze,_,_,group=make_freeze(tmp_path)
    eff,pulses,ctrl,rar=reports(freeze,group,auc_gain=.4,m0=.1,mr=.1,rng=np.random.default_rng(1))
    gate=formal_analysis(freeze,eff,pulses,ctrl,rar)
    assert gate['decision']=='GO_V1b' and not gate['window_stop_supported']
    assert sum(v['n_parents'] for v in gate['secondary']['rarity_strata'].values())==N
    eff,pulses,ctrl,rar=reports(freeze,group,auc_gain=.4,m0=.1,mr=0.,rng=np.random.default_rng(2))
    assert formal_analysis(freeze,eff,pulses,ctrl,rar)['decision']=='NO_GO_MECHANISM_SMALL'
    eff,pulses,ctrl,rar=reports(freeze,group,auc_gain=0.,m0=.1,mr=.1,rng=np.random.default_rng(3))
    assert formal_analysis(freeze,eff,pulses,ctrl,rar)['decision']=='NO_GO_RESOURCE_SMALL'


def test_formal_analysis_rejects_incomplete_or_unbound_inputs(tmp_path):
    freeze,_,_,group=make_freeze(tmp_path)
    eff,pulses,ctrl,rar=reports(freeze,group,auc_gain=.1,m0=.02,mr=.02,rng=np.random.default_rng(4))
    eff[1]['summaries']['B1']['parent_rows'].pop()
    with pytest.raises(ValueError):formal_analysis(freeze,eff,pulses,ctrl,rar)
    eff,pulses,ctrl,rar=reports(freeze,group,auc_gain=.1,m0=.02,mr=.02,rng=np.random.default_rng(5))
    pulses[2]['freeze_sha256']='other'
    with pytest.raises(ValueError):formal_analysis(freeze,eff,pulses,ctrl,rar)
    eff,pulses,ctrl,rar=reports(freeze,group,auc_gain=.1,m0=.02,mr=.02,rng=np.random.default_rng(6))
    eff[0]['config']['alpha_x']=.15
    with pytest.raises(ValueError):formal_analysis(freeze,eff,pulses,ctrl,rar)
    eff,pulses,ctrl,rar=reports(freeze,group,auc_gain=.1,m0=.02,mr=.02,rng=np.random.default_rng(7))
    del pulses[1]
    with pytest.raises(ValueError):formal_analysis(freeze,eff,pulses,ctrl,rar)


def test_v1b_export_has_every_parent_arm_bundle(tmp_path):
    freeze,_,_,group=make_freeze(tmp_path)
    eff,*_=reports(freeze,group,auc_gain=.1,m0=0,mr=0,rng=np.random.default_rng(8))
    rows=v1b_export(freeze,eff[0])
    assert len(rows)==5*N and len({r['bundle_id'] for r in rows})==5*N
    assert all(r['source_budget_units']==800 and r['training_seed']==0 for r in rows)
    with pytest.raises(ValueError):v1b_export(freeze,eff[1])
