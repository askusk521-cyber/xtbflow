"""Formal V1a screen: freeze, split resolution, single final analysis, V1b export.

Sampling code may import only `resolve_queries`/`load_freeze`; everything else
reads sealed, already matched outputs. The freeze is written once before any
screen generation and binds the queries, configuration, models, costs and power
plan by SHA-256. The final analysis refuses any input that is not bound to it.
"""
import json
from pathlib import Path

import numpy as np

from .costs import BUDGETS
from .data import digest,file_hash
from .gate import decide_v1a
from .metrics import cluster_bootstrap,cluster_summary

ARMS=('A0','A1','A2','B0','B1')
SEEDS=(0,1,2)
AMPLITUDES=(.05,.10,.20)
# Section 8.6 strata on best-known-event frequency in the 128-proposal B0 pilot.
RARITY_EDGES=((0.,'unseen_in_pilot'),(1/32,'rare'),(1/8,'intermediate'),(1.,'common'))
EVENT_RESPONSE_FUTILITY=.02


def _self_hash(freeze):
    return digest({k:v for k,v in freeze.items() if k!='freeze_sha256'})


def build_freeze(*,power_plan,split,reserve_rows,parent_of,config,bindings,decision):
    """Return (freeze, screen query rows) from a passing pre-screen power plan.

    `reserve_rows` are the reactant-only screen-reserve queries and `parent_of`
    maps query_id to parent_id from the parent catalogue; only the planned
    parents are kept, in their reserve order. No screen output may exist yet.
    """
    if power_plan['status']!='POWER_PLAN_CANDIDATE_PASS':
        raise ValueError('power plan does not support a frozen N')
    if power_plan['screen_results_used_for_planning'] or power_plan['split_hash']!=split['split_hash']:
        raise ValueError('power plan provenance mismatch')
    inventory=power_plan['candidate_inventory'];planned=list(inventory['parent_ids'])
    if len(planned)!=power_plan['planned_n_parents'] or not 200<=len(planned)<=350:
        raise ValueError('planned N outside the protocol range')
    if not set(planned)<=set(split['screen_reserve']['parent_ids']):
        raise ValueError('planned parents outside the screen reserve')
    if set(planned)&(set(split['development']['parent_ids'])|set(split['train']['parent_ids'])):
        raise ValueError('planned parents overlap training or development')
    by_query={r['query_id']:r for r in reserve_rows}
    if len(by_query)!=len(reserve_rows):raise ValueError('duplicate reserve query')
    rows=[r for r in reserve_rows if parent_of[r['query_id']] in set(planned)]
    if sorted(parent_of[r['query_id']] for r in rows)!=sorted(planned):
        raise ValueError('every planned parent needs exactly one anchored query')
    freeze=dict(schema='xtbflow-v1a-freeze/1',stage='formal_screen_frozen',
                split_hash=split['split_hash'],parent_ids=sorted(planned),
                formula_groups=list(inventory['formula_groups']),group_sizes=list(inventory['sizes']),
                n_parents=len(planned),n_formula_groups=len(inventory['formula_groups']),
                screen_queries_sha256=digest(rows),config=config,bindings=bindings,
                power_plan=dict(status=power_plan['status'],planning_guard=power_plan['planning_guard'],
                                target=[s for s in power_plan['scenarios'] if s['label']=='target'],
                                nuisance=power_plan['nuisance']),
                planning_decision=decision,training_seeds=list(SEEDS),sampling_seeds=[0],
                single_final_analysis=True,quantum_evaluations_allowed=False)
    freeze['freeze_sha256']=_self_hash(freeze)
    return freeze,rows


def load_freeze(path):
    freeze=json.loads(Path(path).read_text())
    if freeze.get('schema')!='xtbflow-v1a-freeze/1' or freeze['freeze_sha256']!=_self_hash(freeze):
        raise ValueError('freeze record altered or unrecognized')
    return freeze


def resolve_queries(path,frozen=None):
    """Development queries need no freeze; any other input must match the freeze."""
    path=Path(path);rows=[json.loads(line) for line in path.read_text().splitlines()]
    if frozen is None:
        if path.name!='development_queries.jsonl':raise ValueError('screen input requires --frozen')
        return rows,'development',None
    freeze=load_freeze(frozen)
    if path.name!='screen_queries.jsonl' or digest(rows)!=freeze['screen_queries_sha256']:
        raise ValueError('queries differ from the frozen screen inventory')
    return rows,'screen',freeze


def evaluation_parents(split,manifest,frozen=None):
    """Parent ids an offline evaluator may read, checked against the source run."""
    name=manifest.get('split_name','development')
    if name=='development':
        if manifest.get('freeze_sha256') is not None:raise ValueError('development run bound to a freeze')
        return set(split['development']['parent_ids']),'development_queries.jsonl'
    if frozen is None:raise ValueError('screen evaluation requires --frozen')
    freeze=load_freeze(frozen)
    if manifest.get('freeze_sha256')!=freeze['freeze_sha256'] or freeze['split_hash']!=split['split_hash']:
        raise ValueError('screen run not bound to this freeze')
    return set(freeze['parent_ids']),'screen_queries.jsonl'


def query_source_ok(catalogue,query_name,manifest):
    rows=[json.loads(line) for line in (Path(catalogue)/query_name).read_text().splitlines()]
    return file_hash(Path(catalogue)/query_name)==manifest['query_source_sha256'],rows


def rarity_stratum(frequency):
    if not 0<=frequency<=1:raise ValueError('frequency outside [0,1]')
    if frequency==0:return 'unseen_in_pilot'
    for edge,name in RARITY_EDGES[1:]:
        if frequency<=edge+1e-12:return name
    raise AssertionError


def _check(report,freeze,kind,seed=None):
    if report.get('freeze_sha256')!=freeze['freeze_sha256'] or report['split_hash']!=freeze['split_hash']:
        raise ValueError(f'{kind} report not bound to the freeze')
    if seed is not None and report['training_seed']!=seed:raise ValueError(f'{kind} seed mismatch')


def _rows(rows,freeze,kind):
    index={r['parent_id']:r for r in rows}
    if len(index)!=len(rows) or set(index)!=set(freeze['parent_ids']):
        raise ValueError(f'{kind}: parent inventory differs from the freeze')
    return index


def _summary(values,groups):
    if len(set(groups))<2:
        return dict(estimate=float(np.mean(values)),n_parents=len(values),
                    n_formula_groups=len(set(groups)),status='TOO_FEW_GROUPS_FOR_INTERVAL')
    return cluster_summary(values,groups)


def window_stop(pulse_reports,order,groups):
    """Section 7.7: every amplitude/direction bounded below 0.02 with real displacement."""
    detail={};supported=True
    for a in AMPLITUDES:
        rows=[{r['parent_id']:r for r in rep['windows'][0]['amplitudes'][str(a)]['parent_rows']}
              for rep in pulse_reports]
        item={}
        for key in ('valid_event_change','random_valid_event_change'):
            s=_summary([np.mean([rr[p][key] for rr in rows]) for p in order],groups)
            item[key]=s
            if s.get('status') or s['upper_one95']>=EVENT_RESPONSE_FUTILITY:supported=False
        coverage=np.mean([rr[p]['coverage'] for rr in rows for p in order])
        realised=np.mean([rr[p]['actual_rms'] for rr in rows for p in order])/max(coverage,1e-12)
        item.update(coverage=float(coverage),mean_actual_rms_when_applicable=float(realised))
        if coverage<.5 or realised<.9*a:supported=False
        detail[str(a)]=item
    return supported,detail


def formal_analysis(freeze,efficiency,pulses,controls,rarity):
    """The one pre-specified analysis. Inputs: per-seed reports keyed by seed."""
    if sorted(efficiency)!=list(SEEDS) or sorted(pulses)!=list(SEEDS):
        raise ValueError('all three training seeds required')
    for s in SEEDS:
        _check(efficiency[s],freeze,'efficiency',s);_check(pulses[s],freeze,'pulse',s)
        if sorted(efficiency[s]['summaries'])!=sorted(ARMS):raise ValueError('five efficiency arms required')
        cfg=efficiency[s]['config']
        for key in ('path','alpha_x','alpha_b','guidance_start','guidance_stop','a2_times','cap_units'):
            if cfg[key]!=freeze['config'][key]:raise ValueError('efficiency config differs from freeze: '+key)
        if cfg['namespace']!='efficiency':raise ValueError('formal efficiency noise namespace required')
        if len(pulses[s]['windows'])!=1 or pulses[s]['windows'][0]['requested_t']!=freeze['config']['t_star']:
            raise ValueError('pulse report not at the frozen t*')
        if pulses[s]['path']!=freeze['config']['path']:raise ValueError('pulse path differs from freeze')
    _check(controls,freeze,'controls');_check(rarity,freeze,'rarity',0)
    if rarity['config']['namespace']!='rarity_pilot' or rarity['config']['cap_units']!=50*128:
        raise ValueError('rarity pilot must be 128 independent B0 proposals')

    eff={s:{arm:_rows(efficiency[s]['summaries'][arm]['parent_rows'],freeze,arm) for arm in ARMS} for s in SEEDS}
    mech={s:_rows(pulses[s]['windows'][0]['parent_rows'],freeze,'pulse') for s in SEEDS}
    ctrl=_rows(controls['parent_rows'],freeze,'controls')
    pilot=_rows(rarity['summaries']['B0']['parent_rows'],freeze,'rarity')
    order=sorted(freeze['parent_ids']);groups=[eff[0]['A2'][p]['split_group'] for p in order]
    if any(eff[s][arm][p]['split_group']!=g for s in SEEDS for arm in ARMS for p,g in zip(order,groups)):
        raise ValueError('formula group differs between reports')

    def mean_seed(fn):return np.array([np.mean([fn(s,p) for s in SEEDS]) for p in order])
    d=mean_seed(lambda s,p:eff[s]['B1'][p]['auc']-eff[s]['A2'][p]['auc'])
    m0=mean_seed(lambda s,p:mech[s][p]['M0']);mr=mean_seed(lambda s,p:mech[s][p]['MR'])
    primary=cluster_summary(d,groups);mech_none=cluster_summary(m0,groups);mech_random=cluster_summary(mr,groups)
    stop_supported,stop_detail=window_stop([pulses[s] for s in SEEDS],order,groups)
    gate=decide_v1a(True,True,freeze['power_plan']['status']=='POWER_PLAN_CANDIDATE_PASS',
                    primary,mech_none,mech_random,window_stop_supported=stop_supported)

    secondary={}
    secondary['event_level_best_channel_auc_B1_minus_A2']=cluster_summary(
        mean_seed(lambda s,p:eff[s]['B1'][p]['event_auc']-eff[s]['A2'][p]['event_auc']),groups)
    secondary['budget16_binary_B1_minus_A2']=cluster_summary(
        mean_seed(lambda s,p:eff[s]['B1'][p]['hits'][2]-eff[s]['A2'][p]['hits'][2]),groups)
    secondary['auc_B1_minus_B0']=cluster_summary(
        mean_seed(lambda s,p:eff[s]['B1'][p]['auc']-eff[s]['B0'][p]['auc']),groups)
    secondary['near_best_any_reference_auc_note']='any_reference_hits retained per arm; near-best window not separately labeled here'
    secondary['continuous_controls']=controls['summary']
    secondary['primary_cluster_bootstrap']=cluster_bootstrap(d,groups)
    secondary['per_seed_primary']={str(s):cluster_summary(
        [eff[s]['B1'][p]['auc']-eff[s]['A2'][p]['auc'] for p in order],groups) for s in SEEDS}
    secondary['per_seed_mechanism']={str(s):dict(
        M0=cluster_summary([mech[s][p]['M0'] for p in order],groups),
        MR=cluster_summary([mech[s][p]['MR'] for p in order],groups)) for s in SEEDS}
    secondary['mechanism_by_amplitude']={str(s):{a:{k:v for k,v in item.items() if k!='parent_rows'}
        for a,item in pulses[s]['windows'][0]['amplitudes'].items()} for s in SEEDS}
    secondary['event_transitions']={str(s):pulses[s]['windows'][0]['transitions'] for s in SEEDS}
    secondary['window_stop_detail']=stop_detail
    ev=secondary['event_level_best_channel_auc_B1_minus_A2']
    secondary['event_gain_not_converted_to_geometry']=bool(ev['estimate']>0 and primary['estimate']<=0)

    curves={}
    for arm in ARMS:
        curves[arm]={k:np.mean([[eff[s][arm][p][k] for p in order] for s in SEEDS],axis=(0,1)).tolist()
                     for k in ('hits','event_hits','any_reference_hits','completed_candidates',
                               'valid_candidates','unique_events')}
        curves[arm]['auc']=float(np.mean([[eff[s][arm][p]['auc'] for p in order] for s in SEEDS]))
        curves[arm]['event_auc']=float(np.mean([[eff[s][arm][p]['event_auc'] for p in order] for s in SEEDS]))
    strata={}
    for p,g,v in zip(order,groups,d):
        r=pilot[p]
        if r['n_proposals']!=128:raise ValueError('pilot did not plan 128 proposals')
        strata.setdefault(rarity_stratum(r['best_event_count']/128),[]).append((v,g,r['best_event_count']))
    secondary['rarity_strata']={k:dict(n_parents=len(v),best_event_counts=[c for _,_,c in v],
        primary=_summary([x for x,_,_ in v],[g for _,g,_ in v])) for k,v in strata.items()}
    failures={str(s):efficiency[s]['attempt_status_counts'] for s in SEEDS}
    return dict(schema='xtbflow-v1a-gate/1',decision=gate['decision'],reason=gate['reason'],
                freeze_sha256=freeze['freeze_sha256'],n_parents=len(order),n_formula_groups=len(set(groups)),
                primary=primary,mechanism_vs_none=mech_none,mechanism_vs_random=mech_random,
                window_stop_supported=stop_supported,secondary=secondary,curves=curves,budgets=list(BUDGETS),
                attempt_status_counts=failures,
                source_sha256=dict(efficiency={str(s):efficiency[s]['source_sha256'] for s in SEEDS},
                                   pulses={str(s):pulses[s]['source_sha256'] for s in SEEDS},
                                   controls=controls['source_sha256'],rarity=rarity['source_sha256']),
                parent_values=[dict(parent_id=p,split_group=g,AUC_B1_minus_A2=float(a),M0=float(b),MR=float(c))
                               for p,g,a,b,c in zip(order,groups,d,m0,mr)],
                scope=('Finite T1x catalogue proxy screen conditional on three frozen training runs; '
                       'PROXY_MATCH is not a certified TS and no quantum calculation was run.'))


def v1b_export(freeze,report):
    """Section 8.8 source population: budget 16, training/sampling seed 0, every arm."""
    _check(report,freeze,'efficiency',0)
    rows=[]
    for arm in ARMS:
        for r in sorted(report['summaries'][arm]['parent_rows'],key=lambda r:r['parent_id']):
            v=r['v1b_budget16']
            row=dict(fold_id=0,parent_id=r['parent_id'],split_group=r['split_group'],
                     query_id=r['query_id'],arm=arm,training_seed=0,sampling_seed=0,
                     source_budget_equiv=16,source_budget_units=800,
                     cost_table_hash=freeze['bindings']['cost_table_sha256'],
                     bundle_id=f"{r['parent_id']}_{arm}_train0_budget16",
                     candidate_ids=v['candidate_ids'],attempt_ids=v['attempt_ids'],
                     proxy_best_reference_hit=v['proxy_best_reference_hit'],
                     best_event_candidate_ids=v['best_event_candidate_ids'],
                     stream_complete=True,stream_sha256=report['source_sha256'])
            row['source_hash']=digest(row);rows.append(row)
    _rows([r for r in rows if r['arm']=='B1'],freeze,'v1b export')
    return rows
