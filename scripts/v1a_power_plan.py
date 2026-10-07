"""Plan complete-gate power from independent development parent summaries."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from xtbflow.v1.data import file_hash,write_json
from xtbflow.v1.power import nuisance,simulate_gate


def main():
    ap=argparse.ArgumentParser()
    for key in ('a2-report','b1-report','window-report','catalogue','out'):
        ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--time',type=float,required=True)
    ap.add_argument('--repeats',type=int,default=20000)
    ap.add_argument('--sd-multiplier',type=float,default=1.1)
    ap.add_argument('--icc-increment',type=float,default=.05)
    for key in ('extra-a2-reports','extra-b1-reports','extra-window-reports'):
        ap.add_argument('--'+key,type=Path,nargs='*',default=[])
    a=ap.parse_args()
    ar=json.loads(a.a2_report.read_text());br=json.loads(a.b1_report.read_text())
    wr=json.loads(a.window_report.read_text());split=json.loads((a.catalogue/'split_manifest.json').read_text())
    if any(r['split_hash']!=split['split_hash'] for r in (ar,br,wr)):
        raise ValueError('development split mismatch')
    # A's two serial stages always use linear clocks; its metadata joint-path
    # setting is inactive. B's actual joint path must agree with the pulse run
    # when the full configuration freeze is assembled.
    for key in ('alpha_x','guidance_start','guidance_stop','cap_units'):
        if ar['config'][key]!=br['config'][key]:raise ValueError('A2/B1 shared setting mismatch: '+key)
    if ar['config']['cap_units']!=3200:raise ValueError('full five-budget development stream required')
    a2={r['parent_id']:r for r in ar['summaries']['A2']['parent_rows']}
    b1={r['parent_id']:r for r in br['summaries']['B1']['parent_rows']}
    windows=[w for w in wr['windows'] if w['requested_t']==a.time]
    if len(windows)!=1:raise ValueError('window not uniquely specified')
    mech={r['parent_id']:r for r in windows[0]['parent_rows']}
    dev=set(split['development']['parent_ids'])
    if any(set(r)!=dev for r in (a2,b1,mech)):raise ValueError('incomplete development parents')
    order=sorted(dev);groups=[a2[p]['split_group'] for p in order]
    values=np.array([[b1[p]['auc']-a2[p]['auc'],mech[p]['M0'],mech[p]['MR']] for p in order])
    lengths=[len(a.extra_a2_reports),len(a.extra_b1_reports),len(a.extra_window_reports)]
    if len(set(lengths))!=1 or lengths[0] not in (0,2):raise ValueError('use one or all three development training seeds')
    per_seed=[values];seed_ids=[ar.get('training_seed',0)]
    if br.get('training_seed',0)!=seed_ids[0] or wr.get('training_seed',0)!=seed_ids[0]:
        raise ValueError('base development seeds differ')
    for apath,bpath,wpath in zip(a.extra_a2_reports,a.extra_b1_reports,a.extra_window_reports):
        aa,bb,ww=(json.loads(p.read_text()) for p in (apath,bpath,wpath))
        if any(r['split_hash']!=split['split_hash'] for r in (aa,bb,ww)):
            raise ValueError('extra development split mismatch')
        seed=aa['training_seed']
        if bb['training_seed']!=seed or ww['training_seed']!=seed:raise ValueError('extra seeds differ')
        for key in ('alpha_x','guidance_start','guidance_stop','a2_times','cap_units'):
            if aa['config'][key]!=ar['config'][key]:raise ValueError('extra A2 configuration differs')
        for key in ('path','alpha_x','guidance_start','guidance_stop','cap_units'):
            if bb['config'][key]!=br['config'][key]:raise ValueError('extra B1 configuration differs')
        if ww['path']!=bb['config']['path']:raise ValueError('pulse/efficiency path mismatch')
        av={r['parent_id']:r for r in aa['summaries']['A2']['parent_rows']}
        bv={r['parent_id']:r for r in bb['summaries']['B1']['parent_rows']}
        mv={r['parent_id']:r for w in ww['windows'] if w['requested_t']==a.time for r in w['parent_rows']}
        if any(set(r)!=dev for r in (av,bv,mv)):raise ValueError('extra seed incomplete')
        per_seed.append(np.array([[bv[p]['auc']-av[p]['auc'],mv[p]['M0'],mv[p]['MR']] for p in order]));seed_ids.append(seed)
    if sorted(seed_ids) not in ([0],[0,1,2]):raise ValueError('unexpected seed inventory')
    values=np.mean(per_seed,axis=0)
    nu=nuisance(values,groups)
    if len(seed_ids)==3:
        nu['scope']='Measured average of three development training seeds; no assumed variance-reduction factor.'
    nu['training_seeds']=seed_ids
    nu['per_seed_sd']=[v.std(0,ddof=1).tolist() for v in per_seed]
    parents=json.loads((a.catalogue/'parent_catalog.json').read_text())
    selected=set(split['screen_reserve']['parent_ids']);by_group={}
    for group in split['screen_reserve']['formula_groups']:
        by_group[group]=[p['parent_id'] for p in parents if p['parent_id'] in selected and p['split_group']==group]
    candidates=[];pids=[];gs=[];sizes=[]
    for group,ids in by_group.items():
        pids+=ids;gs.append(group);sizes.append(len(ids))
        if 200<=len(pids)<=350:
            candidates.append(dict(parent_ids=list(pids),formula_groups=list(gs),sizes=list(sizes)))
    if not candidates:raise ValueError('no complete-group 200-350 candidate inventory')
    scans=[];chosen=None
    # Conservative planning guard is exposed, fixed before screen inspection.
    # It is not a claim to cover every nuisance value compatible with 21 dev groups.
    for candidate in candidates:
        result=simulate_gate(values,groups,candidate['sizes'],[.05,.02,.02],repeats=a.repeats,
                             sd_multiplier=a.sd_multiplier,icc_increment=a.icc_increment)
        scans.append(result)
        print(json.dumps(dict(stage='target_scan',N=result['n_parents'],G=result['n_formula_groups'],
                              GO=result['GO'],mcse=result['mcse_GO'])),flush=True)
        if result['GO']-1.645*result['mcse_GO']>=.8:
            skew=simulate_gate(values,groups,candidate['sizes'],[.05,.02,.02],repeats=a.repeats,
                               sd_multiplier=a.sd_multiplier,icc_increment=a.icc_increment,distribution='empirical_skew')
            result['empirical_skew_check']=skew
            if skew['GO']-1.645*skew['mcse_GO']>=.8:
                chosen=candidate;break
    diagnostic=chosen or candidates[-1]
    scenarios=[]
    effects=[('target',[.05,.02,.02]),('zero_primary',[0,.02,.02]),('zero_mechanism',[.05,0,0]),
             ('zero_M0',[.05,0,.02]),('zero_MR',[.05,.02,0]),('all_zero',[0,0,0]),
             ('small_useful',[.025,.01,.01])]
    for distribution in ('normal','empirical_skew'):
        for label,mu in effects:
            for sensitivity,sdf,inc in [('estimated',1.,0.),('planning_guard',a.sd_multiplier,a.icc_increment)]:
                result=simulate_gate(values,groups,diagnostic['sizes'],mu,repeats=a.repeats,
                                     sd_multiplier=sdf,icc_increment=inc,distribution=distribution)
                scenarios.append(dict(label=label,sensitivity=sensitivity,**result))
    nulls=[r for r in scenarios if r['label'].startswith('zero_') or r['label']=='all_zero']
    inflation=any(r['GO']>.05+max(.005,2.58*r['mcse_GO']) for r in nulls)
    support_warning=any(r['outside_support_fraction']>.01 for r in scenarios)
    status='HOLD_POWER' if chosen is None else 'POWER_PLAN_CANDIDATE_PASS'
    if inflation or support_warning:status='POWER_MODEL_REPAIR_REQUIRED'
    report=dict(status=status,nuisance=nu,planning_guard=dict(sd_multiplier=a.sd_multiplier,icc_increment=a.icc_increment),
                target_scan=scans,scenarios=scenarios,null_inflation=inflation,support_warning=support_warning,
                planned_n_parents=None if status!='POWER_PLAN_CANDIDATE_PASS' else len(chosen['parent_ids']),
                candidate_inventory=diagnostic if chosen is None else chosen,
                planned_n_formula_groups=None if status!='POWER_PLAN_CANDIDATE_PASS' else len(chosen['formula_groups']),
                max_group_fraction=max(diagnostic['sizes'])/sum(diagnostic['sizes']),
                source_hashes={str(p):file_hash(p) for p in [a.a2_report,a.b1_report,a.window_report,
                    *a.extra_a2_reports,*a.extra_b1_reports,*a.extra_window_reports]},
                split_hash=split['split_hash'],screen_results_used_for_planning=False,power_plan_frozen=False,
                development_values=[dict(parent_id=p,split_group=g,AUC=float(v[0]),M0=float(v[1]),MR=float(v[2]))
                                    for p,g,v in zip(order,groups,values)],
                scope='Conditional working-model planning; no screen outcomes. Final engineering/configuration freeze remains separate.')
    write_json(a.out,report)
    print(json.dumps(dict(status=status,N=report['planned_n_parents'],G=report['planned_n_formula_groups'],
                         null_inflation=inflation,support_warning=support_warning),indent=2))


if __name__=='__main__':main()
