"""Offline development diagnostics; never import this module in sampling code."""
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import spearmanr

from .metrics import cluster_summary


def score_calibration(rows, score_key='score_at_pulse', mean_key='mean_kcal_at_pulse'):
    """Labels exist ONLY for a structure-matched terminal reference record.

    Within-parent pair concordance excludes tied true labels and is parent weighted.
    This descriptive development analysis cannot establish score qualification alone.
    """
    counts=Counter(r['proxy_status'] for r in rows)
    matched=[r for r in rows if r['proxy_status']=='PROXY_MATCH']
    grouped=defaultdict(list)
    for r in matched:
        if r['catalog_barrier_kcal'] is None:raise ValueError('matched label missing')
        grouped[r['parent_id']].append(r)
    parents=[]
    for pid,rr in sorted(grouped.items()):
        y=np.array([r['catalog_barrier_kcal'] for r in rr])
        s=np.array([r[score_key] for r in rr])
        pairs=[]
        for i in range(len(rr)):
            for j in range(i):
                if abs(y[i]-y[j])<1e-8:continue
                sign=np.sign((s[i]-s[j])*(y[i]-y[j]))
                pairs.append(float(sign))
        rho=None
        if len(rr)>1 and np.ptp(y)>1e-8 and np.ptp(s)>1e-8:
            rho=float(spearmanr(s,y).statistic)
        parents.append(dict(parent_id=pid,split_group=rr[0]['split_group'],n_matched=len(rr),
                            informative_pairs=len(pairs),spearman=rho,
                            concordance_minus_discordance=None if not pairs else float(np.mean(pairs))))
    informative=[r for r in parents if r['informative_pairs']]
    concordance=None
    if len({r['split_group'] for r in informative})>=2:
        concordance=cluster_summary([r['concordance_minus_discordance'] for r in informative],
                                    [r['split_group'] for r in informative])
    return dict(status_counts=dict(counts),n_total=len(rows),n_matched=len(matched),
                matched_mae_kcal=None if not matched else float(np.mean([
                    abs(r[mean_key]-r['catalog_barrier_kcal']) for r in matched])),
                within_parent_concordance=concordance,parent_rows=parents,
                selection_bias='Catalogue barriers label only matched endpoints; unmatched states remain unlabeled.')


def event_class(row):
    if row['proxy_status']=='INVALID_OUTPUT':return 'invalid'
    if row['proxy_status']=='OUTSIDE_CATALOGUE_EVENT':return 'unknown'
    return 'known'


def window_summary(rows, manifest, parents):
    times=manifest['requested_times'];amps=manifest['amplitudes'];n=manifest['proposals']
    branches=['F0']+[f'{b}_{a}' for a in amps for b in ('F_S','F_R','C_S')]
    indexed={}
    for r in rows:
        key=(r['query_id'],r['proposal'],r['requested_t'],r['branch'])
        if key in indexed:raise ValueError('duplicate window record')
        if r['training_seed']!=manifest['training_seed'] or r['sampling_seed']!=0:
            raise ValueError('unexpected seed')
        indexed[key]=r
    expected={(q,j,t,b) for q in parents for j in range(n) for t in times for b in branches}
    if set(indexed)!=expected:raise ValueError('incomplete or unexpected window pairs')
    output=[]
    for t in times:
        parent_rows=[];transitions=Counter();by_amp=defaultdict(list)
        f0rows=[]
        for q,p in sorted(parents.items()):
            measures=defaultdict(list)
            for j in range(n):
                f0=indexed[q,j,t,'F0'];f0rows.append(f0)
                for a in amps:
                    fs,fr,cs=(indexed[q,j,t,f'{b}_{a}'] for b in ('F_S','F_R','C_S'))
                    if cs['b_dec']!=f0['b_dec'] or cs['decode_status']!=f0['decode_status']:
                        raise ValueError('replay event differs from paired F0')
                    if cs['predicted_channel_id']!=f0['predicted_channel_id']:
                        raise ValueError('replay channel differs from F0')
                    actual=[r['actual_rms'] for r in (fs,fr,cs)]
                    if max(actual)-min(actual)>1e-7 or not 0<=actual[0]<=a+1e-6:
                        raise ValueError('unequal or excessive pulse amplitude')
                    if not fs['applicable'] and (actual[0]!=0 or any(
                            r['b_dec']!=f0['b_dec'] for r in (fs,fr,cs))):
                        raise ValueError('inapplicable pulse changed event')
                    m0=fs['event_utility']-f0['event_utility']
                    mr=fs['event_utility']-fr['event_utility']
                    pair=dict(M0=m0,MR=mr,coverage=float(fs['applicable']),actual_rms=actual[0],
                              valid_event_change=float(event_class(fs)!='invalid' and
                                  event_class(f0)!='invalid' and fs['predicted_channel_id']!=f0['predicted_channel_id']),
                              random_valid_event_change=float(event_class(fr)!='invalid' and
                                  event_class(f0)!='invalid' and fr['predicted_channel_id']!=f0['predicted_channel_id']),
                              valid_to_invalid=float(event_class(f0)!='invalid' and event_class(fs)=='invalid'),
                              beneficial=float(m0>1e-12),harmful=float(m0< -1e-12),
                              unchanged=float(fs['predicted_channel_id']==f0['predicted_channel_id']))
                    for key,value in pair.items():measures[key].append(value)
                    by_amp[a].append(dict(parent_id=p['parent_id'],split_group=p['split_group'],**pair))
                    transitions[f"{event_class(f0)}->{event_class(fs)}"]+=1
            parent_rows.append(dict(parent_id=p['parent_id'],split_group=p['split_group'],
                                     **{k:float(np.mean(v)) for k,v in measures.items()}))
        groups=[r['split_group'] for r in parent_rows]
        amp_summary={}
        for a,rr in by_amp.items():
            amp_parents=[];amp_groups=[]
            for pid in sorted({r['parent_id'] for r in rr}):
                pp=[r for r in rr if r['parent_id']==pid]
                amp_parents.append({k:float(np.mean([r[k] for r in pp])) for k in measures})
                amp_groups.append(pp[0]['split_group'])
            amp_summary[str(a)]={k:cluster_summary([r[k] for r in amp_parents],amp_groups)
                                 for k in ('M0','MR','valid_event_change','random_valid_event_change')}
        output.append(dict(requested_t=t,actual_t_x=f0rows[0]['actual_t_x'],
                           actual_t_b=f0rows[0]['actual_t_b'],step_index=f0rows[0]['step_index'],
                           summary={k:cluster_summary([r[k] for r in parent_rows],groups) for k in measures},
                           amplitudes=amp_summary,transitions=dict(transitions),parent_rows=parent_rows,
                           matched_F0_calibration=score_calibration(f0rows)))
    return dict(windows=output,integrity='COMPLETE_PAIRED_WINDOW_PASS',
                interpretation='Development diagnostics only. No formal gate and no score qualification without A0/A2/B0/B1 rollout calibration.')
