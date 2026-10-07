import copy

import pytest

from xtbflow.v1.development import score_calibration,window_summary


def fixture():
    manifest=dict(requested_times=[.2],amplitudes=[.05,.1,.2],proposals=1,training_seed=0)
    parents={f'q{k}':dict(parent_id=f'p{k}',split_group=f'g{k}') for k in range(3)}
    rows=[]
    for q,p in parents.items():
        for branch in ['F0']+[f'{b}_{a}' for a in manifest['amplitudes'] for b in ('F_S','F_R','C_S')]:
            rows.append(dict(query_id=q,**p,proposal=0,requested_t=.2,branch=branch,training_seed=0,
                sampling_seed=0,b_dec=[[1]],decode_status='VALID',actual_rms=0,applicable=False,
                predicted_channel_id='same',event_utility=.5,proxy_status='PROXY_MATCH',
                catalog_barrier_kcal=10,score_at_pulse=1,mean_kcal_at_pulse=12,
                actual_t_x=.2,actual_t_b=.4,step_index=10))
    return rows,manifest,parents


def test_inapplicable_pairs_retained_and_missing_rejected():
    rows,m,p=fixture();r=window_summary(rows,m,p)
    s=r['windows'][0]['summary']
    assert s['M0']['estimate']==s['MR']['estimate']==s['coverage']['estimate']==0
    assert len(r['windows'][0]['parent_rows'])==3
    with pytest.raises(ValueError,match='incomplete'):window_summary(rows[:-1],m,p)
    with pytest.raises(ValueError,match='duplicate'):window_summary(rows+[rows[0]],m,p)


def test_replay_and_equal_amplitude_checked():
    rows,m,p=fixture();changed=copy.deepcopy(rows)
    changed[3]['b_dec']=[[2]]
    with pytest.raises(ValueError,match='replay'):window_summary(changed,m,p)
    changed=copy.deepcopy(rows);changed[2]['actual_rms']=.05
    with pytest.raises(ValueError,match='unequal'):window_summary(changed,m,p)


def test_unmatched_never_labels_or_inflates_calibration():
    rows,m,p=fixture();rr=[rows[0],dict(rows[0],score_at_pulse=2,catalog_barrier_kcal=20),
                         dict(rows[0],proxy_status='OUTSIDE_CATALOGUE_EVENT',catalog_barrier_kcal=None)]
    r=score_calibration(rr)
    assert r['n_total']==3 and r['n_matched']==2
    assert r['parent_rows'][0]['concordance_minus_discordance']==1
    assert r['matched_mae_kcal']==5
