from dataclasses import asdict, replace
import json

import pytest

from xtbflow.governance.mechanism_handoff import load_mechanism_candidates
from xtbflow.sampling.guidance_fixture import fixture_query
from xtbflow.sampling.integrated_guidance import candidate_from_state
from xtbflow.validation.certification import file_hash, identity, write_json


def bundle(tmp_path):
    cfg_path=tmp_path/'config.json';write_json(cfg_path,dict(arms=['N','T','F','C']))
    query,packet,state=fixture_query();protocol=dict(config_sha256=file_hash(cfg_path),source_commit='a'*40)
    plan=dict(source_commit='a'*40,config_sha256=file_hash(cfg_path),protocol=protocol,candidates=[])
    report=dict(status='generation_completed',reference_gate='eligible',settlement='settled',termination='confirmed',
                work=dict(unresolved_started=0),source_commit='a'*40,config_sha256=file_hash(cfg_path),candidates=[],new_candidate_handoff=[])
    for k,arm in enumerate(['N','T','F','C']):
        changed=replace(state,x=state.x*(1+.001*k))
        candidate=candidate_from_state(packet,changed,lambda b:b.int(),candidate_id=arm,protocol=protocol,
                    upstream=dict(status='completed',termination='confirmed',settlement='settled',reservations={}))
        sha=identity(candidate.content())
        plan['candidates'].append(dict(arm=arm,candidate=asdict(candidate),candidate_sha256=sha))
        report['candidates'].append(dict(arm=arm,status='generated',candidate_sha256=sha))
        report['new_candidate_handoff'].append(dict(arm=arm,candidate_sha256=sha,cache_hit=True,new_replay_calls=0))
    write_json(tmp_path/'plan.json',plan);report['candidate_plan_sha256']=file_hash(tmp_path/'plan.json')
    write_json(tmp_path/'report.json',report)
    return cfg_path,report,plan


def test_new_intervened_mapped_candidates_survive_exact_handoff(tmp_path):
    cfg,report,plan=bundle(tmp_path)
    _,rows=load_mechanism_candidates(tmp_path,cfg,file_hash(tmp_path/'report.json'))
    assert [arm for arm,c in rows]==['N','T','F','C']
    assert len({identity(c.content()) for arm,c in rows})==4
    for (_,candidate),item in zip(rows,plan['candidates']):
        assert candidate.coordinates_angstrom==item['candidate']['coordinates_angstrom']
        assert candidate.atom_ids==['map:1','map:2']


@pytest.mark.parametrize('change',[dict(settlement='unsettled'),dict(termination='unknown'),dict(work=dict(unresolved_started=1)),dict(reference_gate='do_not_launch')])
def test_reference_gate_rejects_unknown_or_unsettled_new_work(tmp_path,change):
    cfg,report,_=bundle(tmp_path);write_json(tmp_path/'report.json',report|change)
    with pytest.raises(ValueError,match='gate_closed'):
        load_mechanism_candidates(tmp_path,cfg,file_hash(tmp_path/'report.json'))


def test_reference_cannot_replace_frozen_candidate_or_drop_an_arm(tmp_path):
    cfg,report,plan=bundle(tmp_path);sha=file_hash(tmp_path/'report.json')
    plan['candidates'][2]['candidate']['coordinates_angstrom'][0][0]+=.01
    write_json(tmp_path/'plan.json',plan)
    with pytest.raises(ValueError,match='plan_identity'):
        load_mechanism_candidates(tmp_path,cfg,sha)
    report['candidates'].pop();write_json(tmp_path/'report.json',report)
    with pytest.raises(ValueError,match='arm_denominator'):
        load_mechanism_candidates(tmp_path,cfg,file_hash(tmp_path/'report.json'))
