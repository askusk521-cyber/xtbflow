"""O2 DFT analysis on a synthetic run root; no quantum chemistry."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import review_open_world_o2_analyze as o2a  # noqa: E402
from xtbflow.v1.data import write_json  # noqa: E402

CONFIG = ROOT / 'configs/review/open_world_o2_analysis.json'
K = 627.509474


def rows(n_grad, stage='irc'):
    return ([dict(stage=stage, kind='scf', wall_s=1.0)] * n_grad + [dict(stage=stage, kind='gradient', wall_s=0.5)] * n_grad
            + [dict(stage='ts_final_hessian', kind='hessian', wall_s=4.0), dict(stage=stage, kind='geometric', wall_s=99.0)])


def chain(status, e_ts, local=None, irc='IRC_COMPLETE'):
    v = dict(strict_status=status, ts=dict(energy_hartree=e_ts), irc=dict(status=irc), wall_s=10.0, stage_timings=rows(2))
    if local is not None:
        v['local_barrier_kcal'] = local
    return v


def build(tmp, specs):
    """specs: parent -> dict(anchor=E or None, reference=(status, E, local), candidate=(status|exception, E), xtb=delta)."""
    run, items, parents, ledger, job = tmp / 'run', [], [], [], 100
    for pid, s in specs.items():
        parents.append(dict(parent_id=pid, split_group=s['group'], formula=s['group'], anchor_reference_id='r-' + pid))
        for kind in ('anchor', 'reference', 'candidate'):
            iid = f'{kind[:3]}{pid}'
            item = dict(parent_id=pid, kind=kind, item_id=iid, z=[6, 1])
            if kind == 'reference':
                item.update(reference_id=s.get('ref_id', 'r-' + pid), catalog_barrier_kcal=s['catalog'])
            if kind == 'candidate':
                item.update(candidate_id='c-' + pid, delta_ea_xtb_kcal=s['xtb'])
            items.append(item)
            work = run / 'o2-run' / f'{kind}_{iid}'
            work.mkdir(parents=True)
            job += 1
            write_json(work / 'manifest.json', dict(slurm_job_id=str(job)))
            ledger.append(dict(item_id=iid, kind=kind, parent_id=pid, job_id=str(job), outcome='VERDICT', runtime_s=60))
            if kind == 'anchor':
                write_json(work / 'verdict.json', dict(status='MINIMUM_PASS', energy_hartree=s['anchor'], wall_s=1.0,
                                                       stage_timings=[]))
            elif s[kind][0] == 'EXCEPTION':
                write_json(work / 'unexpected_failure.json', dict(type='EngineError', message='SCF_LIMIT', wall_s=5.0,
                                                                  stage_timings=rows(1)))
            else:
                write_json(work / 'verdict.json', chain(*s[kind]))
    write_json(run / 'o2-plan.json', dict(items=items, protocol_version='v1b-h-gpu4pyscf-3', config_sha256='abc'))
    (run / 'o2-jobs').mkdir()
    write_json(run / 'o2-jobs' / 'ledger_v4.json', ledger + [dict(item_id='x', kind='anchor', parent_id='p0', outcome='PRIOR_ATTEMPT')])
    write_json(run / 'o2-jobs' / 'ledger_v2.json', [dict(item_id='q', kind='anchor', parent_id='p0', job_id='7', outcome='CHAIN_EXCEPTION',
                                                         runtime_s=2, slurm_state='FAILED')])
    (run / 'o2-run-preqc-failed').mkdir()
    write_json(run / 'o2-run-preqc-failed' / 'quarantine.json', [dict(item_id='q', job_id='7')])
    write_json(tmp / 'parents.json', parents)
    return run, tmp / 'parents.json'


SPECS = {
    'p1': dict(group='G1', anchor=-1.0, catalog=60.0, xtb=-1.0,
               reference=('STRICT_JOINT_GRAPH_VALID', -0.9, 62.0), candidate=('STRICT_JOINT_GRAPH_VALID', -0.902, 70.0)),
    'p2': dict(group='G2', anchor=-2.0, catalog=50.0, xtb=3.0, ref_id='other',
               reference=('STRICT_JOINT_GRAPH_VALID', -1.92, 51.0), candidate=('STRICT_JOINT_GRAPH_VALID', -1.91, 55.0)),
    'p3': dict(group='G3', anchor=-3.0, catalog=40.0, xtb=0.0,
               reference=('STRICT_JOINT_GRAPH_VALID', -2.94, 41.0), candidate=('STRICT_JOINT_GRAPH_VALID', -2.94, 41.0)),
    'p4': dict(group='G4', anchor=-4.0, catalog=40.0, xtb=1.0,
               reference=('EXCEPTION', None), candidate=('STRICT_EVENT_MISMATCH', -3.9, 30.0)),
}


def test_load_item_and_stage_summary(tmp_path):
    run, _ = build(tmp_path, SPECS)
    rec = o2a.load_item(run / 'o2-run' / 'reference_refp4')
    assert rec['status'] == 'CHAIN_EXCEPTION:EngineError:SCF_LIMIT' and rec['verdict'] is None
    assert o2a.load_item(tmp_path / 'nowhere')['status'] == 'MISSING'
    s = o2a.stage_summary(rows(3))
    assert s['calls'] == dict(scf=3, gradient=3, hessian=1)
    assert s['qc_kernel_wall_s'] == pytest.approx(3 * 1.5 + 4.0)  # geometric wrapper time is not added
    assert s['stage_geometric_wall_s'] == dict(irc=99.0)


def test_analysis_pairs_only_double_passes_and_applies_frozen_rules(tmp_path):
    run, parents = build(tmp_path, SPECS)
    out = o2a.analyze(run, parents, CONFIG)
    cert = out['certification']
    assert cert['candidate']['passed']['success'] == 3 and cert['candidate']['planned'] == 4
    assert cert['reference']['passed']['success'] == 3
    assert cert['reference']['statuses']['CHAIN_EXCEPTION:EngineError:SCF_LIMIT'] == 1
    paired = {r['parent_id']: r for r in out['paired']['rows']}
    assert sorted(paired) == ['p1', 'p2', 'p3']
    assert out['paired']['excluded'][0]['parent_id'] == 'p4'
    assert paired['p1']['delta_ea_dft_kcal'] == pytest.approx(-0.002 * K)
    assert paired['p1']['candidate_barrier_vs_anchor_kcal'] == pytest.approx(0.098 * K)
    assert paired['p1']['sign_agree'] and paired['p2']['sign_agree']
    assert not paired['p3']['sign_agree']  # exact zero counts as disagreement
    assert out['paired']['below_zero']['success'] == 1 and out['paired']['at_most_plus5']['success'] == 2
    assert out['paired']['delta_ea_dft_summary']['n_parents'] == 3
    cat = {r['parent_id']: r for r in out['catalogue_check']['rows']}
    assert cat['p1']['diff_vs_anchor_kcal'] == pytest.approx(0.1 * K - 60.0) and cat['p1']['flag_abs_gt']
    assert cat['p2']['same_reaction_as_anchor'] is False and cat['p1']['same_reaction_as_anchor'] is True
    gpu = out['gpu']
    assert gpu['total_runtime_s'] == 12 * 60 + 2 and gpu['by_category']['pre_qc_infrastructure'] == dict(jobs=1, runtime_s=2)
    assert out['items'][0]['slurm_runtime_s'] == 60


def test_results_are_byte_identical_on_rerun(tmp_path):
    run, parents = build(tmp_path, SPECS)
    write_json(tmp_path / 'a.json', o2a.analyze(run, parents, CONFIG))
    write_json(tmp_path / 'b.json', o2a.analyze(run, parents, CONFIG))
    assert (tmp_path / 'a.json').read_bytes() == (tmp_path / 'b.json').read_bytes()


def test_delta_summary_needs_two_groups():
    assert o2a.delta_summary([dict(delta_ea_dft_kcal=1.0, split_group='G')])['status'] == 'NOT_ESTIMABLE'
