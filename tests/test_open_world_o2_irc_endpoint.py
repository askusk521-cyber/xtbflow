"""Supplementary O2 IRC-endpoint diagnostic on synthetic inputs; no quantum chemistry."""
import json
from pathlib import Path, PurePosixPath
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import review_open_world_o2_irc_endpoint as ie  # noqa: E402
from xtbflow.v1.data import file_hash, write_json  # noqa: E402

K = 627.509474
CHECKS = dict(ts_energy_abs_hartree=1e-7, ts_coordinates_max_abs_angstrom=1e-6)
TS_X = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]]


def frame(iteration, energy, dz):
    return f"2\nIteration {iteration} Energy {energy: .8f}\nC 0.0 0.0 0.0\nH 0.0 0.0 {1.5 + dz:.10f}\n"


def trajectory(path, ts_energy=-1.0):
    """Forward end -> TS -> backward end, as geomeTRIC writes irc_direction='both'."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frame(2, -1.2, -0.4) + frame(1, -1.1, -0.2) + frame(0, ts_energy, 0.0)
                    + frame(1, -1.05, 0.2) + frame(2, -1.15, 0.5))
    return path


def test_irc_ends_takes_first_and_last_frame_after_checks(tmp_path):
    frames = ie.read_frames(trajectory(tmp_path / 'irc.xyz'))
    ts = dict(energy_hartree=-1.0, x_ts=TS_X)
    first, last, info = ie.irc_ends(frames, [6, 1], ts, CHECKS)
    assert first[1, 2] == pytest.approx(1.1) and last[1, 2] == pytest.approx(2.0)
    assert info['ts_index'] == 2 and info['first'] == dict(iteration=2, energy_hartree=-1.2)
    with pytest.raises(ValueError, match='energy'):
        ie.irc_ends(frames, [6, 1], dict(ts, energy_hartree=-1.001), CHECKS)
    with pytest.raises(ValueError, match='coordinates'):
        ie.irc_ends(frames, [6, 1], dict(ts, x_ts=[[0, 0, 0], [0, 0, 1.6]]), CHECKS)
    with pytest.raises(ValueError, match='symbols'):
        ie.irc_ends(frames, [6, 8], ts, CHECKS)
    with pytest.raises(ValueError, match='one TS frame'):
        ie.irc_ends(frames + frames[2:3], [6, 1], ts, CHECKS)


class FakeQC:
    HARTREE_TO_KCAL = K

    class SCFError(RuntimeError):
        pass

    def __init__(self, statuses, identity=None, raise_scf=False):
        self.statuses, self.identity, self.raise_scf, self.calls = list(statuses), identity, raise_scf, []

    def minimum_stage(self, z, x, method, meter, workdir, label):
        self.calls.append((label, Path(workdir).name, float(x[1][2])))
        if self.raise_scf:
            raise self.SCFError('SCF_LIMIT')
        status = self.statuses.pop(0)
        return dict(status=status, x=x, energy_hartree=-2.0 - len(self.calls), frequencies_cm1=[10.0])

    def endpoint_identity(self, z, b_r, perms, b_predicted, x1, x2):
        self.calls.append(('identity',))
        return self.identity


ITEM = dict(b_r=[[0]], perms=[[0]], b_predicted=[[0]])


def test_endpoint_diagnostic_mirrors_certification_chain_order():
    qc = FakeQC(['MINIMUM_PASS', 'MINIMUM_GRAY_ZONE'], dict(status='STRICT_JOINT_GRAPH_VALID', reactant_end=1))
    first, last = np.array(TS_X) + [0, 0, 0.1], np.array(TS_X) + [0, 0, 0.2]
    out = ie.endpoint_diagnostic(qc, [6, 1], first, last, ITEM, -1.0, None, None, 'w')
    assert qc.calls == [('end_first', 'first', 1.6), ('end_last', 'last', 1.7), ('identity',)]
    assert out['diagnostic_status'] == 'STRICT_JOINT_GRAPH_VALID'
    assert out['local_barrier_kcal'] == pytest.approx((-1.0 - -4.0) * K)  # reactant end = second minimum

    qc = FakeQC(['MINIMUM_PASS', 'MIN_OPT_LIMIT'])
    out = ie.endpoint_diagnostic(qc, [6, 1], first, last, ITEM, -1.0, None, None, 'w')
    assert out['diagnostic_status'] == 'ENDPOINT_FAILURE' and ('identity',) not in qc.calls

    out = ie.endpoint_diagnostic(FakeQC([], raise_scf=True), [6, 1], first, last, ITEM, -1.0, None, None, 'w')
    assert out == dict(diagnostic_status='PROTOCOL_FAILURE', failure='SCF_LIMIT')


def build(tmp):
    run, pid = tmp / 'run', 'p1'
    items = [dict(kind=kind, item_id=kind[:3] + '1', parent_id=pid, z=[6, 1], x_start=TS_X) for kind in
             ('anchor', 'reference', 'candidate')]
    items[2].update(b_r=[[0]], perms=[[0]], b_predicted=[[0]], delta_ea_xtb_kcal=-0.6)
    write_json(run / 'o2-plan.json', dict(items=items, input_sha256={}))
    o2 = run / 'o2-run'
    write_json(o2 / 'anchor_anc1' / 'verdict.json', dict(status='MINIMUM_PASS', energy_hartree=-1.5))
    write_json(o2 / 'reference_ref1' / 'verdict.json', dict(strict_status='STRICT_JOINT_GRAPH_VALID',
                                                             ts=dict(energy_hartree=-1.1), irc=dict(status='IRC_COMPLETE')))
    write_json(o2 / 'candidate_can1' / 'verdict.json', dict(
        strict_status='PROTOCOL_FAILURE', irc=dict(status='IRC_LIMIT'),
        ts=dict(status='TS_OPTFREQ_PASS', energy_hartree=-1.0, x_ts=TS_X, frequencies_cm1=[-500.0, 100.0])))
    trajectory(o2 / 'candidate_can1' / 'irc' / 'irc_irc.xyz')
    config = json.loads((ROOT / ie.CONFIG).read_text(encoding='utf-8'))
    config.update(plan_sha256=file_hash(run / 'o2-plan.json'), items=[dict(
        item_id='can1', kind='candidate', parent_id=pid, o2_verdict_sha256=file_hash(o2 / 'candidate_can1' / 'verdict.json'),
        irc_xyz_sha256=file_hash(o2 / 'candidate_can1' / 'irc' / 'irc_irc.xyz'))])
    write_json(tmp / 'config.json', config)
    out = run / ie.OUT
    write_json(out / 'ledger.json', [dict(item_id='can1', job_id='9', outcome='VERDICT', runtime_s=300)])
    end_x = np.array(TS_X) + [[0, 0, 0], [0, 0, 0.5]]  # equals the last frame
    write_json(out / 'candidate_can1' / 'verdict.json', dict(
        diagnostic_status='STRICT_JOINT_GRAPH_VALID', identity=dict(status='STRICT_JOINT_GRAPH_VALID', reactant_end=0),
        local_barrier_kcal=0.25 * K, wall_s=200.0, stage_timings=[dict(stage='end_first_opt', kind='scf', wall_s=1.0)],
        endpoints=[dict(status='MINIMUM_PASS', x=TS_X, energy_hartree=-1.25, frequencies_cm1=[30.0, 5.0, 40.0, 50.0]),
                   dict(status='MINIMUM_PASS', x=end_x.tolist(), energy_hartree=-1.16, frequencies_cm1=[20.0],
                        first_attempt={})]))
    return run, tmp / 'config.json'


def test_summarize_reports_ts_difference_and_end_descent(tmp_path):
    run, config = build(tmp_path)
    out = ie.summarize(run, config)
    row = out['rows'][0]
    assert row['diagnostic_status'] == 'STRICT_JOINT_GRAPH_VALID' and row['passes']
    assert row['ts_energy_difference_vs_reference_kcal'] == pytest.approx(0.1 * K)
    assert row['candidate_ts_vs_anchor_kcal'] == pytest.approx(0.5 * K)
    assert row['reference_ts_vs_anchor_kcal'] == pytest.approx(0.4 * K)
    first, last = row['ends']
    assert first['drop_from_frame_kcal'] == pytest.approx(0.05 * K)
    assert first['rmsd_frame_to_minimum_angstrom'] == pytest.approx(0.2)  # 1.1 -> 1.5 A bond, centred
    assert first['lowest_frequencies_cm1'] == [5.0, 30.0, 40.0] and not first['retried']
    assert last['rmsd_frame_to_minimum_angstrom'] == pytest.approx(0.0, abs=1e-9) and last['retried']
    assert row['ts_rmsd_xtb_to_dft_angstrom'] == pytest.approx(0.0, abs=1e-9)
    assert row['slurm']['runtime_s'] == 300 and out['gpu']['within_budget']
    assert ie.irc_limit_items(run, json.loads((run / 'o2-plan.json').read_text())) == ['can1']
    write_json(tmp_path / 'a.json', out)
    write_json(tmp_path / 'b.json', ie.summarize(run, config))
    assert (tmp_path / 'a.json').read_bytes() == (tmp_path / 'b.json').read_bytes()


def test_frozen_config_and_job_command():
    cfg = json.loads((ROOT / ie.CONFIG).read_text(encoding='utf-8'))
    assert [e['parent_id'][:4] for e in cfg['items']] == ['4f55', '7dc6']
    assert len(cfg['items']) * cfg['execution']['reserve_s'] <= cfg['execution']['budget_s']
    wrap = ie.item_wrap(PurePosixPath('/src'), PurePosixPath('/runs/r'), 'abc')
    assert wrap.startswith('cd /src;') and ie.PYTHON in wrap and wrap.endswith('run --run-root /runs/r --item abc')
