"""Freeze the V1a screen once, before any screen generation (Section 9.8 step 6).

Binds the passing power plan, the selected development configuration and its
evidence, all model/score checkpoints and the cost table by SHA-256, then writes
the frozen screen queries. Refuses to overwrite an existing freeze or queries.
"""
import argparse
import json
from pathlib import Path

from xtbflow.v1.data import file_hash,write_json
from xtbflow.v1.formal import build_freeze


def main():
    ap=argparse.ArgumentParser()
    for key in ('catalogue','run-root','power-plan','costs','baseline-report','controls-report','decision','out'):
        ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--drift-reports',type=Path,nargs=3,required=True,help='seed 0/1/2 A2+B1 development streams')
    ap.add_argument('--pulse-reports',type=Path,nargs=3,required=True,help='seed 0/1/2 development pulses at t*')
    ap.add_argument('--superseded-power-plan',type=Path,help='earlier guarded plan, bound for disclosure')
    a=ap.parse_args()
    queries=a.catalogue/'screen_queries.jsonl'
    if a.out.exists() or queries.exists():raise FileExistsError('freeze already written; the screen is single-shot')
    split=json.loads((a.catalogue/'split_manifest.json').read_text())
    plan=json.loads(a.power_plan.read_text())
    baseline=json.loads(a.baseline_report.read_text());controls=json.loads(a.controls_report.read_text())
    if baseline['status']!='BASELINE_DEVELOPMENT_PASS':raise ValueError('B0 baseline check not passed')
    if controls['status']!='CONTINUOUS_CONTROLS_PASS':raise ValueError('continuous replay controls not passed')
    config=dict(path='sync',alpha_x=.3,alpha_b=.25,guidance_start=.5,guidance_stop=.95,event_start=.5,
                event_stop=.95,a2_times=[.4,.7],cap_units=3200.,t_star=.5,amplitudes=[.05,.10,.20],
                rarity_proposals=128,support_rule='training-label 1%/99% quantiles +/-10 kcal/mol; member SD <= 2*scale')
    seeds=[]
    for path in a.drift_reports:
        r=json.loads(path.read_text());seeds.append(r.get('training_seed',0))
        if r['status']!='DEVELOPMENT_STREAM_AUDIT_PASS':raise ValueError('development stream audit missing')
        for key in ('path','alpha_x','guidance_start','guidance_stop','a2_times','cap_units'):
            if r['config'][key]!=config[key]:raise ValueError(f'{path}: development stream differs from freeze: {key}')
        for arm,d in r['drift'].items():
            if not d['diagnostic_pass']:raise ValueError(f'{path}: drift check failed for {arm}')
    pulse_seeds=[]
    for path in a.pulse_reports:
        r=json.loads(path.read_text());pulse_seeds.append(r['training_seed'])
        if r['integrity']!='COMPLETE_PAIRED_WINDOW_PASS' or r['path']!=config['path']:raise ValueError('pulse evidence')
        if [w['requested_t'] for w in r['windows']]!=[config['t_star']]:raise ValueError('pulse evidence not at t*')
    if sorted(seeds)!=[0,1,2] or sorted(pulse_seeds)!=[0,1,2]:raise ValueError('three development seeds required')
    if any(r['split_hash']!=split['split_hash'] for r in (plan,baseline,controls)):raise ValueError('split mismatch')
    checkpoints={}
    for path in sorted((a.run_root/'generators').glob('*/ckpt_60000.pt'))+sorted((a.run_root/'scores').glob('*/ckpt_20000.pt')):
        checkpoints[str(path.relative_to(a.run_root))]=file_hash(path)
    if len(checkpoints)!=16:raise ValueError('expected 10 generator and 6 score checkpoints')
    reserve=[json.loads(line) for line in (a.catalogue/'screen_reserve_queries.jsonl').read_text().splitlines()]
    parent_of={p['query_id']:p['parent_id'] for p in json.loads((a.catalogue/'parent_catalog.json').read_text())}
    bindings=dict(cost_table_path=str(a.costs.resolve()),cost_table_sha256=file_hash(a.costs),
                  power_plan_sha256=file_hash(a.power_plan),checkpoints=checkpoints,
                  catalogue={n:file_hash(a.catalogue/n) for n in ('split_manifest.json','parent_catalog.json',
                             'reference_catalog.jsonl','screen_reserve_queries.jsonl','development_queries.jsonl')},
                  development_evidence={str(p):file_hash(p) for p in
                      [a.baseline_report,a.controls_report,*a.drift_reports,*a.pulse_reports]},
                  superseded_power_plan_sha256=None if a.superseded_power_plan is None else file_hash(a.superseded_power_plan),
                  decision_sha256=file_hash(a.decision))
    decision=dict(text=a.decision.read_text(encoding='utf-8'),planning_guard_used=plan['planning_guard'])
    freeze,rows=build_freeze(power_plan=plan,split=split,reserve_rows=reserve,parent_of=parent_of,
                             config=config,bindings=bindings,decision=decision)
    with queries.open('x',encoding='utf-8',newline='\n') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    write_json(a.out,freeze)
    print(json.dumps(dict(freeze_sha256=freeze['freeze_sha256'],N=freeze['n_parents'],G=freeze['n_formula_groups'],
                          queries_sha256=file_hash(queries)),indent=2))


if __name__=='__main__':main()
