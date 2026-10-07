"""Measured-development resource projection, distinct from observed execution."""
import argparse
import json
from pathlib import Path
import shutil

from xtbflow.v1.data import file_hash,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-root',type=Path,required=True)
    ap.add_argument('--parents',type=int,default=300);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();root=a.run_root
    sources={'A0':'streams_a015_early','A1':'streams_a030_early','A2':'streams_a030_early',
             'B0':'streams_sync_a015','B1':'streams_sync_a030'}
    manifests={name:json.loads((root/'development'/name/'manifest.json').read_text()) for name in set(sources.values())}
    dev=manifests[sources['A0']]['n_queries'];factor=a.parents/dev
    if any(not m['complete'] or m['n_queries']!=dev for m in manifests.values()):raise ValueError('incomplete timing source')
    arm_times={arm:manifests[name]['gpu_wall_s'][arm] for arm,name in sources.items()}
    efficiency=sum(arm_times.values())*factor*3
    pulse=[]
    for seed in (0,1,2):
        p=root/'development'/f'selected_pulse_s{seed}'/'manifest.json'
        pulse.append(json.loads(p.read_text())['gpu_wall_s'])
    controls=json.loads((root/'development/continuous_controls/manifest.json').read_text())
    mechanism=(sum(pulse)+controls['gpu_wall_s'])*factor
    rarity=arm_times['B0']*factor*128/64
    training={}
    for p in sorted((root/'generators').glob('*/status.json')):
        status=json.loads(p.read_text());training[p.parent.name]=status
    scores={p.parent.name:json.loads(p.read_text()) for p in sorted((root/'scores').glob('*/status.json'))}
    evaluation=json.loads((root/'evidence/streams_a015_early.json').read_text())
    stream_cpu=evaluation['matcher_cpu_wall_s']*factor*3
    pulse_cpu=sum(json.loads((root/'evidence'/f'selected_pulse_s{s}.json').read_text())['matcher_cpu_wall_s'] for s in (0,1,2))*factor
    control_cpu=json.loads((root/'evidence/continuous_controls.json').read_text())['matcher_cpu_wall_s']*factor
    n_candidates=sum(sum(v['completed_candidates'][-1] for v in arm['parent_rows']) for arm in evaluation['summaries'].values())
    rarity_cpu=evaluation['matcher_cpu_wall_s']/n_candidates*a.parents*128
    disk_bytes=0
    for name in manifests:
        for line in (root/'development'/name/'streams.jsonl').open():
            r=json.loads(line)
            if sources.get(r['arm'])==name:disk_bytes+=len(line.encode())
    projected_stream_bytes=disk_bytes*factor*3
    total_gpu=(efficiency+mechanism+rarity)/3600
    report=dict(schema='v1a-resource-forecast/1',planning_parents=a.parents,development_parents=dev,
        device='NVIDIA RTX PRO 6000 Blackwell',available_gpu_count=4,
        measured_development_gpu_wall_s_by_arm=arm_times,
        measured_development_pulse_gpu_wall_s_by_seed=pulse,
        measured_development_continuous_control_gpu_wall_s=controls['gpu_wall_s'],
        observed_training_status=training,observed_score_status=scores,
        observed_generator_gpu_hours_completed=sum(v['elapsed_s'] for v in training.values() if v['status']=='complete')/3600,
        training_gpu_hours_remaining=0. if len(training)==10 and all(v['status']=='complete' for v in training.values()) else None,
        projected_efficiency_gpu_hours=efficiency/3600,projected_mechanism_gpu_hours=mechanism/3600,
        projected_rarity_gpu_hours=rarity/3600,projected_total_gpu_hours=total_gpu,
        projected_matcher_cpu_hours=(stream_cpu+pulse_cpu+control_cpu+rarity_cpu)/3600,
        projected_efficiency_stream_gb=projected_stream_bytes/(1024**3),
        free_disk_gb=shutil.disk_usage(root).free/(1024**3),quantum_cpu_core_hours_v1a=0,
        projected_wall_hours_at_one_gpu=total_gpu,ideal_gpu_only_wall_hours_at_four_gpus=total_gpu/4,
        scheduler_wait_prediction=None,cost_table_sha256=file_hash(root/'evidence/cost_calibration_512.json'),
        source_manifests={name:file_hash(root/'development'/name/'manifest.json') for name in manifests},
        forecast_basis='Measured seed-0 development time, scaled by parents and three training seeds. Includes diagnostic gradients and paired drift shadows; paired formal prefixes can reduce this cost. Excludes queue delays.',
        proposed_screen_gpu_job_limit_hours=1,proposed_total_screen_gpu_cap_hours=5,
        note='Projection, not executed formal cost; rerun with final frozen N.')
    write_json(a.out,report)
    print(json.dumps({k:v for k,v in report.items() if k.startswith('projected_') or k=='training_gpu_hours_remaining'},indent=2))


if __name__=='__main__':main()
