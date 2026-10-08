"""Task N: NFE-matched comparison of the step-count control arms (exploratory).

Inputs are the sealed V1a efficiency reports (A0-50+50, B0-50), the task R
compact streams for their candidate order, and the new control runs
(A0-25+25, B0-25) evaluated by scripts/v1a_stream_evaluate.py. Writes
nfe_result.json and nfe_tables_ZH.md into a new directory.
"""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

from xtbflow.v1.data import file_hash
from xtbflow.v1.formal import load_freeze
from xtbflow.v1.metrics import cluster_summary
from xtbflow.v1.reanalysis import count_prefix,paired_difference

CURVE_KEYS=('hits','event_hits','completed_candidates','valid_candidates')
ENDPOINTS={'joint_best_reference':'joint','event_best':'event'}
STATEMENT=('探索性分析：V1a 正式集已揭盲一次，本目录任何结果都不是确认性证据；'
           'V1a 正式决策 GO_V1b 不变。')


def sealed_hashes(path):
    return {name:digest for digest,name in (l.split() for l in Path(path).read_text().splitlines() if l.strip())}


def cost_bin(table,n):
    for row in table['bins']:
        if row['min_atoms']<=n<=row['max_atoms']:return row
    raise ValueError(n)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--v1a-root',type=Path,required=True)
    ap.add_argument('--sealed',type=Path,required=True,help='docs/evidence/v1a/formal/SHA256SUMS')
    ap.add_argument('--r-manifest',type=Path,required=True,help='docs/evidence/v1a_r/extract_manifest.json')
    ap.add_argument('--r-streams',type=Path,required=True,help='task R extract streams.jsonl.gz')
    ap.add_argument('--new-root',type=Path,required=True,help='directory holding nfe_s{seed}[.json|.candidates.jsonl]')
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(a.config.read_text());seeds=cfg['training_seeds'];root=a.v1a_root
    sealed=sealed_hashes(a.sealed);checks={}

    def check(name,ok):
        checks[name]=bool(ok)
        if not ok:raise ValueError(f'provenance check failed: {name}')
    check('freeze',file_hash(root/'formal/freeze.json')==sealed['formal/freeze.json'])
    freeze=load_freeze(root/'formal/freeze.json');order=sorted(freeze['parent_ids'])
    check('cost_table',file_hash(cfg['cost_table']['path'])==cfg['cost_table']['sha256']==freeze['bindings']['cost_table_sha256'])
    rman=json.loads(a.r_manifest.read_text())
    check('r_streams',file_hash(a.r_streams)==rman['streams_gz_sha256'])
    parents=rman['parents'];groups=[parents[p]['split_group'] for p in order]

    reports={'A0-50+50':{},'B0-50':{},'A0-25+25':{},'B0-25':{}};new_meta={}
    for s in seeds:
        rel=f'screen/efficiency_s{s}.json'
        check(rel,file_hash(root/rel)==sealed[rel])
        rep=json.loads((root/rel).read_text())
        reports['A0-50+50'][s]=rep['summaries']['A0'];reports['B0-50'][s]=rep['summaries']['B0']
        new=json.loads((a.new_root/f'nfe_s{s}.json').read_text())
        man=json.loads((a.new_root/f'nfe_s{s}/manifest.json').read_text())
        expected={k:v for arm in cfg['new_arms'].values() for k,v in arm.items() if k.startswith('nfe_')}
        check(f'nfe_s{s}.binding',new['freeze_sha256']==freeze['freeze_sha256'] and new['training_seed']==s
              and man['nfe_overrides']==expected and man['config']['namespace']==cfg['noise_namespace']
              and man['cost_source_sha256']==cfg['cost_table']['sha256'] and sorted(man['arms'])==['A0','B0']
              and file_hash(a.new_root/f'nfe_s{s}/streams.jsonl')==new['source_sha256']==man['streams_sha256'])
        reports['A0-25+25'][s]=new['summaries']['A0'];reports['B0-25'][s]=new['summaries']['B0']
        new_meta[str(s)]=dict(gpu_wall_s=man['gpu_wall_s'],streams_sha256=man['streams_sha256'],
                              report_sha256=file_hash(a.new_root/f'nfe_s{s}.json'),
                              candidates_sha256=file_hash(a.new_root/f'nfe_s{s}.candidates.jsonl'),
                              attempt_status_counts=new['attempt_status_counts'])
    rows={arm:{s:{r['parent_id']:r for r in rep[s]['parent_rows']} for s in seeds} for arm,rep in reports.items()}
    for arm in rows:
        for s in seeds:check(f'{arm}.s{s}.parents',sorted(rows[arm][s])==order)

    curves={}
    for arm in rows:
        curves[arm]={k:np.mean([[rows[arm][s][p][k] for p in order] for s in seeds],axis=(0,1)).tolist() for k in CURVE_KEYS}
        for k in ('auc','event_auc'):curves[arm][k]=float(np.mean([[rows[arm][s][p][k] for p in order] for s in seeds]))

    def auc_contrast(x,y,key='auc'):
        d=[float(np.mean([rows[x][s][p][key]-rows[y][s][p][key] for s in seeds])) for p in order]
        return cluster_summary(d,groups)
    primary=auc_contrast(*cfg['primary']['contrast'])
    holds=primary['ci_two95'][0]>0
    interp=cfg['primary']['interpretation']['lower_bound_above_zero' if holds else 'otherwise']
    descriptive={f'{x}-{y}':dict(auc=auc_contrast(x,y),event_auc=auc_contrast(x,y,'event_auc'))
                 for x,y in cfg['descriptive']['auc_contrasts']}

    # Candidate order for count-aligned comparisons.
    seq={}
    with gzip.open(a.r_streams,'rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['arm'] in ('A0','B0'):
                seq[{'A0':'A0-50+50','B0':'B0-50'}[r['arm']],r['parent_id'],r['training_seed']]=dict(joint=r['joint'],event=r['event'])
    for s in seeds:
        grouped=defaultdict(list)
        for line in (a.new_root/f'nfe_s{s}.candidates.jsonl').open():
            r=json.loads(line);grouped[r['arm'],r['parent_id']].append(r)
        for (arm,p),cc in grouped.items():
            cc.sort(key=lambda r:r['completion_units'])
            seq[{'A0':'A0-25+25','B0':'B0-25'}[arm],p,s]=dict(joint=[bool(r['hits_best_reference']) for r in cc],
                                                             event=[bool(r['hits_best_event']) for r in cc])
    secondary={}
    for x,y in cfg['secondary']['count_aligned_pairs']:
        item={}
        for endpoint,field in ENDPOINTS.items():
            item[endpoint]={}
            for n in cfg['secondary']['count_grid']:
                cells={}
                for arm in (x,y):
                    c={}
                    for p in order:
                        for s in seeds:
                            v=seq.get((arm,p,s),dict(joint=[],event=[]))[field];m=count_prefix(len(v),n)
                            c[p,s]=None if m is None else float(any(v[:m]))
                    cells[arm]=c
                summary,coverage,_=paired_difference(cells[x],cells[y],order,groups,seeds)
                item[endpoint][str(n)]=dict(summary=summary,parent_coverage=coverage,
                    coverage_insufficient=coverage<cfg['secondary']['count_coverage_min_parents'])
        secondary[f'{x}-{y}']=item

    table=json.loads(Path(cfg['cost_table']['path']).read_text());bins=defaultdict(list)
    for p in order:
        b=cost_bin(table,parents[p]['n_atoms']);w=b['weights']
        cascade=cfg['new_arms']['A0-25+25']['nfe_event']*w['g']+cfg['new_arms']['A0-25+25']['nfe_geometry']*w['h']
        bins[f"{b['min_atoms']}-{b['max_atoms']}"].append(dict(a0_25_25=cascade,b0_50=50.*w['f'],b0_25=25.*w['f']))
    cost=[dict(atom_bin=k,n_parents=len(v),a0_25_25_units=v[0]['a0_25_25'],b0_50_units=v[0]['b0_50'],
               ratio=v[0]['a0_25_25']/v[0]['b0_50'],
               differs_over_10pct=abs(v[0]['a0_25_25']/v[0]['b0_50']-1)>cfg['cost_mismatch_report_fraction'])
          for k,v in sorted(bins.items(),key=lambda kv:int(kv[0].split('-')[0]))]
    weighted=float(np.average([c['ratio'] for c in cost],weights=[c['n_parents'] for c in cost]))
    res=dict(schema='xtbflow-v1a-n-result/1',exploratory=True,statement=STATEMENT,config_version=cfg['version'],
             config_sha256=file_hash(a.config),checks=checks,n_parents=len(order),n_formula_groups=len(set(groups)),
             curves=curves,primary=dict(contrast=cfg['primary']['contrast'],summary=primary,
                                       lower_bound_above_zero=bool(holds),interpretation=interp),
             descriptive=descriptive,secondary=secondary,
             cost=dict(per_bin=cost,parent_weighted_mean_ratio_a0_25_25_over_b0_50=weighted),
             new_runs=new_meta,r_streams_sha256=rman['streams_gz_sha256'])
    (a.out/'nfe_result.json').write_text(json.dumps(res,indent=1,sort_keys=True,allow_nan=False)+'\n')

    def fmt(s):
        if s.get('estimate') is None:return '—'
        if 'ci_two95' not in s:return f"{s['estimate']:+.3f}"
        return f"{s['estimate']:+.4f} [{s['ci_two95'][0]:+.4f}, {s['ci_two95'][1]:+.4f}]"
    lines=['# V1a-N 自动生成表格（探索性）','',STATEMENT,'',
           '| 臂 | 召回 @4/8/16/32/64 | 完成候选 @4/8/16/32/64 | AUC | 事件 AUC |','|---|---|---|---:|---:|']
    for arm in ('A0-50+50','A0-25+25','B0-50','B0-25'):
        c=curves[arm]
        lines.append(f"| {arm} | {' / '.join(f'{v:.3f}' for v in c['hits'])} | "
                     f"{' / '.join(f'{v:.1f}' for v in c['completed_candidates'])} | {c['auc']:.4f} | {c['event_auc']:.4f} |")
    lines+=['',f"**主对比** AUC({cfg['primary']['contrast'][0]}) − AUC({cfg['primary']['contrast'][1]})：{fmt(primary)}；"
            f"区间下限 > 0：{holds}。预先写定的解读：{interp}",'','描述性对比（AUC / 事件 AUC）：']
    for k,v in descriptive.items():lines.append(f"- {k}：{fmt(v['auc'])} / {fmt(v['event_auc'])}")
    lines+=['','次要：同样候选数下的命中率差（括号为母体覆盖率）：','']
    for pair,item in secondary.items():
        for endpoint,row in item.items():
            lines.append(f"- {pair} × {endpoint}："+'；'.join(
                f"n={n} {fmt(r['summary'])} ({r['parent_coverage']:.0%}{' 覆盖不足' if r['coverage_insufficient'] else ''})"
                for n,r in row.items()))
    lines+=['','单候选成本（网络单位，冻结成本表）：','','| 原子箱 | 母体数 | A0-25+25 | B0-50 | 比值 | 差异>10% |','|---|---:|---:|---:|---:|---|']
    for c in cost:
        lines.append(f"| {c['atom_bin']} | {c['n_parents']} | {c['a0_25_25_units']:.2f} | {c['b0_50_units']:.0f} | "
                     f"{c['ratio']:.3f} | {'是' if c['differs_over_10pct'] else '否'} |")
    lines+=['',f'按母体加权的平均比值：{weighted:.3f}','']
    (a.out/'nfe_tables_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(primary=primary,holds=holds,checks=checks),indent=2))


if __name__=='__main__':main()
