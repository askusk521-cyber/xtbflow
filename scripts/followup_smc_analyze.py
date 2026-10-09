"""Frozen smc-2 paired metrics, diagnostics, rankings and closure recommendation."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from followup_smc_oracle import old,SM,R,clean,guard_cpu,inputs,manifest,digest
from xtbflow.v1.smc import population_metrics
from xtbflow.v1.metrics import cluster_summary
from xtbflow.v1.geometry_information import within_parent_rho


def reading(none_ci,random_ci):
    low,high=none_ci;rl,rh=random_ci
    if low>0 and rl>0:return 'IN_GENERATION_SELECTION_HELPS'
    if low>0 and rl<=0<=rh:return 'NOT_PHYSICS_SPECIFIC'
    if low<=0<=high:return 'NO_GAIN_OVER_POSTHOC'
    if high<0:return 'SELECTION_HURTS'
    return 'UNCLASSIFIED_BY_FROZEN_TABLE'


def recommendation(labels):
    if 'IN_GENERATION_SELECTION_HELPS' in labels.values():return 'NEEDS_HELDOUT_CONFIRMATION'
    if labels['aimnet_0.7'] in ('NO_GAIN_OVER_POSTHOC','SELECTION_HURTS'):return 'LINE_CLOSURE_RECOMMENDED'
    return 'NO_FROZEN_RECOMMENDATION'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--new',type=Path,required=True);p.add_argument('--decoded',type=Path,required=True)
    p.add_argument('--scores',type=Path,nargs=3,required=True);p.add_argument('--checkpoints',type=Path,nargs=2,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    clean();guard_cpu();started=time.monotonic()
    gate=json.loads((a.decoded/'manifest.json').read_text())
    if not gate['gate_passed']:raise RuntimeError('STOP survivor gate')
    if a.out.exists():raise FileExistsError(a.out)
    parents,_,_=inputs()
    endpoint=json.loads((SM/'endpoint_rows.json').read_text())+json.loads((a.decoded/'new_rows.json').read_text())
    plans=json.loads((SM/'plans.json').read_text())+json.loads((a.new/'plans.json').read_text())
    arms=sorted({r['arm'] for r in endpoint});newarms=sorted({r['arm'] for r in plans if r['arm'].startswith(('aimnet_','gxtb_'))})
    pids=sorted({r['parent_id'] for r in endpoint});pop=defaultdict(list)
    for r in endpoint:pop[r['arm'],r['parent_id'],r['seed']].append(r)
    if len(arms)!=21 or len(pids)!=62 or len(pop)!=21*62*3:raise ValueError('population coverage')
    if any(len(v)!=32 or {r['proposal'] for r in v}!=set(range(32)) for v in pop.values()):raise ValueError('proposal coverage')
    def summary(values):
        ids=sorted(values)
        if len({parents[x]['split_group'] for x in ids})<2:
            return dict(estimate=float(np.mean(list(values.values()))) if values else None,ci_two95=None,n_parents=len(ids),status='INSUFFICIENT_GROUPS')
        return cluster_summary([values[x] for x in ids],[parents[x]['split_group'] for x in ids])
    rankers={};files=[SM/'endpoint_rows.json',a.decoded/'new_rows.json',a.decoded/'manifest.json',a.new/'plans.json']
    for path in a.scores:
        m=json.loads((path/'manifest.json').read_text());name=m['method'];f=path/'rows.jsonl'
        if not m['complete'] or digest(f)!=m['rows_sha256']:raise RuntimeError('STOP score hash')
        scores={(r['arm'],r['parent_id'],r['seed'],r['proposal']):r['score'] for r in old.rows(f)}
        if len(scores)!=21*62*3*32:raise ValueError('score coverage')
        metrics={k:population_metrics([dict(r,score=scores[(*k,r['proposal'])]) for r in rr],parents[k[1]]['best_channel_ids'],curvature_layers=False) for k,rr in pop.items()}
        keys=sorted(next(iter(metrics.values())))
        pm={arm:{key:{pid:float(np.mean([metrics[arm,pid,seed][key] for seed in (0,1,2)])) for pid in pids} for key in keys} for arm in arms}
        stats={arm:{key:summary(vals) for key,vals in mm.items()} for arm,mm in pm.items()};comparisons={};labels={};decomposition={}
        for arm in newarms:
            random='random_'+arm.split('_')[1]
            for other in ('none',random):
                differences={key:{pid:pm[arm][key][pid]-pm[other][key][pid] for pid in pids} for key in keys}
                comparisons[arm+'-'+other]={key:summary(v) for key,v in differences.items()}
                # H = availability - ranking loss, so delta H = delta availability - delta loss.
                availability=differences['hit_infinity'];delta_h=differences['H']
                loss={pid:availability[pid]-delta_h[pid] for pid in pids}
                decomposition[arm+'-'+other]=dict(availability=summary(availability),ranking_loss=summary(loss),delta_H=summary(delta_h),
                                                  parent_availability=availability,parent_ranking_loss=loss)
            labels[arm]=reading(comparisons[arm+'-none']['H']['ci_two95'],comparisons[arm+'-'+random]['H']['ci_two95'])
        rankers[name]=dict(arms=stats,parent_metrics=pm,comparisons=comparisons,readings=labels,decomposition=decomposition)
        files.extend([f,path/'manifest.json'])
    if set(rankers)!={'aimnet','gxtb','gfn2'}:raise ValueError('ranker coverage')
    planmap={(r['arm'],r['parent_id'],r['seed']):r for r in plans};diagnostics={}
    for arm in newarms:
        changed={};only={};pruned={}
        for pid in pids:
            cv=[];ov=[];pv=[]
            for seed in (0,1,2):
                nr={r['proposal']:r for r in pop['none',pid,seed]};rr=pop[arm,pid,seed]
                clones=[r for r in rr if r['is_clone']];surv=[r for r in rr if not r['is_clone']]
                cv.append(sum(r['channel_id']!=nr[r['ancestor']]['channel_id'] for r in clones)/len(clones))
                ov.append(float(any(r['hits_best_event'] for r in clones) and not any(r['hits_best_event'] for r in surv)))
                if any(r['hits_best_event'] for r in nr.values()):pv.append(float(not any(nr[j]['hits_best_event'] for j in planmap[arm,pid,seed]['survivors'])))
            changed[pid]=float(np.mean(cv));only[pid]=float(np.mean(ov))
            if pv:pruned[pid]=float(np.mean(pv))
        diagnostics[arm]=dict(clone_event_change=summary(changed),clone_only_best=summary(only),false_prune=summary(pruned))
    none={(r['parent_id'],r['seed'],r['proposal']):r for r in endpoint if r['arm']=='none'};correlations={}
    for path in a.checkpoints:
        rr=old.rows(path/'rows.jsonl');method=rr[0]['method'];files.extend([path/'rows.jsonl',path/'manifest.json'])
        for step in (30,35,40,45):
            seedvals={}
            for seed in (0,1,2):
                good=[r for r in rr if r['seed']==seed and r['step']==step and r['score']['status']=='ok' and none[r['parent_id'],seed,r['proposal']]['event_barrier_kcal'] is not None]
                seedvals[seed]=within_parent_rho([r['score']['barrier'] for r in good],[none[r['parent_id'],seed,r['proposal']]['event_barrier_kcal'] for r in good],[r['parent_id'] for r in good],min_n=4)
            common=set.intersection(*(set(v) for v in seedvals.values()))
            correlations[f'{method}_{step/50:.1f}']=summary({pid:float(np.mean([seedvals[s][pid] for s in (0,1,2)])) for pid in sorted(common)})
    result=dict(version='smc-2',exploratory=True,primary_arm='aimnet_0.7',primary_ranker='aimnet',rankers=rankers,
                diagnostics=diagnostics,checkpoint_correlations=correlations,survivor_gate=gate['survivor_checks'],
                recommendation=recommendation(rankers['aimnet']['readings']))
    a.out.mkdir();old.dump(a.out/'results.json',result)
    primary=rankers['aimnet'];comp=primary['comparisons'];s=comp['aimnet_0.7-none']['H'];t=comp['aimnet_0.7-random_0.7']['H']
    report=f"本报告全部结果为探索性；主臂对 none 的 ΔH = {s['estimate']:.9f}，双侧 95% 区间 {s['ci_two95']}，读法 `{primary['readings']['aimnet_0.7']}`。\n\n# smc-2：更换物理打分后的中途筛选\n\n"
    report+=f"主臂对 random_0.7 的 ΔH = {t['estimate']:.9f}，区间 {t['ci_two95']}。研究线建议为 `{result['recommendation']}`，只是建议，最终由负责人决定。\n\n"
    for method,data in rankers.items():
        report+=f"## 终点排序器 {method}\n\n|新臂|对 none ΔH [95% CI]|对同检查点 random ΔH [95% CI]|读法|\n|---|---|---|---|\n"
        for arm in newarms:
            s=data['comparisons'][arm+'-none']['H'];t=data['comparisons'][arm+'-random_'+arm.split('_')[1]]['H']
            report+=f"|{arm}|{s['estimate']:.9f} {s['ci_two95']}|{t['estimate']:.9f} {t['ci_two95']}|{data['readings'][arm]}|\n"
    report+='\n## 复现与边界\n\n原有十三臂只读复用，新臂续跑直接调用原 cmd_g2，保留批次布局及噪声地址。八个新臂的幸存者事件一致率与最大坐标差见 results.json 的 survivor_gate；四项复现门禁分别保留原始证据。所有二十一个臂在三个终点排序器下的指标、同母体聚类区间和配对差保留在 results.json。\n\n误剪、克隆事件改变及仅克隆命中最佳的诊断使用原 smc-1 定义。ΔH 分解为可用性变化减排序损失变化；逐母体分解和区间保留供独立复算。检查点相关性严格按各 seed 计算，再在共同母体上平均。没有进行保留集确认；代理事件命中不等于过渡态或化学认证。\n'
    (a.out/'REPORT_ZH.md').write_text(report)
    manifest(a.out/'manifest.json','analyze',started,files)
    (a.out/'SHA256SUMS').write_text(''.join(f'{digest(f)}  {f.name}\n' for f in sorted(a.out.iterdir()) if f.name!='SHA256SUMS'))


if __name__=='__main__':main()
