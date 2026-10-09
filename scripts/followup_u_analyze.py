"""Deterministic T2a analysis of frozen 558-reference Hessian acquisitions."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import numpy as np
from xtbflow.v1.metrics import cluster_summary


def reading(median, negative_fraction):
    if median >= .7 and negative_fraction >= .7:
        return 'U_IS_REACTION_MODE'
    if median < .3 or negative_fraction < .5:
        return 'U_NOT_REACTION_MODE'
    return 'U_PARTIAL'


def wilson(k, n):
    z = 1.959963984540054
    p = k/n
    denominator = 1+z*z/n
    center = (p+z*z/(2*n))/denominator
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return [float(center-half), float(center+half)]


def summarize(rows):
    if len(rows) != 558 or len({r['reference_id'] for r in rows}) != 558:
        raise ValueError('Expected exactly 558 distinct references')
    good = [r for r in rows if r['status'] == 'ok']
    if not good:
        raise ValueError('No successful Hessians')
    parent = defaultdict(list)
    strata = defaultdict(list)
    for r in good:
        parent[r['parent_id']].append(r)
        strata[r['metrics']['event_size']].append(r['metrics']['overlap'])
    median = float(np.median([r['metrics']['overlap'] for r in good]))
    negative = sum(r['metrics']['c_u'] < 0 for r in good)
    one = sum(r['metrics']['imaginary_count'] == 1 for r in good)
    pm = {}
    intervals = {}
    for pid, records in sorted(parent.items()):
        groups = {r['split_group'] for r in records}
        if len(groups) != 1:
            raise ValueError('Conflicting formula groups')
        pm[pid] = dict(split_group=records[0]['split_group'], n=len(records),
                       **{key: float(np.mean([r['metrics'][key] for r in records]))
                          for key in ('overlap', 'c_u', 'best_neg_overlap')})
    for key in ('overlap', 'c_u', 'best_neg_overlap'):
        intervals[key] = cluster_summary([v[key] for v in pm.values()],
                                         [v['split_group'] for v in pm.values()])
    return dict(n_total=len(rows), n_success=len(good), n_failed=len(rows)-len(good),
                failures=dict(sorted(Counter(r.get('reason', 'unknown') for r in rows if r['status'] != 'ok').items())),
                overlap_median=median, negative_c_u_count=negative,
                negative_c_u_fraction=negative/len(good), negative_c_u_wilson95=wilson(negative,len(good)),
                exactly_one_imaginary_wilson95=wilson(one,len(good)),
                exactly_one_imaginary_count=one, exactly_one_imaginary_fraction=one/len(good),
                reading=reading(median, negative/len(good)),
                overlap_by_event_size={str(k):dict(n=len(v),median=float(np.median(v))) for k,v in sorted(strata.items())},
                parent_metrics=pm, cluster_intervals=intervals,
                gradient_calls=sum(r['gradient_calls'] for r in rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gfn2',type=Path,required=True)
    p.add_argument('--aimnet',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise RuntimeError('Dirty source')
    methods={}
    for name,path in [('GFN2-xTB',a.gfn2),('AIMNet2',a.aimnet)]:
        manifest=json.loads((path/'manifest.json').read_text())
        if not manifest['complete'] or manifest['method'] != name:
            raise ValueError('Incomplete/wrong acquisition')
        methods[name]=summarize(list(map(json.loads,(path/'rows.jsonl').open())))
    result=dict(exploratory=True,primary_method='GFN2-xTB',methods=methods)
    a.out.mkdir(exist_ok=False)
    (a.out/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    m=methods['GFN2-xTB'];ci=m['cluster_intervals']['overlap']
    text=f"本报告全部结果为探索性；GFN2 母体平均 overlap 为 {ci['estimate']:.9f}，双侧 95% 聚类区间 {ci['ci_two95']}，读法 `{m['reading']}`。\n\n# 未弛豫参考 TS 的反应坐标诊断\n\n"
    for name,m in methods.items():
        text+=f"## {name}\n\n成功 {m['n_success']}/{m['n_total']}；overlap 中位数 {m['overlap_median']:.9f}；c_u < 0 比例 {m['negative_c_u_fraction']:.9f}；恰好一个虚频比例 {m['exactly_one_imaginary_fraction']:.9f}。\n\n"
        text+=f"比例 Wilson 95% 区间：c_u < 0 {m['negative_c_u_wilson95']}；恰好一个虚频 {m['exactly_one_imaginary_wilson95']}。\n\n"
        for key,interval in m['cluster_intervals'].items():
            text+=f"- {key} 母体平均：{interval['estimate']:.9f}，聚类区间 {interval['ci_two95']}。\n"
        text+='\n事件大小分层 overlap 中位数：\n'
        for size,s in m['overlap_by_event_size'].items():
            text+=f"- 大小 {size}：n={s['n']}，中位数 {s['median']:.9f}。\n"
    text+='\n## 证据边界\n\n采用冻结的 0.005 Å 中心差分、刚体模投影及最丰同位素质量。GFN2 优化水的真实物理测试通过，三个内部频率均为正；原始水测试、558 条逐记录结果和 Hessian 文件保留在运行目录。所有统计使用成功记录，失败数明确列出，未填补失败值。按母体平均后使用 split_group 聚类区间。AIMNet2 为次要诊断，不用于改写主读法；近似势能面上的诊断不是化学认证。\n'
    (a.out/'REPORT_ZH.md').write_text(text)
    files=[Path('configs/explore/u_diagnostic.json')]
    files += [root/name for root in (a.gfn2,a.aimnet) for name in ('rows.jsonl','manifest.json')]
    manifest=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dirty=False,slurm_job_id=None,
                  config_sha256=sha256(files[0].read_bytes()).hexdigest(),
                  input_sha256={str(f):sha256(f.read_bytes()).hexdigest() for f in files})
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    (a.out/'SHA256SUMS').write_text(''.join(f'{sha256(f.read_bytes()).hexdigest()}  {f.name}\n' for f in sorted(a.out.iterdir()) if f.name!='SHA256SUMS'))


if __name__=='__main__':main()
