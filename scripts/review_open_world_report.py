"""Render a Chinese exploratory report directly from the audited result JSON."""
import argparse
import csv
import json
from pathlib import Path


def interval(v):
    if v is None:return '不可估计'
    lo,hi=v['ci_two95'];return f"{v['success']}/{v['total']}（{100*v['rate']:.2f}%，95%区间 {100*lo:.2f}%–{100*hi:.2f}%）"


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=json.loads(a.results.read_text())
    if not r['complete']:raise ValueError('do not render an incomplete run as a final report')
    a.out.mkdir(parents=True,exist_ok=True)
    groups=r['groups'];outside=groups['OUTSIDE_CATALOGUE_EVENT']['certification']
    lines=['本报告全部结果为探索性。目录外组在 GFN2-xTB 层面的严格认证率为 '+interval(outside)+'；这不是 DFT 认证，也不改变 V1a 的结论。','',
        '# T3：开放世界终点第一阶段','','## 冻结样本与认证率','','|组别|严格认证率及 Wilson 区间|','|---|---|']
    for g in ('PROXY_MATCH','KNOWN_EVENT_GEOMETRY_MISS','OUTSIDE_CATALOGUE_EVENT','BASELINE'):
        lines.append('|'+g+'|'+interval(groups[g]['certification'])+'|')
    lines+=['','阳性对照闸门：'+('通过' if r['positive_control_passed'] else '失败；禁止解读目录外组')+'。配置先于任何量化计算提交，样本重新抽取与冻结配置逐字节一致。所有候选按哈希排序选取，每组内每母体最多一条；组间可能重复母体，因此组间差异不是独立随机试验。',
        '', '## 与同母体基准的势垒差','','只对候选和基准链均严格通过的母体比较 TS 电子能量差，单位 kcal/mol。',
        '- ΔEa_xTB < 0：'+interval(r['delta_ea']['below_zero']),
        '- ΔEa_xTB ≤ +5：'+interval(r['delta_ea']['at_most_plus5']),
        '- 目录外候选虽通过、但基准未通过而排除的母体数：'+str(len(r['delta_ea']['excluded_baseline_parents'])),
        '', '|母体|ΔEa_xTB kcal/mol|','|---|---:|']
    for row in r['delta_ea']['rows']:lines.append(f"|{row['parent_id']}|{row['delta_ea_kcal']:.6f}|")
    lines+=['','## 失败分布','','|组别|状态|数量|','|---|---|---:|']
    for g,data in sorted(groups.items()):
        for status,n in sorted(data['statuses'].items()):lines.append(f'|{g}|{status}|{n}|')
    lines+=['','优化或 IRC 达到步数上限、数值问题和图感知失败不能解释为通道不存在。严格认证仅要求一端为反应物、另一端为预测产物，并满足该 xTB 链的频率与收敛条件；不证明通道在更高层级方法中存在。',
        '', '## 算力、实现与复现','','全部为 CPU 直接运行（负责人追加授权），GPU 用量为零，没有提交 DFT 作业。每条链的墙钟和梯度调用见 verdicts.csv，汇总见下表。数值 Hessian 使用梯度中心差分并对称化，真实优化水分子的平动转动投影秩和正内部频率检查通过，证据见 hessian_check.json。',
        '', '|组别|墙钟秒合计|梯度调用合计|','|---|---:|---:|']
    for g,d in sorted(groups.items()):lines.append(f"|{g}|{d['wall_seconds']:.6f}|{d['gradient_calls']}|")
    lines+=['','纯函数 harmonic、classify_saddle、classify_minimum、endpoint_identity 直接来自 qc_protocol，没有改写，也未调用该模块的 DFT 引擎。数值设置、干净源码提交、配置及原始文件哈希在 manifest.json；完整环境见 pip-freeze.txt。',
        '', '```bash','export CUDA_VISIBLE_DEVICES= PYTHONPATH=src:vendor/mechai_reusable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1',
        'python scripts/review_open_world_analyze.py --root /home/lhshen/xtbflow-runs/review-align-20261009/open-621b900 --out results.json',
        '```','','原始链保留于 n2，逐条 verdict 哈希列于 SHA256SUMS.remote；分析重跑必须与提交的 results.json 字节一致。',
        '', '## 下一阶段','','STAGE2_PROPOSAL_ZH.md 提供至多十条新通道的 DFT 认证提案、样本排序规则和算力预估。这里只提交提案，没有运行授权，也没有运行。']
    (a.out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    with (a.out/'verdicts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['group','id','parent_id','status','wall_seconds','gradient_calls']);w.writeheader();w.writerows(r['verdicts'])


if __name__=='__main__':main()
