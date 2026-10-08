"""Assemble bounded repository evidence without copying tensors or large rows."""
import argparse
import json
from pathlib import Path
import shutil
import hashlib


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dest',type=Path,required=True);a=p.parse_args()
    run=a.run; dest=a.dest;dest.mkdir(parents=True,exist_ok=True)
    r=json.loads((run/'results.json').read_text())
    def fmt(v):
        ci=v.get('ci_two95')
        return f"{v['estimate']:+.6f} [{ci[0]:+.6f}, {ci[1]:+.6f}]" if ci else str(v['estimate'])
    primary=r['comparisons']['relaxed_0.7-none']['H'];random=r['comparisons']['relaxed_0.7-random_0.7']['H']
    lines=[f"本报告全部结果为探索性。主臂 relaxed_0.7 的 H 对 none 差值为 {fmt(primary)}，对 random_0.7 为 {fmt(random)}（双侧95%分子式聚类区间）。",'',
        '# 生成中物理选择（smc-1）','',f"主读法：**{r['readings']['relaxed_0.7']}**。H 为 hit@1、hit@2、hit@4 均值；代理命中不代表认证的过渡态。",'',
        '## 检查点响应与消融','', '| s | relaxed-none ΔH [95% CI] | relaxed-random ΔH | relaxed-raw ΔH | 读法 |','|---|---|---|---|---|']
    for s in (.6,.7,.8,.9):
        arm=f'relaxed_{s:.1f}';raw=f'raw_{s:.1f}';rnd=f'random_{s:.1f}'
        lines.append(f"| {s:.1f} | {fmt(r['comparisons'][arm+'-none']['H'])} | {fmt(r['comparisons'][arm+'-'+rnd]['H'])} | {fmt(r['comparisons'][arm+'-'+raw]['H'])} | {r['readings'][arm]} |")
    lines += ['', '## 绝对指标与诊断','', '| 臂 | H | hit@∞ | 合法率 | 不同合法事件数 |','|---|---|---|---|---|']
    for arm,m in sorted(r['arms'].items()):lines.append(f"| {arm} | {fmt(m['H'])} | {fmt(m['hit_infinity'])} | {fmt(m['legal_rate'])} | {fmt(m['distinct_legal_events'])} |")
    lines += ['', '| 臂 | 误剪率 | 克隆事件改变率 | 仅克隆命中最佳 |','|---|---|---|---|']
    for arm,d in sorted(r['diagnostics'].items()):lines.append(f"| {arm} | {fmt(d['false_prune'])} | {fmt(d['clone_event_change'])} | {fmt(d['clone_only_best'])} |")
    s0=r['reproduction']['s0'];c1=r['reproduction']['c1'];c2=r['reproduction']['c2']
    lines += ['', '## 复现门禁与评分器','',
        f"S0：{s0['n']} 条真实 TS；raw 最大误差 {s0['raw_max_error']} kcal/mol，rho={s0['raw_rho']['estimate']}。relaxed rho={fmt(s0['relaxed_rho'])}；完整可用母体数 {s0['relaxed_rho']['n_parents']}。",
        f"真实 TS 成功负曲率比例 {s0['negative_curvature_fraction']:.6f}，按冻结分支关闭曲率分层。relaxed 失败 {s0['statistics']['failures']}；降能分位数见 results.json，负值表示弛豫升能，而非报告符号错误。",
        f"C1：{c1['n']} 条 none 事件/status，差异 {c1['event_status_mismatches']}；检查点 raw 最大有限值误差 {c1['raw_max_error']} kcal/mol。两边共同的 raw 失败数量见 coverage_audit.json；不能把失败称为成功复现能量。",
        '', '| 臂 | 幸存者数 | 事件一致率 | max |Δx| Å |','|---|---|---|---|']
    for arm,s in sorted(c2['survivor_checks'].items()):lines.append(f"| {arm} | {s['n']} | {s['agreement']:.9f} | {s['max_abs_dx']:.9g} |")
    lines += ['', '## 原始证据与复算','',
        f"原始输出留在 n2 `{run}`；张量和逐候选数据不进入仓库。SHA256SUMS 包含原始输出及日志哈希。各阶段 manifest 保存已推送源码、干净状态、配置、输入和 Slurm 作业号。",
        '```bash',f'cd {run.parent}/source-2116d43',
        'export CUDA_VISIBLE_DEVICES= PYTHONPATH=src:vendor/mechai_reusable OMP_NUM_THREADS=1',
        f'nice -n 10 /home/lhshen/miniconda3/envs/xtbflow/bin/python scripts/explore_smc.py analyze --out {run} --workers 1',
        '```','analyze-repeat.sha256 与逐字节 cmp 记录验证确定性。独立 coverage_audit.json 核查完整种群、方案及复现，不代替化学验证。','',
        '## 偏差、资源和局限','',
        '- 开发集已多次使用；检查点 0.7 和时钟 lead3 都用过开发集结果，不具有独立确认性。',
        '- 闭世界指标把目录外通道算作未命中；xTB 与 ωB97X 的势垒排序不一致。',
        '- S0 relaxed 相关只在该母体全部参考记录成功时计入；检查点相关只用三个 seed 均可估计的母体。缺失引入选择偏差。',
        '- 每次 relaxed 额外计算初始 raw 能量用于诊断，因此成功候选通常为24次调用而非约23次；没有更改力、步数或打分排序。',
        '- 共享前缀和 none 基线为实际复用；幸存者仍按要求重新续跑。逻辑每臂步数相同，实际 GPU 总成本不等于分别从零跑13臂。',
        '- CPU 轨迹冒烟只覆盖 G1，并非全流程；S0/C1/C2 正式门禁和独立覆盖检查补充验证，但不能倒称已经做过完整冒烟。',
        '- Slurm accounting 未启用。GPU 墙钟以 scontrol 为准；CPU 未全部采集进程核时，采用评分累计耗时及分配核数×墙钟上界，不冒充实测 CPU 总核时。',
        '- 所有资源数字和评分层计数见 results.json 的 resource_manifests/scorer_statistics；C2 和分析另有 time -v 日志。',
        '- 不做 DFT，不用 V1b/screen，不修改已有科学模块，不合并 PR。']
    (dest/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    for name in ['results.json','coverage_audit.json']+[f'{s}_manifest.json' for s in ('s0','g1','c1','g2','c2')]:
        src=run/name
        if src.stat().st_size>1_000_000:raise ValueError(f'oversized evidence: {name}; keep on n2')
        shutil.copyfile(src,dest/name)
    sums=[]
    for f in sorted(run.iterdir()):
        if f.is_file():sums.append(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+str(f))
    for name in ('pytest.log','unittest.log','slurm-2699.txt','slurm-2702.txt','c2-time.txt','analyze-time.txt','analyze-repeat.sha256'):
        f=run.parent/name
        if f.exists():sums.append(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+str(f))
    (dest/'SHA256SUMS').write_text('\n'.join(sums)+'\n')


if __name__=='__main__':main()
