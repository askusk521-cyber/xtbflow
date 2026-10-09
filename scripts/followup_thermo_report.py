"""Render T1b Chinese report/recall plot from independently replayed results.
All scientific numbers are read directly from results.json. No quantum calls.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess


def report_text(r):
    c=r['comparisons'][r['primary']];s=c['summary'];lo,hi=s['ci_two95']
    lines=[f"本报告全部结果为探索性；主比较 AUC(S_N_thermo) − AUC(B0) = {s['estimate']:.9f}，双侧 95% 区间 [{lo:.9f}, {hi:.9f}]，读法为 `{c['reading']}`。",
           '', '# T1b：产物热力学筛选枚举基线', '',
           f"使用全部 {r['n_parents']} 个冻结 screen 母体；训练集冻结阈值 τ = {r['tau_kcal']:.9f} kcal/mol。没有使用 M0 官方 val/test，没有重新生成生成侧候选。",
           '', '## 配对比较', '', '| 比较 | AUC 差 | 双侧 95% 区间 | 读法 |', '|---|---:|---|---|']
    for name,c in sorted(r['comparisons'].items()):
        s=c['summary'];lo,hi=s['ci_two95']
        lines.append(f"| {name} | {s['estimate']:.9f} | [{lo:.9f}, {hi:.9f}] | {c.get('reading','')} |")
    c=r['conditional_best_in_b2f2'];s=c['summary'];lo,hi=s['ci_two95']
    lines.extend(['','## 覆盖率与条件分析','',
                  f"b2f2 最佳通道覆盖率：{r['best_coverage']:.9f}；目录通道覆盖率的母体均值：{r['catalogue_coverage_parent_mean']:.9f}。",
                  f"只在最佳通道属于 b2f2 的母体上，主比较为 {s['estimate']:.9f}，区间 [{lo:.9f}, {hi:.9f}]，`{c['reading']}`。",
                  f"b2f2 事件总数 {r['n_b2f2']['total']}，每母体中位数 {r['n_b2f2']['median']}，95% 分位 {r['n_b2f2']['p95']}。",
                  '', '最佳事件不属于 b2f2、但生成臂曾命中的母体数：'])
    for arm,count in sorted(r['generator_hit_outside_best_count'].items()):lines.append(f'- {arm}: {count}')
    p=r['pipeline']
    lines.extend(['','## 流水线与失败','',
                  f"成功 {p['successful']} / {p['total']}，成功率 {p['success_rate']:.9f}。",
                  f"成功事件中 ΔE > τ 的数量为 {p['filtered_successes']}；占所有事件的比例 {p['fraction_filtered_of_all']:.9f}，占成功事件的比例 {p['fraction_filtered_of_success']:.9f}。",
                  f"每事件产物几何与单点的平均墙钟时间为 {p['mean_product_wall_s']:.9f} 秒；该值不含每母体一次的反应物构建、初始化及测试开销。完整 CPU 用量见资源文件。",'', '失败原因：'])
    for reason,count in sorted(p['failure_reasons'].items()):lines.append(f'- `{reason}`: {count}')
    lines.extend(['','## 冻结定义、验证和限制','',
                  '排序规则及阈值来源见 configs/explore/enum_thermo.json 与 enum_thermo_tau.json。平局按通道 SHA256；失败事件按冻结规则放置，不删除。',
                  '指标先按母体与训练 seed 计算，再在母体内平均。配对差使用 split_group 分组的 cluster_summary 双侧区间。k 网格、删失规则与原枚举分析一致；每个比较的 retained/excluded k 和母体名单在 results.json 中。',
                  'T1b-0 与 B0 完成候选门禁逐字节复现；三个最小枚举母体的 b2f2 过滤与直接枚举通道集合完全一致。两次最终分析的逐字节核验记录随证据交付。',
                  '本次未采用 171 母体子集：前五个训练母体的计时预算判断选择了全 screen。预估和实际资源必须分别报告，不能把预估当成实际用量。',
                  '本实验只有代理事件召回，不证明提出了可认证过渡态，也不证明化学发现或机制优势。训练阈值和 screen 数据不构成独立保留集确认。',
                  'RDKit 未收敛或无法嵌入的情况计为失败；无 MMFF/UFF 参数但嵌入成功时按交接保留几何并记录 no_forcefield。不得根据结果修改阈值或规则。',''])
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--repeat',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args()
    if a.results.read_bytes()!=a.repeat.read_bytes():raise RuntimeError('Analysis byte replay failed')
    a.out.mkdir(exist_ok=False)
    r=json.loads(a.results.read_text())
    shutil.copyfile(a.results,a.out/'results.json')
    (a.out/'REPORT_ZH.md').write_text(report_text(r))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    for name in ('B0','S_rand2','S_N2','S_thermo','S_N_thermo'):
        rates=r['rates'][name];ks=sorted(map(int,rates))
        ax.plot(ks,[rates[str(k)] for k in ks],marker='o',label=name)
    ax.set_xscale('log',base=2);ax.set_xticks(ks,labels=list(map(str,ks)))
    ax.set(xlabel='Distinct-event budget k',ylabel='Best catalogue event recall',ylim=(0,1),title='Exploratory T1b (screen)')
    ax.legend();fig.tight_layout();fig.savefig(a.out/'recall.png',dpi=150,metadata={'Software':'xtbflow T1b'});plt.close(fig)
    manifest=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),slurm_job_id=None,
                  config_sha256=sha256(Path('configs/explore/enum_thermo.json').read_bytes()).hexdigest(),
                  input_sha256={str(p):sha256(p.read_bytes()).hexdigest() for p in (a.results,a.repeat)},
                  tau_sha256=sha256(Path('configs/explore/enum_thermo_tau.json').read_bytes()).hexdigest(),
                  byte_replay=True,exploratory=True)
    if manifest['dirty']:raise RuntimeError('Dirty rendering source')
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    (a.out/'SHA256SUMS').write_text(''.join(f'{sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in sorted(a.out.iterdir()) if p.name!='SHA256SUMS'))


if __name__=='__main__':main()
