"""Render frozen enumeration results without recomputing or tuning inference."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=json.loads(a.results.read_text());primary=r['comparisons']['S_N-B0'];s=primary['summary'];lo,hi=s['ci_two95']
    lines=[f"本报告全部结果为探索性。主比较 AUC(枚举+S_N)−AUC(B0) 为 {s['estimate']:.6f}，双侧95%区间 [{lo:.6f}, {hi:.6f}]，预写读法为 {primary['reading']}。",'',
        '# T1：冻结 b3f3 枚举基线','','本任务不改变 V1a 结论。最佳通道命中是目录事件代理，不是物理认证的过渡态。成本轴为送验的不同事件数，忽略网络调用成本，因而对生成模型有利。',
        '', '## 冻结域与覆盖','','训练集事件大小审计和五个训练母体可行性检查决定 b3f3；配置及域选择在正式评估前推送。最终合法性复用 valid_endpoint，事件去重复用 canonical_event；断键多重集按反应物自同构取代表，不改变规范事件集合。',
        f"screen 母体数：{r['n_parents']}；最佳通道覆盖率：{r['best_coverage']:.6f}；目录通道覆盖率的母体均值：{r['channel_coverage_parent_mean']:.6f}。",
        f"合法事件数中位数：{r['n_enum']['median']:.3f}；95%分位：{r['n_enum']['p95']:.3f}；事件总数：{r['n_enum']['total']}。",'',
        '## 配对比较与删失','','|比较|AUC差|双侧95%区间|保留k|剔除k|读法|','|---|---:|---|---|---|---|']
    for name,d in sorted(r['comparisons'].items()):
        ss=d['summary'];lo,hi=ss['ci_two95'];lines.append(f"|{name}|{ss['estimate']:.6f}|[{lo:.6f}, {hi:.6f}]|{d['kept_k']}|{d['excluded_k']}|{d['reading']}|")
    lines+=['','生成侧某k不同事件不足时对应seed删失，先在母体内平均未删失seed。删失格比例超过5%的k不进该比较的AUC；原始梯形权重删除对应点后归一化。各比较保留网格可能不同，不能跨行把AUC直接当同一估计量比较。没有剩余seed的母体不进入配对分析，清单保留于结果。',
        '', '## 各k召回','','|方法|1|2|4|8|16|32|64|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,curve in sorted(r['rates'].items()):lines.append('|'+name+'|'+'|'.join('NA' if curve[str(k)] is None else f"{curve[str(k)]:.6f}" for k in (1,2,4,8,16,32,64))+'|')
    lines+=['','随机枚举使用超几何解析期望，不模拟。每个screen母体各k均与独立组合数表达式检查一致。枚举耗尽不算删失；生成侧高k点可能只供描述。',
        '', '## 枚举覆盖不到但生成命中的母体',f"最佳通道不在枚举集合的母体数：{len(r['outside_best_parents'])}。以下计数为任一冻结seed在完整原始流中命中，不限于AUC网格。",'', '|生成臂|命中母体数|','|---|---:|']
    for arm,ids in sorted(r['generator_hit_outside_best'].items()):lines.append(f'|{arm}|{len(ids)}|')
    if 'development' in r:
        d=r['development'];lines+=['','## dev辅助检查',f"母体数 {d['n_parents']}；最佳通道覆盖率 {d['best_coverage']:.6f}；目录通道覆盖率均值 {d['channel_coverage_parent_mean']:.6f}；合法事件数中位数 {d['n_enum']['median']:.3f}，95%分位 {d['n_enum']['p95']:.3f}。dev结果不用于选域或调参。"]
    lines+=['','## 复现与证据边界','','manifest.json 列出配置哈希、干净已推送源码与原始结果文件；SHA256SUMS.remote 指向 n2 大文件。训练集所有域内目录事件的重新生成检查、B0完整候选轴全精度复现、CPU/GPU与Slurm用量详见 VALIDATION.md。',
        '', '```bash','export CUDA_VISIBLE_DEVICES= PYTHONPATH=src:vendor/mechai_reusable OPENBLAS_NUM_THREADS=1',
        'python scripts/review_enum_analyze.py --enumdir /home/lhshen/xtbflow-runs/review-align-20261009/enum-screen-9403d6c --devdir /home/lhshen/xtbflow-runs/review-align-20261009/enum-dev-7c28436 --out results.json',
        '```','','实际推理必须复现结果字节，不能以测试通过代替科学检查。图 recall.png 展示相同召回表，统计解释以删失表和配对区间为准。']
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
