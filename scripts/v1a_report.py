"""Single final V1a analysis (Section 10): gate, V1b source population, report.

Runs once on sealed, matched screen reports bound to one freeze. Refuses to run
again into an existing output directory; re-analysis is a new study version.
"""
import argparse
import json
from pathlib import Path

from xtbflow.v1.data import file_hash,write_json
from xtbflow.v1.formal import ARMS,formal_analysis,load_freeze,v1b_export


def fmt(s):
    if 'lower_one95' not in s:return f"{s['estimate']:+.4f}（组数不足，无区间）"
    lo,hi=s['ci_two95']
    return (f"{s['estimate']:+.4f}，双侧95% [{lo:+.4f}, {hi:+.4f}]，"
            f"单侧95%界限 L={s['lower_one95']:+.4f} / U={s['upper_one95']:+.4f}")


def markdown(gate,freeze):
    sec=gate['secondary'];lines=[
        '# xtbflow V1a 正式代理筛查结果','',
        f"**决策：`{gate['decision']}`**（{gate['reason']}）。P={gate['n_parents']} 母体，G={gate['n_formula_groups']} 个分子式组；"
        f"三个训练 seed 平均，采样 seed 0。冻结记录 `{freeze['freeze_sha256']}`。",'',
        f"- 主比较 B1−A2 归一化 log-budget 联合召回 AUC：{fmt(gate['primary'])}",
        f"- 机制 M₀ = E[U(F_S)−U(F0)]：{fmt(gate['mechanism_vs_none'])}",
        f"- 机制 Mᵣ = E[U(F_S)−U(F_R)]：{fmt(gate['mechanism_vs_random'])}",'',
        '这是有限 Transition1x 目录上的代理筛查：`PROXY_MATCH` 只表示接近冻结参考记录，不是经频率/IRC 认证的 TS；'
        '本轮没有任何新量化计算。GO 不证明 AUC≥+0.05 或机制≥+0.02；非 GO 不证明 H1 为假。','',
        '## 五臂成本曲线（母体×三 seed 平均）','',
        '| 臂 | 召回 @4/8/16/32/64 | 事件召回 @16 | 完成候选 @4/8/16/32/64 | AUC | 事件 AUC |','|---|---|---:|---|---:|---:|']
    for arm in ARMS:
        c=gate['curves'][arm]
        lines.append(f"| {arm} | {' / '.join(f'{v:.3f}' for v in c['hits'])} | {c['event_hits'][2]:.3f} | "
                     f"{' / '.join(f'{v:.1f}' for v in c['completed_candidates'])} | {c['auc']:.4f} | {c['event_auc']:.4f} |")
    lines+=['','## 关键次要指标（不设闸门）','',
        f"- 事件级最佳通道 AUC B1−A2：{fmt(sec['event_level_best_channel_auc_B1_minus_A2'])}",
        f"- 预算 16 二元召回 B1−A2：{fmt(sec['budget16_binary_B1_minus_A2'])}",
        f"- AUC B1−B0：{fmt(sec['auc_B1_minus_B0'])}",
        f"- 主差值整组 bootstrap 双侧95%：{sec['primary_cluster_bootstrap']['ci_two95']}",
        f"- 事件层收益未转化为可匹配几何：{sec['event_gain_not_converted_to_geometry']}",
        f"- 窗口停止条件成立：{gate['window_stop_supported']}",'','逐训练 seed 主差值：']
    for s,v in sec['per_seed_primary'].items():lines.append(f"- seed {s}：{fmt(v)}")
    lines+=['','连续引导控制（B1_cont/B2_cont，相对 F0）：']
    for k,v in sec['continuous_controls'].items():lines.append(f"- {k}：{fmt(v)}")
    lines+=['','罕见度分层（训练 seed 0 的 B0 128 提议探查，主差值）：']
    for k,v in sorted(sec['rarity_strata'].items()):
        lines.append(f"- {k}：n={v['n_parents']}，{fmt(v['primary'])}")
    lines+=['','## 冻结时的规划决定','',freeze['planning_decision']['text'].strip(),'',
        '## 未执行与边界','','- V1b 量化认证未启动；本目录的 `v1b_source_population.jsonl` 只是预算 16、训练 seed 0 的冻结源总体。',
        '- 区间条件于这三次训练的冻结模型和分子式独立的工作假设；不估计重训练方差。','']
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--frozen',type=Path,required=True)
    ap.add_argument('--efficiency',type=Path,nargs=3,required=True,help='matched screen streams, seeds 0/1/2')
    ap.add_argument('--pulses',type=Path,nargs=3,required=True,help='matched screen pulses, seeds 0/1/2')
    ap.add_argument('--controls',type=Path,required=True)
    ap.add_argument('--rarity',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    a=ap.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=False)
    freeze=load_freeze(a.frozen)
    load=lambda p:json.loads(p.read_text())
    efficiency={};pulses={}
    for p in a.efficiency:
        r=load(p);efficiency[r['training_seed']]=r
    for p in a.pulses:
        r=load(p);pulses[r['training_seed']]=r
    gate=formal_analysis(freeze,efficiency,pulses,load(a.controls),load(a.rarity))
    gate['input_file_sha256']={str(p):file_hash(p) for p in [*a.efficiency,*a.pulses,a.controls,a.rarity,a.frozen]}
    write_json(a.out_dir/'gate.json',gate)
    rows=v1b_export(freeze,efficiency[0])
    with (a.out_dir/'v1b_source_population.jsonl').open('x',encoding='utf-8',newline='\n') as f:
        for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
    (a.out_dir/'RESULT_ZH.md').write_text(markdown(gate,freeze),encoding='utf-8',newline='\n')
    print(json.dumps(dict(decision=gate['decision'],primary=gate['primary'],M0=gate['mechanism_vs_none'],
                          MR=gate['mechanism_vs_random'],v1b_rows=len(rows)),indent=2))


if __name__=='__main__':main()
