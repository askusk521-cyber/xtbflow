# V1b D1 开发链报告（2026-10-08）

按《xtbflow V1b 第一轮验证推进指导》v2.1 任务 D1 执行。量化程序按负责人选择改用 gpu4pyscf + geomeTRIC（原指导模板为 ORCA 6.1；n2 无量化 ORCA）。所有 GPU 计算经 Slurm（作业 2630、2635、2666 及两次 `srun` 对照）。D1 只用开发集，不涉及正式 V1a 母体；正式 H 计算尚未开始。

## 方法（协议 `v1b-h-gpu4pyscf-3`）

RKS ωB97X/6-31G(d)（球谐 d），电子能量，无色散、无溶剂、无密度拟合；`grids.level=5`、`conv_tol=1e-10`；初始与最终均为解析 Hessian。几何收敛阈值对齐 ORCA TightOpt（最大梯度 1e-4、RMS 梯度 3e-5 Eh/bohr，能量 1e-6 Eh，位移 1e-3/6e-4 bohr）；灰区重试用 VeryTight。频率在平移/转动精确张成空间的正交补中对角化，不按个数丢弃最小频率；−30 cm⁻¹ 为分流阈值。

D1 期间的两次统一修订（全部臂、参考共用）：

1. IRC 原设每方向 80 步、trust 0.1：8 条中 3 条两向均撞上限。改为 trust 0.2，再改为每方向 300 步（一条反向 IRC 在 150 步时仍在稳定下降）。
2. 端点最低点若仍有明显负模（< −30 cm⁻¹），沿该模位移（最大原子 0.10 Å）后以 VeryTight 重新优化一次再判定（一例为 −30.9 cm⁻¹）。

修订 3 的两项尚未在对应失败案例上重跑验证（见下）。

## 结果

| 项目 | 原子 | 代理层 | 结果 | GPU 墙钟 s |
|---|---:|---|---|---:|
| 8 个反应物锚点 | 6–18 | — | 全部 `MINIMUM_PASS` | 43–278 |
| 原目录最佳参考 TS | 6 | — | `STRICT_JOINT_GRAPH_VALID` | 880 |
| 候选 02781d1e | 6 | MATCH_BEST | `STRICT_JOINT_GRAPH_VALID`（修订 2 复算一致） | 1292 |
| 候选 0043b651 | 14 | MATCH_BEST | 反向 IRC 150 步未完 → 协议失败（待修订 3 复验） | 3012 |
| 候选 0021602a | 15 | MATCH_NONBEST | `STRICT_JOINT_GRAPH_VALID` | 1301 |
| 候选 003317dd | 11 | MATCH_NONBEST | IRC 完成，一端 −30.9 cm⁻¹ → 协议失败（待端点重试复验） | 4728 |
| 候选 007235d0 | 18 | KNOWN_EVENT_GEOMETRY_MISS | 修订 2 后 `STRICT_JOINT_GRAPH_VALID` | 4339 |
| 候选 023d1155 | 8 | KNOWN_EVENT_GEOMETRY_MISS | `STRICT_JOINT_GRAPH_VALID` | 1439 |
| 候选 02481a8c | 7 | OUTSIDE_CATALOGUE_EVENT | `STRICT_EVENT_MISMATCH`（连通明确，事件不同） | 1113 |
| 候选 001d44cd | 18 | OUTSIDE_CATALOGUE_EVENT | `IRC_REACTANT_MISMATCH` | 2253 |

8/8 候选精修为明确一阶鞍点。6–10 条链只用于兼容性与成本，不估计精度。

## 成本

D1 总计 GPU 墙钟约 6.1 小时（单卡），分配核时约 44（8 核或 4 核/作业）。整链（含重试）每条候选 1100–4700 s，均值约 0.68 GPU 小时；参考 TS 链约 0.25–0.7；锚点约 0.04。SCF（每个几何步复用上一步密度）占链耗时约 80%。

gpu4pyscf 对照（`docs/evidence/v1b/gpu4pyscf_benchmark.json`）：18 原子/120 基函数时，GPU 对 8 核 CPU 的 Hessian 为 40 s 对 309 s、梯度 1.5 s 对 11 s；6 原子时 GPU 无优势。能量差 ≤ 8×10⁻¹² Eh。新进程首次 GPU SCF 含约 2 分钟 cupy 即时编译。

## 正式规模工作表（指导第 13 节）

D0：M=342，Δ^proxy=+0.0673；需 QC 的母体层 B1_only 56、A2_only 33、both 34、neither_with_best_event 44（C\* 候选共 289）；neither_empty 175 无需 QC。每母体 1 条原最佳参考。阶段 A 非 invalid 层 20 个。

| 方案 | 阶段 B | 阶段 A | 最坏 GPU 小时 | 4 卡并行墙钟 | 精度 |
|---|---|---|---:|---:|---|
| 指导式分配 | 两个不一致层普查 89 + both 12/34 + neither_with 12/44 | 20 层各 2 | 约 230 | 约 2.5 天 | r=0.10 时正态半宽约 0.027；若抽样层修正全为 0 则改用较宽的超几何区间 |
| 四层普查 | 167 母体全部 | 20 层各 2 | 约 330 | 约 3.5 天 | 无抽样误差，决策只受未解决标签影响 |

最坏值按每个母体—臂全部 C\* 候选完整链计算；见证即停与能量短路会降低实际用量。正式运行前先用修订 3 复验 0043b651、003317dd 两条（约 1.5 GPU 小时）。
