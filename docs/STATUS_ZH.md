# xtbflow 项目现状索引（2026-10-17）

本文只汇总、不产生新结论：所有数字摘自所链 PR 的描述、评论或报告。**所有 PR 都未合并**，合并须负责人明确批准。2026-10-06 之后的工作按负责人的阶段指导（M0 → V1a → V1b）推进，没有回写到路线入口 #93 及第 1–5 线 issue（#94–#98）；本文补上这层索引。

## 一页摘要

- **主线**：M0 闸门 `NO-GO`（10-07；只表示无引导、等候选数时联合的单候选命中率没有超过串联，不是对联合方向的否定，见下文“M0 `NO-GO` 的读法”）→ V1a 代理筛查 `GO_V1b`（10-07）→ V1b 做完 D0/D1 后**由负责人暂缓**（10-08）。
- **之后全部是探索性检查**（V1a 开发集 / 已揭盲筛查集，不构成确认性证据）：几何信息、几何先行时钟、鞍点感知物理引导、生成中 SMC 筛选、枚举基线、物理预言机对比、开放世界 xTB/DFT。
- **主要发现**：
  - 无引导联合 B0 的单位预算效率最高，但优势可由积分步数解释（#109），单候选质量与串联无差别（#107）。
  - 几何引导（B1）在任何验证成本下都不在前沿（#107）。
  - 真实 TS 几何没有可检出的排序增益；物理信号窗口最多约 35–38%（#110–#112）。
  - 生成中按物理筛选没有胜过事后排序（#116–#118、#121）。
  - 生成胜过 b3f3 穷举 + 实反应势垒头（#113），也胜过 b2f2 穷举 + AIMNet2 热力学筛选（#120）。
- **GPU 现状（2026-10-17 经 Slurm 实测）**：
  - GPU 计算正常：PyTorch 与 gpu4pyscf 都能运行；用冻结代码重算 10-09 的锚点能量，差 5.7×10⁻¹⁴ Eh（Slurm 3371）。
  - 坏的只是 `nvidia-smi`：NVML 用户态库 595.99 与已加载内核模块 595.91.07 不一致。
  - 开放世界 DFT（#119）剩 28 项停着，原因是冻结的执行脚本在量化前用 `nvidia-smi` 记录 GPU 型号，失败即退出。V1b 的正式运行脚本不调用 `nvidia-smi`，不受影响。
- **当前没有运行中的本项目作业**（2026-10-17 `squeue -u lhshen` 为空）。

## 1. 各阶段结论

| 阶段 | 日期 | 结论（预写读法） | PR | 主要证据 |
|---|---|---|---|---|
| M0 联合 vs 串联闸门 | 10-07 | `NO-GO`：主终点 Δ(B−A) = −0.0150 [−0.0670, 0.0326]，合法率 −0.0267 [−0.0398, −0.0126]，35 个测试母体 | #101→#103→#102→#104→#105（栈，#105 含全部；含义与边界见 #105 评论） | `docs/m0/M0_GATE_REPORT.md`、`docs/evidence/m0/gate_result.json` |
| V1a 代理筛查 | 10-07 | `GO_V1b`：AUC B1−A2 +0.0931（单侧 L +0.0723），M₀ +0.0212，Mᵣ +0.0211；**边界**：无引导 B0 的 AUC 0.494 高于 B1 的 0.265 | #106（含 M0 栈） | `docs/evidence/v1a/formal/`、`docs/v1a/FREEZE_DECISION_ZH.md` |
| V1a-R 零算力重分析 | 10-08 | 同候选数下 A1>A0、B1≈B0、B0≈A0；计入验证成本 κ 时前沿依次为 B0（κ≤300）→ A1 → A2（κ≥3000）；B1 在任何 κ 下都不在前沿 | #107 | `docs/evidence/v1a_r/REPORT_ZH.md` |
| V1a-N 步数对齐对照 | 10-08 | AUC(B0-50) − AUC(A0-25+25) = −0.016 [−0.045, +0.013]：B0 的预算优势可由积分步数解释 | #109 | `docs/evidence/v1a_n/REPORT_ZH.md` |
| V1b 修订草案 v2.2 | 10-08 | 四臂认证草案，**未冻结** | #108 | `docs/v1b/AMENDMENT_v2.2_ZH.md` |
| V1b D0 + D1 | 10-08 | D0：M=342，Δ^proxy = +0.0673。D1：8/8 候选精修为一阶鞍点；协议升到 `v1b-h-gpu4pyscf-3`；003317dd 的端点软模仍是 `PROTOCOL_FAILURE`。约 6.1 + 0.6 GPU·h。**负责人 10-08 暂缓 V1b** | #122（草稿，分支 `codex/v1b-20261007`） | `docs/v1b/D1_REPORT_ZH.md` |
| 几何信息检查 | 10-08 | 真实 TS 几何：`INCONCLUSIVE`；时间窗口：`NO_USABLE_WINDOW`（xTB 窗口 14.5%） | #110 | `docs/evidence/explore_geoinfo/REPORT_ZH.md` |
| 几何先行时钟 | 10-08 | `GEOMETRY_LEAD_OPENS_WINDOW`：窗口 27.9% / 34.6%，合法率约 −1.5 pt | #111 | `docs/evidence/explore_geolead/REPORT_ZH.md` |
| 鞍点感知物理引导 pg-1 + geolead-2 | 10-08 | α0.3 `SHIFT_NOT_SPECIFIC`，α1.0 `NO_DETECTABLE_SHIFT`；窗口在 35–38% 见顶 | #112 | `docs/explore/PHYSICS_GUIDANCE_DESIGN_ZH.md` |
| SMC 生成中筛选 smc-1 | 10-09 | `NO_GAIN_OVER_POSTHOC`：relaxed_0.7 − none = −0.016 [−0.032, +0.000]；原始能量排序消融（#117）与严格顺序复验（#118）结论不变 | #116、#117、#118 | `docs/evidence/explore_smc/REPORT_ZH.md` |
| 枚举基线（b3f3） | 10-09 | `GENERATION_BEATS_ENUMERATION`：AUC(S_N) − AUC(B0) = −0.177 [−0.230, −0.123] | #113 | `docs/evidence/review_enum/` |
| 物理预言机对比 | 10-09 | GFN2 ρ 精确复现 0.7491；AIMNet2 的 ρ 提升：参考几何 +0.175 [0.071, 0.278]，生成几何 +0.111 [0.056, 0.166] | #114 | `docs/evidence/review_oracle/REPORT_ZH.md` |
| 开放世界 xTB 认证（O1） | 10-09 | 阳性对照 17/20（闸门通过）；目录外 17/40 通过 xTB 认证；ΔEa < 0 只有 1/12 | #115 | `docs/evidence/review_open_world/` |
| 枚举 + 热力学筛选（T1b） | 10-10 | 主比较 −0.0554 [−0.1070, −0.0039]；按交接修正后为 −0.0563 [−0.1075, −0.0051]；热力学筛选相对不筛（S_N2）+0.051 | #120 | `docs/evidence/explore_enum_thermo/` |
| 反应坐标诊断 + 预言机 SMC（T2a/T2b） | 10-10 | T2a `U_NOT_REACTION_MODE`（真实 TS 上 c_u<0 只占 19%）；T2b 8 臂全部 `NO_GAIN_OVER_POSTHOC` → 预写建议 `LINE_CLOSURE_RECOMMENDED` | #121 | `docs/evidence/explore_u_diagnostic/`、`docs/evidence/explore_smc_oracle/` |
| 开放世界 DFT（O2）+ AIMNet2 训练集核查 | 10-10 | 3b：两个模型都是 `UNRESOLVED`。3a：锚点通过 1 项、参考链失败 1 项（`EngineError:SCF_LIMIT`，按决定不重跑），基础设施失败 3 项；已用 801 GPU s，**剩 28 项等 GPU** | #119 | `docs/evidence/review_open_world_o2/DECISION_ZH.md`、`INFRASTRUCTURE_ZH.md` |

各 PR 的验收意见（逐项复跑、字节一致回放、独立复算）已贴在对应 PR 评论里：#113–#121。

### M0 `NO-GO` 的读法

M0 只检验一件事：参数、步数、数据相同且**无引导**时，联合在"32 个候选中事件与 TS 几何同时命中"的比例上是否**高于**串联。

- **结果是"没有更好"，不是"更差"**：区间 [−0.067, 0.033] 包含 0。触发 NO-GO 的是"点估计 ≤ 0"这一条。
- **M0 没有检验的**：计算开销；推理期几何信号改写事件（联合独有）；双向条件生成等。
- **M0 指导预写的后续动作没有执行**：预写的是"放弃联合卖点、转向方向 B"。负责人保留联合生成作为架构前提，项目转入 V1a。
- **后续证据**（V1a 筛查集已揭盲一次，属探索性）：
  - 等候选数时，联合与串联的单候选质量相同（#107）；
  - 每个模态积分步数相同时，联合每个候选的开销约为串联的一半（#109）。

完整说明见 #105 的 2026-10-17 评论。

## 2. PR 叠放关系

```text
main
├─ #101 m0/task0-env ← #103 task1-data ← #102 task2-model ← #104 task3-train ← #105 task4-eval
└─ #106 codex/v1a-20261007（含 M0 栈）
   ├─ #107 v1a-r/reanalysis ← #108 v1b/amend-v2.2
   │                        ← #109 v1a-n/nfe-control
   ├─ #110 explore/geometry-information ← #111 explore/geometry-lead-clock
   │     ├─ #112 explore/physics-guidance ← #116 explore/smc-selection ← #117 smc-raw-ranking
   │     │                                                              ← #118 smc-strict-replication
   │     ├─ #113 review/enum-baseline
   │     └─ #114 review/oracle-benchmark
   └─ #122 codex/v1b-20261007（V1b D0/D1，草稿）← #115 review/open-world-o1 ← #119 review/open-world-o2
explore/followup-base（#116 + 合并 #113、#114 + 交接文档，无 PR）← #120 explore/enum-thermo
                                                               ← #121 explore/smc-oracle
```

五线路线 #93 时期的 PR（#61–#100）和 issue（#74–#98）仍保持原状，10-06 以后没有更新。

## 3. 待负责人决定的事项

1. **开放世界 DFT（#119）怎么继续**，两条路任选其一：
   - 等管理员重载 NVIDIA 内核模块或重启 node2，不改代码；
   - 批准一个带版本号的最小改动：执行脚本和调度器改用 CUDA 运行时（cupy）记录 GPU 信息并做预检，`qc_protocol.py`、样本和设置都不变。

   `nvidia-smi` 恢复后，按 #119 运行第三版调度器：

   ```bash
   scripts/review_open_world_dispatch_v3.py --root /home/lhshen/xtbflow-runs/followup-20261009 \
     --source /home/lhshen/xtbflow-runs/followup-20261009/source-bb09c34 --quarantine-preqc
   ```

   代码在 `086fe82`，n2 克隆 `source-086fe82`。预计约 11 GPU·h，在 16 GPU·h 上限内。
2. **V1b 是否恢复**。负责人 10-17 已同意可同时占 2 张卡，以下几项仍待定：
   - 规模：指导式分配约 230 GPU·h；四层普查约 330 GPU·h。
   - 版本：v2.1 两臂，或 v2.2 四臂（#108，305/564 GPU·h，用的是暂定单价）。
   - 端点软模规则：003317dd 端点负模 −42.3 cm⁻¹，需要预先写定统一的处理办法。
   - 正式运行脚本 `scripts/v1b_run.py` 要求清单中已置位 `d1_protocol_validated` 和 `decision_rule_frozen`，两者都必须由有记录的步骤写入。
3. **生成中筛选研究线是否关闭**：#121 给出预写建议 `LINE_CLOSURE_RECOMMENDED`，这只是建议，不是负责人决定。
4. **未合并 PR 的处理**：#101–#122 共 22 个。需要决定哪些合并、哪些作为证据存档后关闭。叠放顺序见第 2 节。

## 4. 规范文档（指导与交接）

| 文档 | 用途 | 位置 |
|---|---|---|
| 课题方向规划：以并联事件—几何生成为核心（2026-10-07） | 研究方向、实验组 E1–E5、按结果选论文主卖点的决策树、与竞争工作的差异；V1a 指导引用其中的母体数估算 | `docs/guides/DIRECTION_PLAN_ZH.md`（本次补入） |
| M0 阶段执行指导 | M0 任务 0–4 与去/留闸门 | `docs/guides/M0_GUIDE_ZH.md`（本次补入） |
| V1a 第一轮验证推进指导 v2.1 | V1a 代理筛查 | 分支 `codex/v1a-20261007`：`docs/v1a/protocol-v2.1.md` |
| V1b 第一轮验证推进指导 v2.1 | V1b DFT 认证 | `docs/guides/V1B_GUIDE_v2.1_ZH.md`（本次补入） |
| V1a 后续任务交接 | 任务 R/N/V（#107–#109） | `docs/guides/V1A_FOLLOWUP_HANDOFF_ZH.md`（本次补入） |
| 综述对齐交接 | #113–#115 的执行规范，第 10 节为验收清单 | `docs/handoffs/HANDOFF_REVIEW_ALIGNMENT_ZH.md`（本次补入） |
| SMC 筛选交接 | #116–#118 的执行规范 | `docs/handoffs/HANDOFF_SMC_SELECTION_ZH.md`（本次补入） |
| 三项后续任务交接 | #119–#121 的执行规范，第 11 节为验收清单 | `docs/handoffs/HANDOFF_FOLLOWUP_ZH.md`（与 `explore/followup-base` 上的同一文件逐字节相同） |

补入的文件是作者本地保存的版本，原样提交，未改动。#118 引用的负责人口头修订（"全部允许"）不在交接文件中；负责人已于 2026-10-09 确认 CPU 可直接在 n2 登录 shell 运行（见 #113 验收评论）。

## 5. n2 运行目录（原始输出，未入库，只提交哈希）

所有目录都在 `/home/lhshen/xtbflow-runs/` 下：

| 用途 | 目录 |
|---|---|
| M0 | `m0/stabilize-20261007/{replay,tune,full,test}` |
| V1a | `v1a-20261007` |
| V1a-R / V1a-N | `v1a-r-20261008`、`v1a-n-20261008` |
| V1b | `v1b-20261007` |
| 几何信息 / 几何先行 / 物理引导 | `explore-geoinfo-20261008` |
| SMC | `explore-smc-20261009` |
| 综述对齐三项 | `review-align-20261009` |
| 后续三项 | `followup-20261009` |
| 验收 | `acceptance-20261009`、`acceptance-smc-20261009`、`acceptance-followup-20261010` |
