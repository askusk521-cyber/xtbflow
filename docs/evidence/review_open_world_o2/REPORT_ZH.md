# 开放世界第二阶段（O2）：xTB 新通道的 DFT 认证报告

本报告全部结果为探索性。

- **候选认证**：经 xTB 认证的 10 条目录外候选中，4 条在 DFT 层面通过严格认证，即 40%（Wilson 95% 区间 16.8%–68.7%）。DFT 方法为 ωB97X/6-31G(d)，协议 `v1b-h-gpu4pyscf-3`。
- **配对母体**：候选链与基准链都通过的母体有 4 个。这 4 个母体的 ΔEa_DFT 全部为正，在 +8.8 到 +54.7 kcal/mol 之间：没有一条候选通道的势垒低于同母体的已知最佳通道。按母体聚类的均值为 +23.8 kcal/mol，95% 区间 [−9.7, +57.3]。

所有数字都来自同目录的 `results.json`。

## 1. 问题与做法

**问题**：#115 在 xTB 层面认证了一批目录外新通道。它们在 DFT 层面是否仍然成立？势垒与同母体的已知最佳通道相比是高还是低？

**样本**：
- 规则（交接第 8.1 节）：从 #115 中，取"目录外候选与同母体基准都通过 xTB 严格认证"的 12 个母体；按 ΔEa_xTB 升序排列，平局用 sha256(candidate_id)，每个母体一条，取前 10。
- 名单冻结在 `configs/review/open_world_o2.json`，于 `97169c4` 推送，早于任何 DFT 作业。
- 这 10 个母体都属于 V1a 的 screen 集。

**计算项**：每个母体 3 项，共 30 项。
- `anchor`：锚定反应物的极小点。
- `reference`：#115 基准所用 T1x 记录的 TS。
- `candidate`：#115 中该候选经 xTB 优化后的 TS。

**协议**：
- 未修改的 `qc_protocol.py`：TS 精修与频率、双向 IRC（每方向 300 步）、端点极小点与频率、反应物和产物的图身份判定。
- 只执行协议内置的重试，不做其他重试。

**执行**：
- Slurm 2752、2753 由执行脚本第 1 版（`bb09c34`）运行。
- 其余 28 项由执行脚本第 2 版和调度器第 4 版（`1823534`）运行。第 2 版只改了 GPU 型号的记录方式，见 `DECISION_NVML_ZH.md`。
- 全程单卡串行。

**分析**：
- 六项指标在冻结配置中预先写定。
- 实现这些指标的操作细则写在 `configs/review/open_world_o2_analysis.json`，与分析脚本一起在 `e45f85d` 推送，之后才计算任何势垒。
- 在 n2 的干净克隆 `source-e45f85d` 上运行两次，`results.json` 逐字节相同。

## 2. 认证通过数

| 计算项 | 通过 / 计划 | 比例与 Wilson 95% 区间 | 状态分布 |
|---|---|---|---|
| 锚点 | 10 / 10 | 100%（72.2%–100%） | `MINIMUM_PASS` 10 |
| 基准（参考 TS） | 9 / 10 | 90%（59.6%–98.2%） | `STRICT_JOINT_GRAPH_VALID` 9；链异常 `EngineError:SCF_LIMIT` 1 |
| 候选（xTB 新通道） | 4 / 10 | 40%（16.8%–68.7%） | `STRICT_JOINT_GRAPH_VALID` 4；`STRICT_EVENT_MISMATCH` 2；`PROTOCOL_FAILURE` 2；`EVALUATION_UNRESOLVED` 1；链异常 `EngineError:SCF_LIMIT` 1 |

**分母**：每类都是 10 个计划项。每项只尝试一次；没有判定的链异常计为"已尝试、未通过"。

**分歧发生在哪一步**：
- 10 条候选都过了 TS 精修，得到一阶鞍点。
- 分歧出在 IRC 和端点身份这两步，见第 5 节。

## 3. 配对母体的势垒

候选与基准都通过严格认证的母体有 4 个。ΔEa_DFT = E_TS(候选) − E_TS(基准)，按电子能量计算。两列势垒都相对同母体锚点极小点。

| 母体（分子式） | ΔEa_xTB | ΔEa_DFT | 候选势垒 | 基准势垒 |
|---|---:|---:|---:|---:|
| `3e4899195c01fa43baa1`（C5H8O2） | +20.40 | +19.48 | 68.60 | 49.12 |
| `8962b71ed3a7a171af9e`（C5H7NO） | +26.26 | +54.71 | 123.49 | 68.78 |
| `a9460cfa7d3e4acccdb6`（C6H10O） | +2.84 | +12.28 | 83.65 | 71.38 |
| `c0f8c54d5d73fed95b92`（C4H6N2O） | +2.66 | +8.77 | 85.93 | 77.16 |

单位均为 kcal/mol。

**计数与符号**：
- ΔEa_DFT < 0：0 / 4（Wilson 区间 0%–49.0%）。
- ΔEa_DFT ≤ +5：0 / 4（Wilson 区间 0%–49.0%）。
- DFT 与 xTB 的符号一致率：4 / 4（Wilson 区间 51.0%–100%）。4 个母体的 ΔEa_xTB 本身都是正的。

**ΔEa_DFT 的母体聚类均值**：+23.8 kcal/mol，95% 区间 [−9.7, +57.3]（4 个母体，4 个分子式组，t 分布 3 个自由度）。母体太少，区间很宽，这个均值只作描述。

**散点图**：`delta_ea_scatter.svg`。

- **幅度**：3 个母体的 ΔEa_DFT 明显大于 ΔEa_xTB（+54.7 对 +26.3；+12.3 对 +2.8；+8.8 对 +2.7），另 1 个接近（+19.5 对 +20.4）。在这 4 个母体上，xTB 低估了新通道相对已知最佳通道的势垒差。只有 4 个点，不能推广。
- **入选样本中 ΔEa_xTB 最低的候选**：只有它的 ΔEa_xTB 为负，即母体 `4f5559117464139fee00`，−0.60 kcal/mol。
  - 这条候选在 DFT 中 IRC 达到 300 步上限（`PROTOCOL_FAILURE`），按冻结规则不进入配对。
  - **更正（2026-10-19）**：本报告初版写的是"它在 DFT 层面没有得到检验"，这不准确。它的 DFT TS 已经优化，并通过了频率检查，能量比同母体基准 TS 高 43.76 kcal/mol；该值由 `results.json` 中两者的 TS 能量算出，不属于冻结的六项指标。
  - 所以 DFT 找到的鞍点并不比已知通道低。剩下的问题只是这个鞍点是否连接 xTB 预测的事件。
  - 补充诊断（`irc_endpoint/REPORT_ZH.md`）的回答是否定的：这个鞍点一端是预测产物，另一端是第三种异构体，两端都不是反应物（`IRC_REACTANT_MISMATCH`）。因此 +43.76 不是预测通道的势垒差，预测通道在 DFT 下的 TS 仍未找到。
  - 补充诊断不改变本节的计数。

## 4. 健全性检查：基准势垒与目录势垒

对 9 条通过的基准，计算"相对锚点的 DFT 势垒 − T1x 目录势垒"：
- 差的绝对值最大为 0.041 kcal/mol；
- 用 IRC 反应物端代替锚点时最大为 0.173 kcal/mol；
- 没有一条超过 2 kcal/mol 的讨论阈值。

9 条中有 7 条基准反应与锚点所用的 T1x 记录不同（`same_reaction_as_anchor = false`），差值同样很小。

结论：本协议（gpu4pyscf，`grids.level=5`）在这些母体上能复现 T1x 目录能量。第 3 节的势垒差不是由 DFT 设置与目录不一致造成的。

## 5. 未通过的项

这些都是计算结果，**不能**说成"通道不存在"。

| 母体 | 项 | 状态 | 发生在哪一步 |
|---|---|---|---|
| `5b2d3d450552ae85d0f4`（8 原子） | 基准、候选 | 链异常 `EngineError:SCF_LIMIT` | 两条链都在 IRC 中 SCF 达到迭代上限。按 10-10 决定记为计算失败，不重跑；该母体没有 DFT 配对结果 |
| `4f5559117464139fee00` | 候选 | `PROTOCOL_FAILURE` | TS 通过；IRC 达到每方向 300 步上限（`IRC_LIMIT`） |
| `7dc6922783059ead3fb0` | 候选 | `PROTOCOL_FAILURE` | 同上，`IRC_LIMIT` |
| `8fe6d50aaad53c21d3c5` | 候选 | `EVALUATION_UNRESOLVED` | IRC 完成，两端都是极小点，但端点身份判定无法给出结论 |
| `e193d2b0c22148150613`、`a6ee19115420a88ad88e` | 候选 | `STRICT_EVENT_MISMATCH` | IRC 连通两个极小点，但对应的事件与预测事件不同：xTB 认证的事件在 DFT 层面没有复现 |

## 6. 用时与 GPU 台账

**总量**：33 个 Slurm 作业，合计 37,395 GPU 秒 ≈ 10.39 GPU 小时，在 16 GPU 小时上限内。
- 30 个是计算项的唯一一次尝试，共 37,389 s。
- 3 个是 10-10 因 `nvidia-smi` 失败、在量化之前就退出的作业，共 6 s。

**按类型**：

| 类型 | 作业数 | GPU 秒合计 | 中位数 | 最大 | 梯度调用 | Hessian 调用 |
|---|---:|---:|---:|---:|---:|---:|
| 锚点 | 10 | 1,179 | 123 | 162 | 100 | 10 |
| 基准 | 10 | 12,741 | 1,125 | 2,529 | 1,191 | 38 |
| 候选 | 10 | 23,469 | 1,927 | 6,575 | 2,606 | 34 |

- 表中的合计、中位数和最大值，以及下面的平均值，都由 `results.json` 的 `items` 字段直接计算。第 3 节的 −0.60 kcal/mol 来自冻结配置 `open_world_o2.json` 的样本表。
- 最长的作业是 `4f555911…` 的候选链，6,575 s，以 `IRC_LIMIT` 结束。
- 候选链平均 2,347 s ≈ 0.65 GPU 小时，与 D1 实测的 0.68 小时相近。
- 每一项的 Slurm 运行时间、链墙钟、各阶段 QC 计算时间、geomeTRIC 时间，以及 SCF、梯度、Hessian 调用数，都在 `results.json` 的 `items` 字段里。

**不在台账里的 GPU 用量**：
- 2026-10-17 的诊断检查：Slurm 3371 和两次短的 `srun` 检查，合计约 5 分钟；
- 调度器第 4 版在 Slurm 内的 CUDA 预检，数秒。

## 7. 偏差与限制

**执行偏差**（都已有单独记录）：
- 2753 的 SCF 异常按 10-10 决定保留为失败（`DECISION_ZH.md`）。
- 2825–2827 是量化前的基础设施失败。10-17 移到 `o2-run-preqc-failed/` 后重新排队，三项各只有一次真正的尝试（`INFRASTRUCTURE_ZH.md`）。
- 10-17 起执行脚本改用 CUDA 运行时记录 GPU，协议不变（`DECISION_NVML_ZH.md`）。
- `INFRASTRUCTURE_ZH.md` 里"所有 GPU 计算受影响"的说法不对，已在 `DECISION_NVML_ZH.md` 更正。

**分析时序**：
- 分析细则写于全部 30 项跑完之后。
- 写细则时，作者看过各项的协议状态和 IRC 局部势垒，但没有计算 ΔEa_DFT、相对锚点的势垒、目录差或符号一致率。
- 六项指标本身在 DFT 之前已冻结。

**样本与推断**：
- 10 个母体是 12 个合格母体中 ΔEa_xTB 最低的，且属于 screen 集。
- 4 个配对母体又是以 DFT 成功为条件的子集。
- 所以不能据此推出总体发现率，也不能推广到其他母体。
- n = 4，所有区间都很宽；符号一致率 4/4 的 Wilson 下界只有 51%。

**能量口径**：
- 气相电子能量，ωB97X/6-31G(d)，与 T1x 一致。
- 没有零点能和热校正，没有溶剂。
- 势垒是 TS 相对锚点极小点的能量差。

**其他**：
- 软模和 `IRC_LIMIT` 都按协议原样判定，没有新增规则。
- 分析只用 CPU，在 n2 登录 shell 上运行（负责人 2026-10-09 授权），没有提交 Slurm 作业。

## 8. 复现

在 n2 上，于干净克隆 `source-e45f85d` 中使用 `xtbflow` 环境：

```bash
export PYTHONPATH=src:vendor/mechai_reusable CUDA_VISIBLE_DEVICES=
R=/home/lhshen/xtbflow-runs/followup-20261009
P=/home/lhshen/xtbflow-runs/v1a-20261007/data/parent_catalog.json
python scripts/review_open_world_o2_analyze.py analyze --run-root $R --parents $P --out results.json
python scripts/review_open_world_o2_analyze.py plot --results results.json --out delta_ea_scatter.svg
python scripts/review_open_world_o2_analyze.py manifest --run-root $R --parents $P --out-dir .
```

- **重跑**：两次重跑的 `results.json` 逐字节相同，sha256 为 `eb99b097…`。
- **输入哈希**：`manifest.json` 与 `SHA256SUMS.remote` 记录了全部 159 个原始输入的 sha256，包括每项的 verdict、manifest、environment、execution 文件，以及台账和 Slurm 终态快照。
- **n2 测试**（`e45f85d`）：bootstrap 458 个文件通过，reusable unittest OK，pytest 331 通过。

## 9. 本目录文件

| 文件 | 内容 |
|---|---|
| `results.json` | 全部指标与逐项记录（逐字节可复现） |
| `delta_ea_scatter.svg` | ΔEa_xTB 对 ΔEa_DFT 的散点图（只作描述） |
| `manifest.json`、`SHA256SUMS.remote` | 来源提交、配置哈希、原始输入哈希、Slurm 作业号 |
| `DECISION_ZH.md`、`INFRASTRUCTURE_ZH.md`、`DECISION_NVML_ZH.md` | 执行过程中的三份决定与事件记录 |
| `SHA256SUMS` | 本目录其他文件的哈希 |
