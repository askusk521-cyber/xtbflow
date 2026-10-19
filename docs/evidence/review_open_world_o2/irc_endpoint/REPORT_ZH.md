# O2 补充诊断：两条 IRC 达到上限的候选链

本报告全部结果为探索性，属于补充诊断，不改变 O2 主报告（`../REPORT_ZH.md`）和 `../results.json` 中的任何计数。

- **4f55（xTB 认为比已知通道低 0.60 kcal/mol 的那条）**：补充判定为 `IRC_REACTANT_MISMATCH`。
  - DFT 鞍点的一端正是预测的产物，另一端是第三种异构体，两端都不是反应物。
  - 所以这个鞍点属于"预测产物 ↔ 另一异构体"这一步，不是"反应物 → 预测产物"。预测通道在 DFT 下的 TS 仍未找到。
  - O2 中这条候选的 TS 比基准 TS 高 43.76 kcal/mol，这个差值**不是**预测通道的势垒差。
- **7dc6**：补充判定为 `STRICT_JOINT_GRAPH_VALID`，两端分别是反应物和预测产物。
  - TS 比同母体基准 TS 高 22.64 kcal/mol，xTB 的值是 +23.73。
  - 但反应物一侧的连接主要靠几何优化而不是 IRC：IRC 停在 TS 以下仅 2.3 kcal/mol 处，之后优化又下降 98.3 kcal/mol、移动 0.65 Å 才到达反应物。
- **算力**：两项共 752 GPU 秒（Slurm 3490、3491），在 2 GPU 小时上限内。

## 1. 问题与依据

**背景**：O2 中这两条候选链的 TS 优化与频率都通过了，但反向 IRC 用满了每方向 300 步（`IRC_LIMIT`）。按协议 `v1b-h-gpu4pyscf-3`，它们记为 `PROTOCOL_FAILURE`，O2 主结果保持不变。

**负责人批准（2026-10-19）**："批准：先写好补充诊断规则并推送，再在 Slurm 上单卡运行：从 4f55 和 7dc6 的 IRC 两端最后一帧优化到极小点、算频率、判图身份。结果标为补充诊断，不改 O2 主结论。"

**规则先于计算**：
- 规则写在 `configs/review/open_world_o2_irc_endpoint.json`，与执行脚本一起于 `b6b1cc7` 推送，之后才提交作业。
- 写规则时，作者已看过两条链的 O2 判定、`irc.log` 末尾、轨迹各帧能量，以及这两个母体的 TS、基准和锚点能量；当时还没有优化任何端点。

**事先已知的一点**：两条候选的 DFT TS 能量在 O2 中已经确定。所以这次诊断只能判断鞍点是否连接预测事件，不可能让候选变得比基准更低。这一点在规则文件的 `already_known` 字段里写明。

## 2. 做法

**取哪两帧**：存档轨迹 `irc/irc_irc.xyz` 的第一帧和最后一帧。
- geomeTRIC 1.1.1 在双向 IRC 换方向时会把正向帧倒序，所以文件依次是正向端、TS、反向端。
- 这两帧就是 IRC 收敛时 `certification_chain` 本来会拿去优化的两帧。

**帧检查**（不通过就在任何量化计算之前停下）：
- 只有一帧标为 `Iteration 0`；
- 它的能量和坐标与 O2 存档 TS 一致，坐标差约 5×10⁻¹¹ Å；
- 每帧的元素与计划项一致。

**计算**：在干净克隆 `source-b6b1cc7` 中执行，其 `qc_protocol.py` 的哈希与 O2 计划记录的相同。
- 先后对两帧调用未修改的 `qc_protocol.minimum_stage`（TightOpt，200 步，带协议内置的一次统一重试，解析 Hessian 频率）；
- 两端都是极小点后，用计划项中的反应物图、原子置换和预测产物图调用未修改的 `endpoint_identity`；
- 以上与 `certification_chain` 第 326–336 行的调用和顺序完全相同。

**执行**：每项一个 Slurm 作业，单卡，时限 1 小时，全用户串行，4f55 先跑。每项只尝试一次。

## 3. 结果

| 母体 | 补充判定 | ΔEa_xTB | DFT TS − 基准 TS | DFT TS − 锚点 | xTB TS 与 DFT TS 的 RMSD |
|---|---|---:|---:|---:|---:|
| `4f5559117464139fee00`（C5H6N2，13 原子） | `IRC_REACTANT_MISMATCH` | −0.60 | +43.76 | 105.05 | 0.149 Å |
| `7dc6922783059ead3fb0`（C6H8O，15 原子） | `STRICT_JOINT_GRAPH_VALID` | +23.73 | +22.64 | 100.57 | 0.520 Å |

单位为 kcal/mol。RMSD 为同原子顺序下最优旋转后的值。

### 3.1 4f55

| 端 | 取自 IRC 第几步 | 该帧相对 TS | 优化后再下降 | 帧到极小点 RMSD | 极小点 | 相对锚点 | 图身份 |
|---|---:|---:|---:|---:|---|---:|---|
| 第一帧（正向，IRC 已收敛） | 206 | −71.15 | 0.00 | 0.0003 Å | `MINIMUM_PASS`，最低频 165 cm⁻¹ | +33.90 | 第三种异构体 |
| 最后一帧（反向，停在上限） | 301 | −74.29 | 0.07 | 0.040 Å | `MINIMUM_PASS`，最低频 117 cm⁻¹ | +30.69 | 预测产物 |

**说明**：
- 反向 IRC 虽然用满了步数，最后一帧其实已经在极小点附近：再优化只下降 0.07 kcal/mol、移动 0.04 Å。
- 两端都不是反应物，这与"IRC 没走完"无关。

**图比较**（描述性，写于计算之后，脚本见 `02d2e31` 的 `graphs` 子命令，结果在 `graphs.json`）：
- 预测事件相对反应物：N0–C1 由三键变双键，C1–C2 由单键变双键，H7 从 C2 移到 N0。
- 最后一帧的极小点与预测产物的图完全相同。
- 第一帧的极小点与预测产物相比：H7 从 N0 移到 C1，新增 N0–C2 键，C1–C2 由双键变单键。

### 3.2 7dc6

| 端 | 取自 IRC 第几步 | 该帧相对 TS | 优化后再下降 | 帧到极小点 RMSD | 极小点 | 相对锚点 | 图身份 |
|---|---:|---:|---:|---:|---|---:|---|
| 第一帧（正向，IRC 已收敛） | 35 | −0.49 | 0.00 | 0.001 Å | `MINIMUM_PASS`，最低频 58 cm⁻¹ | +100.08 | 预测产物 |
| 最后一帧（反向，停在上限） | 301 | −2.31 | 98.26 | 0.654 Å | `MINIMUM_PASS`，最低频 131 cm⁻¹ | 0.00 | 反应物 |

**说明**：
- 反应物一端优化后的能量与 O2 锚点相差 7.6×10⁻⁸ Eh。
- 局部势垒（TS 减反应物端）为 100.57 kcal/mol。
- 预测产物只比 TS 低 0.49 kcal/mol，是一个很浅的极小点。
- 反向 IRC 走了 300 步仍只比 TS 低 2.3 kcal/mol。反应物这一侧的连通性主要来自从该帧出发的几何优化，而不是 IRC。这是补充判定与严格通过的主要区别。

## 4. 对 O2 结论的影响

**主结果不变**：候选 4/10 严格通过，配对母体 4 个，各项计数照旧。

**补充视角**（不进入主结果）：
- 若把 7dc6 也算作连通，在 DFT 下与预测事件相连的候选有 5 条。它们的 TS 都比同母体基准高，范围 +8.8 到 +54.7 kcal/mol。
- xTB 认为与已知通道差距在 3 kcal/mol 以内的 4 条候选，在 DFT 下的情况：
  - 2 条严格通过，分别高 8.8、12.3；
  - 1 条端点身份无法判定（8fe6），TS 高 9.6；
  - 1 条（4f55）在 xTB TS 附近（RMSD 0.149 Å）找到的 DFT 鞍点不连接反应物，xTB 认证的事件没有在 DFT 中复现。

**对主报告的更正**：O2 主报告第 3 节原写"它在 DFT 层面没有得到检验"，已改为带日期的更正，并指向本报告。

## 5. 限制

- **只是补充判定，不是严格通过**：端点由几何优化而不是 IRC 到达。7dc6 的反应物一侧尤其如此，优化跨过了 98 kcal/mol 的能量下降。
- **结果相关的信息**：写规则时已看过轨迹的帧能量。图比较是在计算之后加的描述，不参与判定。
- **驱动版本不同**：n2 于 2026-10-19 17:19 重启，内核驱动从 O2 时的 595.91.07 变为 595.99.02（`nvidia-smi` 因此恢复正常）。cupy、gpu4pyscf、pyscf、geomeTRIC 的版本和 `qc_protocol.py` 都没有变。7dc6 反应物端与 O2 锚点的能量差为 7.6×10⁻⁸ Eh，在 TightOpt 收敛容差之内。
- **能量口径**：气相电子能量，ωB97X/6-31G(d)，没有零点能、热校正和溶剂。
- **样本**：n = 2，不能据此推广。
- 不作任何化学结论。

## 6. 复现

在 n2 上，于干净克隆 `source-02d2e31` 中运行（CPU；`graphs` 需要 RDKit，用 `xtbflow-qc` 环境）：

```bash
export PYTHONPATH=src:vendor/mechai_reusable CUDA_VISIBLE_DEVICES=
R=/home/lhshen/xtbflow-runs/followup-20261009
python scripts/review_open_world_o2_irc_endpoint.py summarize --run-root $R --out results_irc_endpoint.json
/home/lhshen/miniconda3/envs/xtbflow-qc/bin/python scripts/review_open_world_o2_irc_endpoint.py graphs --run-root $R --out graphs.json
```

- **重跑结果**：
  - `summarize` 在 `b6b1cc7` 上跑两次、在 `02d2e31` 上跑一次，三次输出逐字节相同（sha256 `819176bc…`）；
  - `graphs` 在 `02d2e31` 上跑两次，输出相同（`36c7c978…`）。
- **n2 测试**（`02d2e31`）：bootstrap 467 个文件通过，reusable unittest 通过，pytest 336 通过。
- **执行命令**（已执行，单次使用）：

  ```bash
  python scripts/review_open_world_o2_irc_endpoint.py submit --run-root $R --source $R/source-b6b1cc7
  ```

## 7. 本目录文件

| 文件 | 内容 |
|---|---|
| `results_irc_endpoint.json` | 判定、端点能量与频率、TS 能量差、RMSD、用时、全部输入的 sha256 |
| `graphs.json` | 端点图与反应物、预测产物的键级差异（描述性） |
| `SHA256SUMS.remote` | n2 上 34 个原始文件的 sha256：输入的计划、O2 判定、IRC 轨迹，以及补充运行的全部输出、台账和 Slurm 终态快照 |
| `SHA256SUMS` | 本目录其他文件的哈希 |
