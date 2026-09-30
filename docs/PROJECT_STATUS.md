# xtbflow 当前项目状态

核查基准：2026-09-30。源码基线为 `ec803556605465b6309f32a15a529c2f9470679a`，即
远端 `origin/main` 当前提交。本页是活动状态入口；初始化交接、旧资产索引和已结束的
行动清单保留在 [`archive/handoffs/`](../archive/handoffs/) 中，不再作为当前执行指令。

## 仓库与分支

- 本地新工作线从最新 `origin/main` 创建，工作树初始干净。
- GitHub 远端 `origin/main` 与本地同步基线均为 `ec803556`。
- 已合并的基础 PR 是 #65（人工合并审批规则）和 #66（验证、实验控制与运行保护）。
- 当前开放的工作线仍按依赖推进：#63 数据／尝试记录合同，#64 端点与参考验证，#67
  Transition1x 隔离开发诊断，#68 CP2K 重放，#69 CP2K–GFN2 桥接，#70 批量预检，#71
  Reaction-QM 来源审计，以及 #61 研究治理。它们未因列在本页而获得合并授权。
- 堆叠 PR 的 CPU CI 触发范围仍待单独集成；在合并前必须确认目标工作分支也收到检查。

## n2 观察结果

本轮通过 `ssh n2` 只读核查，真实 Git checkout 是
`~/xtbflow/xtbflow`，不是 `~/xtbflow` 顶层目录。该 checkout 当前为
`codex/track-b-real-development@fb6b234`，工作树干净，已 fetch 到
`origin/main@ec803556`，因此本地 checkout 落后主线 3 个提交。现有历史分支和执行
worktree 不应直接 reset 或覆盖；新的真实作业应从目标提交创建独立执行目录。

观测到的 `xtbflow` 环境为 Python 3.10.19、PyTorch 2.10.0+cu128、NumPy 1.26.4、
SciPy 1.15.3、RDKit 2024.3.3、ASE 3.29.0、tblite 0.7.0、pytest 9.1.1 和 PyYAML
6.0.3。节点 Slurm 为 21.08.5，`main` 分区提供 `gpu:pro6000:4`、192 CPU 和约
500000 MiB 内存；未通过 Slurm 分配时 `torch.cuda.is_available()` 为 false。当前
shell 的 PATH 中没有 `xtb`、`cp2k.psmp` 或 `cp2k.popt`，所以不能把包导入或缓存存在
写成已完成的 xTB/CP2K 运行资格。

n2 的 `~/.cache/xtbflow` 约 3.6G，包含 Reaction-QM v2、RGD1 v6 和 Transition1x
缓存。原始字节仍只保留在 n2 缓存，仓库只提交来源、哈希、审计和合同；缓存内容不应
复制入 Git。

## 当前科学边界

- #66 提供软件控制、预算结算、失败证据和无守恒对照的基础；它没有完成真实
  independent／serial／joint 训练，也没有证明联合架构领先。
- Reaction-QM/RGD1 的公开资产审计应先完成坐标到 map 的独立证据、电子态、分组和
  overlap audit；缺失字段保持 `quarantine`，不得猜填。
- Transition1x、SPICE E/F 和历史 CP2K/GFN2 记录各自有明确用途，不能跨来源拼接成
  反应监督或 confirmatory 结论。

## 下一步交付

1. 从本页所示主线创建独立工作树，并将堆叠 PR 的 CPU CI 触发修复提交为单独 PR。
2. 修复 #64 的逐端物种判定和 #70 的批次总预算／失败状态机；保留每次已消费成本。
3. 收缩 #71 的准入边界，使来源记录保持 quarantine，由统一 Track-B 适配器授予准入。
4. 在数据准入和评估协议冻结后，分别运行 independent、serial、joint 及无守恒对照。

真实训练、GPU、GFN2、CP2K 和数据下载必须在记录了 commit、协议哈希、环境、资源
上限和输出位置的 n2 作业中执行；本页不构成这些科学结果的证据。
