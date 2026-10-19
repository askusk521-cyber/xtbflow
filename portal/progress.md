---
hide:
  - navigation
---

# 进展与下一步

本页回答两个问题：**现在 n2 上在跑什么**，以及**接下来暂定做什么**。“正在运行”由本机定时任务每 30 分钟从 n2 采集（Slurm 队列与登记的长任务日志），写入置顶的[运行状态 issue](https://github.com/askusk521-cyber/xtbflow/issues?q=is%3Aissue+label%3Arun-status)，网站构建时读取，不需要改仓库。“最近完成”与“下一步”来自 [`portal/records/progress.yaml`](https://github.com/askusk521-cyber/xtbflow/blob/main/portal/records/progress.yaml)，改动走 PR。

## 正在运行（自动更新） {#running}

<!-- viz:live -->

## 最近完成的实验 {#recent}

<!-- viz:running -->

!!! info "这项实验在做什么"

    开放世界 DFT 认证（O2）用冻结的 DFT 协议，认证 O1 中通过 xTB 认证的目录外通道，同时核查 AIMNet2 的训练集是否覆盖这些分子。它在 10-10 因执行脚本调用 `nvidia-smi` 失败而停住（n2 的 NVML 用户态库与内核模块版本不一致，GPU 计算本身正常）。负责人 10-17 批准了一个带版本号的最小改动：执行脚本与调度器改用 CUDA 运行时记录 GPU 信息，`qc_protocol.py`、样本和设置都不变。10-18 全部跑完，10-19 完成配对分析与报告。结果与读法见论文 [3.8 节](paper/results.md#open-world)。

## 暂定的下一步 {#next}

下面按依赖排列。标为“需负责人决定”的事项，在得到明确决定之前不会执行。

<!-- viz:next -->

## 主线阶段回顾

<!-- viz:roadmap -->
