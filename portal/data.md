---
hide:
  - navigation
  - toc
---

# 数据与证据

每个实验的证据分三层，越往下越原始：

1. **报告**：`REPORT_ZH.md` 等中文报告，写明问题、做法、结果与读法；
2. **结果文件**：`results.json`、`gate.json`、`SHA256SUMS` 等，提交在对应 PR 的分支上；
3. **原始输出**：模型检查点、采样流、量化计算目录，留在 n2 的 `/home/lhshen/xtbflow-runs/` 下，不入库，仓库只提交哈希。

表中的证据链接都**固定到 PR 的头提交**（`blob/<提交>/…`），分支以后移动或删除，链接仍然有效。原始输出路径相对于 `/home/lhshen/xtbflow-runs/`；悬停可看完整路径。

## 实验数据目录 {#catalog}

<!-- viz:catalog -->

## 数据集与公开清单 {#datasets}

| 数据 | 用途 | 位置 |
|---|---|---|
| Transition1x 官方 HDF5 | 全部生成实验的数据来源 | n2 缓存 `~/.cache/xtbflow`；来源与哈希见 [`docs/PROVENANCE.md`](https://github.com/askusk521-cyber/xtbflow/blob/main/docs/PROVENANCE.md) |
| 公开数据清单 | 来源、许可与准入记录 | [`configs/public_data_manifest.yaml`](https://github.com/askusk521-cyber/xtbflow/blob/main/configs/public_data_manifest.yaml)、[`data/manifests/`](https://github.com/askusk521-cyber/xtbflow/tree/main/data/manifests) |
| 配对试点 v1 | 10-02 的事件—TS 配对开发集 | [`docs/data/paired_pilot_v1.md`](https://github.com/askusk521-cyber/xtbflow/blob/main/docs/data/paired_pilot_v1.md)、[`data/pairs/`](https://github.com/askusk521-cyber/xtbflow/tree/main/data/pairs) |
| 早期资格测试证据 | GFN2、SPICE、CP2K、数据审计（09-27 至 10-02） | [`docs/evidence/`](https://github.com/askusk521-cyber/xtbflow/tree/main/docs/evidence)，见[历史存档](archive.md) |

## PR 叠放关系 {#pr-stack}

10-06 以后的工作都在未合并的 PR 上，按分支叠放。合并须负责人明确批准。

```text
main
├─ #101 m0/task0-env ← #103 task1-data ← #102 task2-model ← #104 task3-train ← #105 task4-eval
└─ #106 codex/v1a-20261007（基于 M0 稳定化快照，缺 #105 最后 5 个提交）
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

## 规范文档 {#guides}

| 文档 | 用途 |
|---|---|
| [课题方向规划](https://github.com/askusk521-cyber/xtbflow/blob/main/docs/guides/DIRECTION_PLAN_ZH.md) | 研究方向、实验组 E1–E5、决策树、与竞争工作的差异 |
| [M0 阶段执行指导](https://github.com/askusk521-cyber/xtbflow/blob/main/docs/guides/M0_GUIDE_ZH.md) | M0 任务 0–4 与去/留闸门 |
| [V1a 指导 v2.1](https://github.com/askusk521-cyber/xtbflow/blob/ce76584fde7af54297ea7902c96dc018d1c6ca2e/docs/v1a/protocol-v2.1.md) | V1a 代理筛查 |
| [V1b 指导 v2.1](https://github.com/askusk521-cyber/xtbflow/blob/main/docs/guides/V1B_GUIDE_v2.1_ZH.md) | V1b DFT 认证 |
| [V1a 后续任务交接](https://github.com/askusk521-cyber/xtbflow/blob/main/docs/guides/V1A_FOLLOWUP_HANDOFF_ZH.md) | 任务 R/N/V（#107–#109） |
| [综述对齐交接](https://github.com/askusk521-cyber/xtbflow/blob/main/docs/handoffs/HANDOFF_REVIEW_ALIGNMENT_ZH.md) | #113–#115 的执行规范 |
| [SMC 筛选交接](https://github.com/askusk521-cyber/xtbflow/blob/main/docs/handoffs/HANDOFF_SMC_SELECTION_ZH.md) | #116–#118 的执行规范 |
| [三项后续任务交接](https://github.com/askusk521-cyber/xtbflow/blob/main/docs/handoffs/HANDOFF_FOLLOWUP_ZH.md) | #119–#121 的执行规范 |
| [项目现状索引](https://github.com/askusk521-cyber/xtbflow/blob/main/docs/STATUS_ZH.md) | 本站的文字版来源 |
