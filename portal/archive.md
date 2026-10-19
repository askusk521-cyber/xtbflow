---
hide:
  - navigation
---

# 历史存档

2026-10-06 起，项目改按 M0 → V1a → V1b 阶段指导推进。在此之前约两周的工作——启动包、工程合同、资格测试、并行工作流和推进路线——保留在这里。它们记录历史事实与边界，**不是现行执行入口**；保留原文也不意味着接受其中所有方案为已验证的结论。

<!-- viz:legacy -->

## 这些遗产在现行工作中的位置 {#reuse}

| 遗产 | 现行工作中的用途 |
|---|---|
| `vendor/mechai_reusable/` | 旧工程 mechai 的可复用代码，原样保存，不可原地修改 |
| 守恒事件流、端点解码器（#35） | M0 沿用其中的 `decode_endpoint` 修复与校验；事件表示改为完整 N×N 对称矩阵 |
| `configs/models/xtbflow_v0.1.yaml` | M0 骨干规格（6 层，128 / 32 通道）与它一致 |
| 合并审批规则（#65） | 至今有效：所有合并须负责人明确批准 |
| 早期资格测试证据（`docs/evidence/*.json`） | GFN2、SPICE、CP2K 的资格结论；不是反应监督数据，不能跨来源拼接成确认性结论 |
| 五条推进线（#93–#98）与执行图（#74–#86） | 10-06 后保持原状，未回写；现行状态以本站与 [`docs/STATUS_ZH.md`](https://github.com/askusk521-cyber/xtbflow/blob/main/docs/STATUS_ZH.md) 为准 |
