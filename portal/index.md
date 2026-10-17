---
hide:
  - navigation
  - toc
---

# xtbflow 项目主页

<p class="xf-lead">守恒事件—几何<b>联合</b>流匹配：只给反应物，同时生成“哪些键断、哪些键成”和对应的过渡态几何，并检验物理信号能否在生成过程中改写反应通道。</p>

<p class="xf-byline">状态截至 2026-10-17 · 仓库 <a href="https://github.com/askusk521-cyber/xtbflow">askusk521-cyber/xtbflow</a> · 本站只汇总、不产生新结论，数字以所链报告为准</p>

<!-- viz:kpis -->

## 研究路线与各阶段结论

<!-- viz:roadmap -->

<p class="xf-muted xf-small">卡片左侧色条：绿 = 通过，红 = 未通过，黄 = 暂缓或草案，灰 = 探索性结论，蓝 = 运行中。点击卡片进入论文中对应的小节。</p>

## 一段话现状

!!! abstract "主线与主要发现"

    **主线**：M0 闸门 `NO-GO`（10-07，无引导时联合没有比串联更好，区间包含 0）→ V1a 代理筛查 `GO_V1b`（10-07）→ V1b 完成 D0/D1 后由负责人暂缓（10-08）。

    **之后都是探索性检查**，主要发现：

    - 无引导联合 B0 的单位预算效率最高，但优势可由积分步数解释；同候选数时单候选质量与串联无差别。
    - 几何引导（B1）在任何验证成本下都不在前沿。
    - 真实 TS 几何没有可检出的排序增益；物理信号的可用窗口最多约 35–38%。
    - 生成中按物理筛选没有胜过事后排序。
    - 生成胜过两种枚举基线（b3f3 + 势垒头；b2f2 + AIMNet2 热力学筛选）。

    **正在运行**：开放世界 DFT 认证（O2）剩余 28 项，见[进展与下一步](progress.md)。

## 从这里开始

<div class="grid cards" markdown>

-   :material-file-document-outline:{ .lg .middle } **论文式说明**

    ---

    摘要、引言、方法与实验设计、结果、讨论与证据边界，按论文结构把项目讲清楚。

    [:octicons-arrow-right-24: 阅读摘要](paper/index.md)

-   :material-progress-clock:{ .lg .middle } **进展与下一步**

    ---

    正在 n2 上运行的实验、进度与日志位置；需要负责人决定的事项。

    [:octicons-arrow-right-24: 查看进展](progress.md)

-   :material-database-search-outline:{ .lg .middle } **数据与证据**

    ---

    每个实验的报告、结果 JSON（固定到提交的永久链接）和 n2 原始输出目录，可筛选。

    [:octicons-arrow-right-24: 打开数据目录](data.md)

-   :material-bookshelf:{ .lg .middle } **参考文献**

    ---

    数据集、相关生成模型、量子化学软件文档、鞍点理论与统计方法。

    [:octicons-arrow-right-24: 查看参考文献](references.md)

-   :material-archive-outline:{ .lg .middle } **历史存档**

    ---

    10-06 之前的启动包、P0–P3 工程合同、Workflow A–D、五条推进线。

    [:octicons-arrow-right-24: 浏览存档](archive.md)

-   :material-book-alphabet:{ .lg .middle } **术语表**

    ---

    A0/A1/A2/B0/B1、Hit@32、AUC、M₀/Mᵣ、D0–D5 等缩写的含义。

    [:octicons-arrow-right-24: 查术语](glossary.md)

</div>
