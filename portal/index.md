---
hide:
  - navigation
---

# xtbflow：让物理信号在生成过程中挑选反应通道

<p class="xf-lead">只给一个反应物分子，让模型同时提出“它会怎么反应”和“反应要翻过的能量山口长什么样”，并尽早找到最容易发生的那几条反应。这个页面用五分钟讲清楚：为什么做、想证明什么、走到了哪里、现在要在哪个路口做选择。</p>

<p class="xf-byline">状态截至 2026-10-19 · 仓库 <a href="https://github.com/askusk521-cyber/xtbflow">askusk521-cyber/xtbflow</a> · 本站只汇总、不产生新结论，数字以所链报告为准</p>

## 一　背景：要解决什么问题 {#background}

化学反应从**反应物**出发，翻过一个能量“山口”到达**产物**。山口的最高点叫**过渡态**（transition state，TS），山口相对反应物的高度叫**势垒**：势垒越低，反应越容易发生。

同一个分子通常有许多条可能的反应路线，本站称为**通道**。每条通道由两样东西确定：

- **化学事件**：哪些化学键断开、哪些化学键形成；
- **过渡态几何**：翻越山口那一刻，所有原子的三维位置。

<figure class="xf-figure" id="fig-home-1" markdown="0">
<!-- viz:figure:reaction_channels -->
<figcaption><b>图 1</b>　同一个反应物的三条通道（示意）。真正关心的是势垒最低的通道 3，但事先并不知道一共有几条通道、各自通向什么产物。</figcaption>
</figure>

传统做法用量子化学（例如 DFT）逐条搜索过渡态。它足够准确，但本项目的认证计算在 GPU 上每条通道仍要十几分钟到近两小时，而且要先猜出该搜哪几条通道。

近年的机器学习生成模型可以直接“画”出过渡态几何，但多数要求**同时给出反应物和产物**。这等于事先已经知道答案是哪条通道。

本项目做更难、也更接近真实发现的**单端设定**：

- 只给反应物，模型自己提出化学事件和过渡态几何；
- 要从许多候选通道里尽早找到势垒最低的几条。

数据来自公开数据集 Transition1x：约一万个有 DFT 能量的反应，分子最多 23 个原子。详见[引言](paper/introduction.md)。

## 二　目标：想证明什么 {#goal}

生成一条通道有两种组织方式：

- **串联**：先决定化学事件，再按这个事件生成过渡态几何。
- **联合**（也称并联）：一个网络在同一个生成过程里同时生成事件和几何，两者边生成边互相影响。

<figure class="xf-figure" id="fig-home-2" markdown="0">
<!-- viz:figure:serial_vs_joint -->
<figcaption><b>图 2</b>　物理信号能在哪里介入。串联中，事件在生成几何之前就已定死，物理信号只能对做完的候选事后打分；联合中，信号在事件尚未定型时作用于几何，有机会把生成“推”到另一条通道上。</figcaption>
</figure>

理论上，如果两种网络都完美，它们表达的是同一个概率分布，所以**不该期待联合“天生更准”**。联合真正独有的是**生成过程中可以被改写**。

!!! abstract "核心主张（主假设 H1）"

    在相同计算预算下，在生成过程中用势垒信号推动正在成形的过渡态几何，联合模型能让低势垒通道被**更早、更多**地找到；串联拿到同样的信号，也只能在事件定死之后重排。

**成功的标准**包括三点：

- 这个优势出现在资源和输入都对齐的对照中；
- 生成的过渡态要经 DFT 认证为真实的山口，而不只是“长得像”数据集里的参考；
- 最后在一直保留的官方测试集上确认一次。

联合还有另外三种按构造才有的能力（双向补全、生成中纠错、更省计算），列为备选卖点，见[引言 1.3 节](paper/introduction.md#operations)。

## 三　大致路线：走到了哪里 {#route}

项目按“闸门”推进：每一步的通过标准在看到数据之前写定，过了才进入下一步。

<div class="xf-journey" markdown>

<div class="xf-step xf-step-nogo" markdown>
<div class="xf-step-when">10-07 · M0</div>

**基础对比：不加引导，联合 vs 串联** <span class="xf-badge xf-nogo"><span aria-hidden="true">=</span>打平</span>

不加任何引导时，两者找对通道的比例没有差别：联合低 1.5 个百分点，区间 −6.7 到 +3.3，包含 0。这说明联合本身并不更准，与理论预期一致。负责人决定保留联合作为前提，转而检验它独有的能力。[详情](paper/results.md#m0)
</div>

<div class="xf-step xf-step-go" markdown>
<div class="xf-step-when">10-07 · V1a</div>

**代理筛查：加入几何引导** <span class="xf-badge xf-go"><span aria-hidden="true">✓</span>通过</span>

加入几何引导后，联合发现已知通道的效率高于最强的串联做法。但这一步只和数据集里的参考结构比“像不像”，未经 DFT 确认。而且不加引导的联合，效率比加引导的还高。[详情](paper/results.md#v1a)
</div>

<div class="xf-step xf-step-paused" markdown>
<div class="xf-step-when">10-08 · V1b</div>

**DFT 认证：上一步的优势是真的吗** <span class="xf-badge xf-paused"><span aria-hidden="true">‖</span>暂缓</span>

计划用 DFT 逐条确认生成的过渡态是不是真实的山口。8 条开发链已经跑通。正式运行约需 230–330 GPU·h，由负责人暂缓。[详情](paper/results.md#v1b)
</div>

<div class="xf-step xf-step-explore" markdown>
<div class="xf-step-when">10-08 → 10-19</div>

**探索性检查：优势从哪里来** <span class="xf-badge xf-explore"><span aria-hidden="true">◇</span>16 项已完成</span>

暂缓期间做了 16 项检查，追问两件事：物理信号在生成过程中到底有没有用？生成比“规则枚举再打分”强在哪？这些检查只用来定方向，不算确认性证据。结果汇总在[下一节](#crossroads)。
</div>

<div class="xf-step xf-step-here" markdown>
<div class="xf-step-when">现在</div>

**十字路口：选下一阶段押在哪里** <span class="xf-badge xf-running"><span aria-hidden="true">●</span>你在这里</span>

主线停在 V1b 之前。n2 上目前没有本项目的作业。
</div>

<div class="xf-step xf-step-todo" markdown>
<div class="xf-step-when">之后</div>

**独立确认与成文** <span class="xf-badge xf-explore"><span aria-hidden="true">○</span>未开始</span>

选定方向后，在保留的官方测试集上做一次定稿前的确认性评测，再写成论文。
</div>

</div>

## 四　当前的十字路口 {#crossroads}

16 项探索性检查的结果分成两类：对主假设不利的，和对项目有利的。

<div class="xf-xr-cols" markdown>

<div class="xf-xr-col xf-xr-con" markdown>
<div class="xf-xr-head">✕　削弱主假设“物理在生成中挑通道”</div>

- **引导不划算。** 把验证候选的成本也算进去后，加引导的联合在任何成本水平下都不是最优选择。不加引导的联合效率高，但这可以由积分步数不同来解释。（[#107][pr107]、[#109][pr109]）
- **几何里可用的信息少。** 直接用真实过渡态几何给事件排序，也没有可检出的增益。物理信号能起作用的生成时段，最多约占全程的 35–38%。（[#110][pr110]–[#112][pr112]）
- **能改变通道，但没有方向。** 引导确实改变了生成的事件，却没有把它推向低势垒。（[#112][pr112]）
- **生成中筛选不如事后排序。** 梯度引导、粒子重采样、物理打分加权，都没有胜过“先生成、再排序”。预写建议是关闭这条研究线。（[#116][pr116]–[#118][pr118]、[#121][pr121]）
</div>

<div class="xf-xr-col xf-xr-pro" markdown>
<div class="xf-xr-head">✓　对项目有利</div>

- **生成胜过规则枚举。** 两种“枚举所有可能的断键方式，再打分”的基线，找通道的效率都输给生成。（[#113][pr113]、[#120][pr120]）
- **联合更省计算。** 每个模态积分步数相同时，联合每个候选的开销约为串联的一半。（[#109][pr109]）
- **能找到数据集之外的真实通道。** 10 条数据集里没有的候选，有 4 条通过 DFT 严格认证。但这 4 条的势垒都不比同一分子已知的最佳通道低。（[O2][pr119]）
- **有更好的打分器。** 机器学习势 AIMNet2 给通道排序的效果优于半经验方法 GFN2-xTB。（[#114][pr114]）
</div>

</div>

!!! question "路口上的问题"

    “物理信号在生成过程中改写通道”这一主假设，目前的探索性证据大多不支持；而“生成”本身作为找通道的工具，表现不错。**联合生成的价值，下一阶段押在哪里？**

下面三条路由本站根据已有文档和结果整理，供讨论。选哪条、是否组合，由负责人决定。

<div class="xf-roads" markdown>

<div class="xf-road" markdown>
<div class="xf-road-tag">路线 A</div>

**回到主线：恢复 V1b 的 DFT 认证**

- **回答**：V1a 的代理优势在 DFT 下是否成立。
- **代价**：约 230–330 GPU·h，可同时用 2 张卡。开跑前要定版本（两臂或四臂）、端点软模的统一处理规则，并冻结判定规则。
- **注意**：之后的分析显示，加引导的联合在计入验证成本后不在效率前沿（[#107][pr107]）。

[V1b 待定事项](progress.md#next)
</div>

<div class="xf-road" markdown>
<div class="xf-road-tag">路线 B</div>

**换卖点：检验联合的其他独有能力**

- **回答**：联合的价值是否在双向补全（H2）、生成中纠错（H3）或更省计算（H4）上。
- **依据**：方向规划中的备选卖点；[#109][pr109] 显示每候选开销约为串联一半。
- **代价**：需要新训练；规划估计第一组实验约 36 GPU·h。目前尚无实验。

[四种独有能力](paper/introduction.md#operations)
</div>

<div class="xf-road" markdown>
<div class="xf-road-tag">路线 C</div>

**转向发现：以“生成 vs 枚举”找新通道为主线**

- **回答**：生成能否找到比已知最佳通道**更低**的新通道（O2 中 0/4）。
- **依据**：[#113][pr113]、[#120][pr120] 的枚举对照，以及 O1/O2 的开放世界认证。
- **注意**：已有规划文档里没有这个方向，是本站按结果列出的。

[开放世界结果](paper/results.md#open-world)
</div>

</div>

无论走哪条路，都有几件事等负责人先拍板：

- 生成中筛选研究线是否关闭；
- O2 是否需要独立验收；
- 22 个未合并的 PR 怎么处理。

完整清单见[进展与下一步](progress.md#next)。

## 继续阅读 {#more}

<div class="grid cards" markdown>

-   :material-file-document-outline:{ .lg .middle } **论文式说明**

    ---

    方法、每个实验的设计与结果、证据边界，按论文结构展开。

    [:octicons-arrow-right-24: 阅读摘要](paper/index.md)

-   :material-progress-clock:{ .lg .middle } **进展与下一步**

    ---

    n2 上正在跑什么（每天自动更新）、最近完成的实验、待决事项。

    [:octicons-arrow-right-24: 查看进展](progress.md)

-   :material-database-search-outline:{ .lg .middle } **数据与证据**

    ---

    每个实验的报告、结果文件和 n2 原始输出目录，可筛选。

    [:octicons-arrow-right-24: 打开数据目录](data.md)

-   :material-bookshelf:{ .lg .middle } **参考文献**

    ---

    数据集、相关生成模型、量子化学软件、鞍点理论与统计方法。

    [:octicons-arrow-right-24: 查看参考文献](references.md)

-   :material-archive-outline:{ .lg .middle } **历史存档**

    ---

    10-06 之前的启动包、工程合同、早期工作流与五条推进线。

    [:octicons-arrow-right-24: 浏览存档](archive.md)

-   :material-book-alphabet:{ .lg .middle } **术语表**

    ---

    M0、V1a、B0/B1、Hit@32 等缩写的含义。

    [:octicons-arrow-right-24: 查术语](glossary.md)

</div>

[pr107]: https://github.com/askusk521-cyber/xtbflow/pull/107
[pr109]: https://github.com/askusk521-cyber/xtbflow/pull/109
[pr110]: https://github.com/askusk521-cyber/xtbflow/pull/110
[pr112]: https://github.com/askusk521-cyber/xtbflow/pull/112
[pr113]: https://github.com/askusk521-cyber/xtbflow/pull/113
[pr114]: https://github.com/askusk521-cyber/xtbflow/pull/114
[pr116]: https://github.com/askusk521-cyber/xtbflow/pull/116
[pr118]: https://github.com/askusk521-cyber/xtbflow/pull/118
[pr119]: https://github.com/askusk521-cyber/xtbflow/pull/119
[pr120]: https://github.com/askusk521-cyber/xtbflow/pull/120
[pr121]: https://github.com/askusk521-cyber/xtbflow/pull/121
