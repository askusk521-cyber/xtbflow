# 主要先行工作与定位依据

核查日期：2026-09-25。本表是定向检索起点，不是穷尽的新颖性审查；正式立项和投稿前需更新。优先使用下列原始论文/官方文档，不从二手介绍抄录能力边界。

| 工作 | 本次核实的相关点 | 原始来源标识 |
|---|---|---|
| FlowER，Joung 等，2025 预印本 | 固定原子及键电子矩阵中的电子重分布，严格守恒；局部价态仍非全部硬约束；部分数据受许可证限制 | arXiv:2502.12979；数据 DOI 10.6084/m9.figshare.28359407.v2 |
| React-OT，Duan 等 | 给定反应物/产物的几何生成，并研究 xTB 端点与低保真预训练 | arXiv:2404.13430 |
| MolGEN，Tuo、Chen、Li | 条件流匹配用于 TS、开放产品生成及网络探索；TS 与产品任务条件不同，不能简单称所有任务都无产物 | arXiv:2507.10530v4（2025-11-05 版本记录） |
| DeePEST-OS / DORTS，Ren 等 | GFN2-xTB 与 MACE Delta-learning；反应区能量/力数据及 TS 优化。不是本项目可直接宣称首创的组合 | Nature Communications，2026-05-08；DOI 10.1038/s41467-026-72945-0 |
| DORTS 数据与 DeePEST-OS 代码 | 论文给出公开存档入口；仍须逐一核验许可、实际文件与适用域 | 数据 DOI 10.5281/zenodo.17141108；代码 DOI 10.5281/zenodo.19305799 |
| Robust generative transition-state models for unseen chemistry | 新元素/化学环境留出与平衡构象自监督预训练 | Nature Computational Science，2026-08-12；DOI 10.1038/s43588-026-01034-5 |
| ReCurveflow | 利用 NEB 曲线路径监督与离路径纠正；给定真实反应物/产物几何；2026-08-21 预印本，非已核实的同行评议结论 | arXiv:2608.20869v1 |
| TSBench | 2026-09-08 预印本；LLM agent 构建 TS 并经量化验证；输入含产物与导出的反应中心，且总体经过 xTB 兼容性预筛，不能充当本项目无产物/完整覆盖的直接证明 | arXiv:2609.08503v1 |
| Kingfisher | 活性溶剂与自动微溶剂化路径搜索，须检查其输入要求和事后活性水判定 | arXiv:2502.07965v1 |
| dxtb | PyTorch 中的可微 xTB 实现；正式采用前验证选定方法、溶剂、导数阶数和SCF稳健性 | DOI 10.1063/5.0216715；官方文档 `https://dxtb.readthedocs.io/en/latest/` |
| xTB 反应路径文档 | `--path` 是给定端点的路径方法；输出 TS guess 不等于验证完成 | `https://xtb-docs.readthedocs.io/en/latest/path.html` |
| ORCA 官方优化文档 | 过渡态/外部势接口与 Hessian 验证；使用时冻结实际安装版本 | `https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/optimizations.html` |
| Dimer 方法，Henkelman、Jónsson，1999 | 不需已知终态的经典鞍点搜索；沿不稳定方向反转力的思想不是新发明 | DOI 10.1063/1.480097 |
| Gentlest Ascent Dynamics，E、Zhou | 指数一鞍点的动力系统构造及分析；不直接保证任意学习引导的全局收敛 | arXiv:1011.0042；DOI 10.1088/0951-7715/24/6/008 |
| Thioester-mediated RNA aminoacylation and peptidyl-RNA synthesis in water | Ala-SEt/核苷实验约束；位点改造、供体水解、pH 与产物稳定性需分别整理 | Nature，2025-08-27；DOI 10.1038/s41586-025-09388-y |

## 应当检验而非预先认定的差异

不是“第一个 FM TS 模型”，不是“第一个半经验势加机器学习”，也不是“第一个多保真反应数据集”。候选差异是：无真实产物输入下的守恒事件—三维溶剂/反应模式联合提议、鞍点感知耦合的可归因增益，以及前瞻微观态/溶剂竞争通道数据和独立化学验证。必须在匹配资源和输入的对照中成立。
