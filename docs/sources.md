# 主要先行工作与定位依据

核查日期：2026-09-29。本表是定向检索起点，不是穷尽的新颖性、专利或优先权审查；正式投稿前必须做增量更新。优先使用原始论文、正式数据记录和官方实现，不从二手介绍推断能力边界。

| 工作 | 本次核实的相关点 | 原始来源标识 |
|---|---|---|
| TSDiff | 从二维反应图生成三维 TS；因此图到 TS 的生成任务不是本项目首创 | arXiv:2304.12233 |
| OA-ReactDiff | 以已知反应物/产物对象条件化，采用 object-aware SE(3) 等变扩散生成 TS | arXiv:2304.06174 |
| FlowER，Joung 等 | 固定原子/键电子表示中的电子重分布与严格全局守恒；不是三维 TS 生成器 | arXiv:2502.12979；数据 DOI 10.6084/m9.figshare.28359407.v2 |
| React-OT，Duan 等 | 给定反应物/产物几何的确定性 TS 生成；研究 GFN2-xTB 数据预训练、端点输入与 DFT 工作流 | DOI 10.1038/s42256-025-01010-0；arXiv:2404.13430 |
| MolGEN，Tuo、Chen、Li | 条件流匹配用于 TS、开放产品生成及网络探索；TS 与产品模块的信息合同不同 | arXiv:2507.10530，提交前复核最新版 |
| TS-DFM | 在分子距离几何空间进行条件流匹配，由反应物/产物端点生成 TS | DOI 10.1038/s41467-026-74101-0；arXiv:2511.17229 |
| TransTS | 从原子映射的反应物—产物对显式学习结构变换并生成 TS，报告 IID/OOD 精修与 intended-path 结果 | arXiv:2608.14076 |
| DeePEST-OS / DORTS，Ren 等 | GFN2-xTB 与 MACE Delta-learning 的反应势和 TS 搜索；不是可单独宣称首创的组合 | DOI 10.1038/s41467-026-72945-0 |
| DORTS 数据与 DeePEST-OS 代码 | 公开存档入口仍需逐一核验许可、实际文件与适用域 | 数据 DOI 10.5281/zenodo.17141108；代码 DOI 10.5281/zenodo.19305799 |
| Robust generative transition-state models for unseen chemistry | 新元素/化学环境留出与平衡构象自监督预训练；最终比较需匹配输入信息 | DOI 10.1038/s43588-026-01034-5 |
| ReCurveflow | NEB 曲线路径监督与离路径纠正；给定真实反应物/产物几何 | arXiv:2608.20869v1 |
| TSBench | agent 构建 TS 并经量化验证；输入与预筛边界必须与 reactant-only 任务分开 | arXiv:2609.08503v1 |
| UniTS | 由二维分子图和反应位点索引生成三维 TS，覆盖有机合成/有机金属案例；正式比较需核实数据和输入 | 官方实现 `licheng-xu-echo/UniTS` 及其 Nature Communications 论文 |
| Kingfisher | 活性溶剂与自动微溶剂化路径搜索；须检查其输入要求和事后活性水判定 | arXiv:2502.07965v1 |
| dxtb | PyTorch 中的可微 xTB；采用前验证选定方法、溶剂、导数阶数和 SCF 稳健性 | DOI 10.1063/5.0216715；官方文档 |
| xTB 反应路径文档 | `--path` 是给定端点的路径方法；输出 TS guess 不等于验证完成 | xTB 官方 path 文档 |
| ORCA 官方优化文档 | 过渡态、外部势与 Hessian 验证；使用时冻结实际安装版本 | ORCA 6.1 官方优化文档 |
| Dimer 方法，Henkelman、Jónsson | 不需已知终态的经典鞍点搜索；沿不稳定方向反转力的思想不是新发明 | DOI 10.1063/1.480097 |
| Gentlest Ascent Dynamics，E、Zhou | 指数一鞍点的动力系统构造及分析；不保证任意学习引导的全局收敛 | arXiv:1011.0042；DOI 10.1088/0951-7715/24/6/008 |
| Thioester-mediated RNA aminoacylation | 水中 thioester/thioacid 化学选择性、RNA 二醇氨酰化与 peptidyl-RNA 实验约束 | DOI 10.1038/s41586-025-09388-y |
| Chemoselective aminonitrile coupling | 水中硫化物介导的 α-aminonitrile 肽连接与蛋白源侧链范围 | DOI 10.1038/s41586-019-1371-4 |
| Activated pyrimidine ribonucleotide synthesis | glycolaldehyde/cyanamide 到 2-aminooxazole 及 amino-oxazoline 路线；phosphate 条件必须从原文/SI 冻结 | DOI 10.1038/nature08013 |

## 应当检验而非预先认定的差异

不是“第一个 FM TS 模型”，不是“第一个守恒反应模型”，不是“第一个半经验势加机器学习”，也不是“第一个多保真反应数据集”。候选差异是：在无正确产物/参考 TS 输入下，构造性守恒的事件状态与三维几何状态是否通过动态双向耦合优于匹配串联基线，以及模型预测的无符号反应投影是否在固定 calculator 预算下改善参考验证 TS 的发现效率。

所有来源能力在写入 [`docs/research/novelty_matrix.md`](research/novelty_matrix.md) 时按输入、输出、约束、物理调用和验证拆分。预印本状态、版本和官方实现应在每次论文冻结前重新核查。
