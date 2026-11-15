# 用少量新增计算扩展已有数据用途：代码调研与主要障碍

日期：2026-11-15。**已将能定位到的相关实现拉到本地并检查核心数据与训练路径；
完成 7 个仓库、27 个关键文件的静态核验，以及 n2 GenerativeQC 安装身份复核。**
本轮没有下载分子数据、训练模型或新增量化计算，不能据此声称复现了论文结果。

结论：这条线可以做，但第一项任务应是“在明确目标层级下，用少量新 E/F 改善
非平衡构象的修正/局部优化”，而不是笼统填满数据库。已有代码能提供模型、损失和
数据读取组件，尚不能直接提供我们需要的**部分标签利用、按真实成本选点与下游评测**。
这是本轮的研究建议，不是已经验证的收益或新颖性结论。

## 1. 本地交付与版本边界

- 代码目录：项目根目录下 `scratch/data-completion-survey-20261115/repos/`。
  采用浅克隆与稀疏检出，保留源代码、配置、文档及 GEOM 教程；没有获取数据包、
  权重或 Git LFS 资产。目录约 15 MiB，不纳入 xtbflow 源码提交。
- 调研分支：`codex/data-completion-survey-20261115`；base 为 `main` 的
  `61269f2241cb182a010763291f349a0258c111f9`。无实现上的堆叠依赖。
  参考了尚未合并的 #206 → #207 → #208 的明确版本证据，未把它们写成 main 已有实现。
- 原工作树 `08764ae` 及其大量未提交修改保持原状。旧文档中的历史状态与新证据分开解释。
- [克隆清单](../evidence/data_completion_survey_20261115/clone_manifest.json)记录远端、
  完整 SHA、耗时、检出范围和一次错误 URL 的失败；[静态核验](../evidence/data_completion_survey_20261115/static_audit.json)
  记录逐文件 SHA-256、源码观察、输入字节和实际检查耗时。
- [n2 安装身份](../evidence/data_completion_survey_20261115/n2_runtime_metadata.json)
  记录本次只读观察；已有科学结果引用原报告，不重跑、不覆盖。

| 文献/用途 | 拉下的官方仓库及 SHA 前缀 | 实际取得的范围 |
| --- | --- | --- |
| GEOM 原始字段与版本 | [learningmatter-mit/geom](https://github.com/learningmatter-mit/geom/tree/ebd5569e130f4dcfbecf2a88718247fc2f71b03e) · `ebd5569` | 四份教程和数据说明，不是原始数据库 |
| 多保真：低层级力＋高层级能量 | [lanl/hippynn](https://github.com/lanl/hippynn/tree/0dfa22d281cc89fea63179f70bbfedff3d4b17a5) · `0dfa22d` | 论文对应 `multifidelity_ani1x_training.py`、网络与数据库实现 |
| EKD 伪力蒸馏 | 同一个 HIPPyNN 仓库 | 获得论文使用的基础库；**尚未定位到该论文专用的完整教师/学生复现实验脚本** |
| ICML 2025 物理弱监督 | [nec-research/PICPS-ML4Sci](https://github.com/nec-research/PICPS-ML4Sci/tree/fc04b1c990a532a6d1d1e1b726615b053ec585a6) · `fc04b1c` | 官方 PIWSL 实现、训练器、ANI 预处理与配置 |
| 能量引导构象生成 | [Rich-XGK/EnFlow](https://github.com/Rich-XGK/EnFlow/tree/2332d74ea3e2b3bee49109133feebce6c332a39c) · `2332d74` | 官方模型、预处理、训练与采样；README 的预训练权重仍为 Coming Soon |
| 无能量引导构象基线 | [shenoynikhil/ETFlow](https://github.com/shenoynikhil/ETFlow/tree/80f82d7d142bc3872412d2966a8d71ed14462b18) · `80f82d7` | EnFlow 所依赖的基础生成模型，便于追踪丢失字段 |
| 梯度标签价值研究 | [qmlcode/qml](https://github.com/qmlcode/qml/tree/607d49eba83f54d2db9bdda11fb3b9824f8796e8) · `607d49e` | 论文声明使用的基础库、梯度核及测试；未确认论文所有实验脚本，默认分支为 archive |
| ANI 原始数据读取 | [aiqm/ANI1x_datasets](https://github.com/aiqm/ANI1x_datasets/tree/c6b6d20b9915a4534204800b634fc1f8ee0a08bc) · `c6b6d20` | 官方 HDF5 loader 与说明，不含 Figshare 数据 |

因此，“基础库已拉下”与“完整论文代码已获得”不能混用。EKD 专用脚本的搜索覆盖
论文正文、其声明的基础库、GitHub 仓库检索与论文号检索；本轮未找到，不推断其永远不公开。

## 2. xtbflow 原来究竟缺什么

**不能把 xtbflow 的历史问题改写成一个普通的 energy-only 构象数据库问题。**

1. 2026-09-30 的 [Reaction-QM 审计](https://github.com/askusk521-cyber/xtbflow/blob/08764ae16efd3bc4c9d8b25e33c4626961bdc6ee/docs/REACTION_QM_WORKFLOW_STATUS_20260930.md)
   显示，endpoint/TS 几何和来源能量并非全都不存在。阻塞点包括 coordinate-to-map
   证据、母体系/反应族分组、跨来源 overlap 和数据许可；RGD1 还有 charge/multiplicity
   缺失。重新计算 E/F 不能唯一恢复这些历史信息。ReactionQM 的 quarantine 不改变。
2. 当时 joint event/TS 训练缺少可准入的样本级配对监督。SPICE2 已有 E/F 与同几何
   GFN2 校正 pilot，但它不提供反应事件、真实 TS 或反应连通标签。见
   [当时的后续行动记录](https://github.com/askusk521-cyber/xtbflow/blob/08764ae16efd3bc4c9d8b25e33c4626961bdc6ee/docs/NEXT_ACTIONS.md)与[同几何配对实现](../../src/xtbflow/data/pairing.py)。
3. 后来已经出现真实训练模型与 DA 开发资产，因此不能把“历史零训练”当作当前状态。
   [11 月 11 日证据](NARROWED_STUDY_REPORT_20261111.md)还涉及曝光池、物理反馈的时机、
   步长和独立认证问题，不能归因于单一缺失标签。
4. [11 月 14 日资产清点](https://github.com/askusk521-cyber/xtbflow/blob/0f91a1a7c62b6523f56b3dfa23a68f2b25417153/docs/research/CHEMISTRY_CANDIDATES_20261114_ZH.md)
   已确认 1,580 条配对 DA 缓存、406 个按反应物图归并的组合，以及 201 条旧开发记录。
   图分组不等于独立反应族，已曝光数据也不能重新成为确认集。

真正可以由新增计算提供的是：**明确条件下的新 E/F、非平衡邻域覆盖、同几何的跨层级
配对，以及经过更完整路径计算才能得到的反应连通证据**。最后一项通常比单点贵得多。
来源身份和历史采样条件应先恢复或标 unknown，不能用补算伪装成恢复历史事实。

## 3. 代码里最影响研究设计的发现

### A. GEOM 的能量常常是“没有被模型用到”，不是“没有计算”

GEOM 的 README 与教程说明了能量和统计权重；MoleculeNet 扩展还包含不同层级及
热校正信息，不能把所有子集的字段含义视为一致。
[官方说明](https://github.com/learningmatter-mit/geom/blob/ebd5569e130f4dcfbecf2a88718247fc2f71b03e/README.md)。

更具体的代码证据是：ETFlow 的
[`prepare_data.py:141`](https://github.com/shenoynikhil/ETFlow/blob/80f82d7d142bc3872412d2966a8d71ed14462b18/scripts/prepare_data.py#L141)
读取并保存 `totalenergy`，但
[`dataset.py:66`](https://github.com/shenoynikhil/ETFlow/blob/80f82d7d142bc3872412d2966a8d71ed14462b18/etflow/data/dataset.py#L66)
重新构造训练图时没有返回能量。权重参与预处理排序，也不等于训练中进行了概率加权。

**后果：必须先做“恢复原有字段、零新增 QC”的基线。** 先建立分子、原子顺序、
conformer ID、几何哈希与原始文件的关联，才能判断接下来是在补标签还是修复流水线。
本轮没有读取实际 GEOM 数据，具体选定版本的字段完整率仍未知。

### B. 现成 loader 倾向于取完整交集，正好丢掉我们想保留的不完整数据

HIPPyNN 多保真示例的
[`load_db:184–201`](https://github.com/lanl/hippynn/blob/0dfa22d281cc89fea63179f70bbfedff3d4b17a5/examples/multifidelity_ani1x_training.py#L184)
要求 CC 能量与两个选定能量都非 NaN，再过滤所有数组。官方 ANI
[`dataloader.py:16–20`](https://github.com/aiqm/ANI1x_datasets/blob/c6b6d20b9915a4534204800b634fc1f8ee0a08bc/dataloader.py#L16)
也对请求的所有性质取非 NaN 交集。

这适合对应示例，**却不是“大量 E-only＋少量 E/F＋部分低层级标签”的通用训练器**。
需要按样本、性质、保真层级分别保存 mask 和 provenance；真实参考、低层级和教师伪力
分开路由。先索引已观测值再计算损失，不能用零填充或 `0 * NaN`。
损失还须声明按构象、原子还是母分子归一，避免有力的小子集无意改变整体权重。

此外，示例第 201 行用两个位置参数调用 split API，而当前
[`database.py:213`](https://github.com/lanl/hippynn/blob/0dfa22d281cc89fea63179f70bbfedff3d4b17a5/hippynn/databases/database.py#L213)
只接受关键字参数。标准库 AST/签名绑定已复现 `too many positional arguments`。
这说明**当前检出的示例需要兼容修复**，不说明论文当年的实验运行失败。
该示例还使用记录级随机划分；我们的构象外推需先按母分子分组再切分。

### C. EnFlow 的能量梯度不能直接接成量化力监督

[`dataset.py:90–96`](https://github.com/Rich-XGK/EnFlow/blob/2332d74ea3e2b3bee49109133feebce6c332a39c/enflow/data/dataset.py#L90)
将能量按单个分子的已保存构象集合标准化。模型
[`model.py:513`](https://github.com/Rich-XGK/EnFlow/blob/2332d74ea3e2b3bee49109133feebce6c332a39c/enflow/models/model.py#L513)
让负能量梯度拟合 `s_t`，该训练路径中的目标来自生成路径位移；fine-tuning 才另加
对标准化来源能量的损失。这不是参考 QC force 的单位和语义。

若接真实力，应明确物理标量势、标准化逆变换、坐标单位与生成引导之间的接口。
新点加入会改变构象集合的 mean/std，不能每轮任意重算后仍称同一训练目标；新分子
推理也不能依赖不可见的完整参考构象能量统计。预处理还按权重截断构象，并对训练/
验证中超过 60 原子的分子作过滤，需要报告这带来的覆盖边界。

最稳妥的首轮是保持生成器固定，使用独立、物理单位明确的 E/F 模型做局部修正，
先检验新标签的价值，再决定是否改变 flow loss 或做流内引导。
预处理将总能量转为 float32 后才在 loader 中中心化；针对小相对能差的新适配器应
先保留 float64 完成能量差/基准处理，再按验证过的精度要求转换。

### D. PICPS 值得做对照，但复用前有实际适配成本

官方实现位置是
[`mdsim/trainers/trainer.py:1352`](https://github.com/nec-research/PICPS-ML4Sci/blob/fc04b1c990a532a6d1d1e1b726615b053ec585a6/mdsim/trainers/trainer.py#L1352)。
物理一致性项会对扰动构型增加模型求值；它们不是新增参考 QC 标签，但训练成本不能省略。

- `preprocessing/ani-1x_new.py` 默认 `--size=100K`，而 `Data_Size` 只列
  50/100/1K/10K。静态核验确认默认值不在表里；在源 NPZ 可用且进入该查询时会失败。
- 同一入口假设已有 `ani1x_wb97x_dz.npz`，并不是从官方 HDF5 到训练库的完整交付。
  数据转换、单位和分组身份需要自行核对。其 split 是样本索引随机打乱。
- 三点一致性代码对随机 `dirc1` 在去中心化前后各乘一次 `delta`，对 `dirc2`
  只乘一次；`delta=0.05` 时对应尺度为 0.0025 与 0.05 Å。**数值表达式可确认，
  是否为作者有意设计尚未验证**。不能未经核查就沿用论文中的单一“扰动大小”。
- `reshape(nb, -1, 3)` 的分子分段假设需在变长分子 batch 上验证；固定原子数数据
  可运行，不代表可以直接用于 GEOM 的变长分子。
- 部分关键源文件头声明非商业研究使用条款；仓库还包含其他组件许可证。
  本轮只保存链接与哈希，不将这些源文件复制进 xtbflow。若要发布修改版需单独处理权限。

### E. 伪力、多保真力与目标层级真力解决的是不同问题

[EKD 论文](https://arxiv.org/abs/2503.14293)使用多个 energy-only 教师的平均梯度指导学生，
并没有新增目标层级的真实力计算；教师训练与伪标签生成也有成本。
[多保真论文](https://arxiv.org/abs/2505.01590)的官方示例提供共享表示与双能量头，
两个力由各自能量的负梯度得到。这两类方法都是必要对照，但都不能证明某个伪力
已经成为目标层级真值。

关键问题应收紧为：**已有能量＋可用弱监督之后，少量新增参考 E/F 的边际收益在哪里？**
同一个几何上的方法差异与不同几何之间的差异必须分开。不能因 SMILES 相同便作严格
Delta 配对，也不能把 GQC B3LYP 力标为 ANI 的 CCSD(T)*/CBS 真力。

QML 的梯度核与测试可用来理解 E-only/E+F 对照；`setup.py` 依赖
`numpy.distutils`、Fortran 和 BLAS，应在隔离环境处理兼容性。
[梯度价值论文](https://arxiv.org/abs/2007.09593)的结论也依赖任务与分子范围，
不能预设“每个力标签总是比每个能量标签划算”。本轮未安装上述外部训练依赖。

## 4. GenerativeQC 实际能提供什么

本次 `ssh n2` 确认主机 node2，xtbflow 的 Python 3.10.19 / PyTorch 2.10.0+cu128，
队列观察为空。GQC 使用另一套 Python 3.11.17，包版本 0.1.0；其运行环境的
NumPy 2.4.6、SciPy 1.17.1、PySCF 2.14.0 不能当作 xtbflow 环境版本。

读取私有运行配置声明的 upstream SHA 为
`2bf3fd98da9c6cfba46ca83345e63e1d934e4780`；本次实际读取原生库 986,609,080 bytes，
SHA-256 为 `67bc98a4549fe8854ed6ae65791c824024bd77a9e8e498c6891dea3b8de670c4`，
与原资格报告一致。这是安装身份复核，**没有再次调用 Calculator 或验证导数**。

| 能力 | 已有证据范围 | 对补数据的影响 |
| --- | --- | --- |
| CUDA B3LYP/def2-SVP，球谐、FP64、无密度拟合 | 小型中性单重态有机体系的有限差分与同级别 PySCF 对照通过 | 可以作为新协议下的小规模 E/F 标签引擎；新元素、电子态或更大分子需小探针 |
| CUDA M06-2X/def2-SVP 力 | 既有测试失败，能量吻合不能解除限制 | 不能沿该路径生成训练力、做优化或动力学 |
| CUDA def2-TZVP 力 | 已测试分子的梯度积分不支持 | 不能假设换个基组名称即可升级标签 |
| Hessian、频率、TS/IRC | 上述 GQC 力资格没有涵盖；已有频率工作由 CPU PySCF 提供 | 不把 E/F 补算写成完整反应路径或自由能数据库补全 |

依据：[GQC pilot](https://github.com/askusk521-cyber/xtbflow/blob/91d3c7f205b1eb470a1c1c7a18049c5a1a0cde72/docs/evidence/gqc_tautomer_20261114/RESULTS_ZH.md)、
[后续方法/基组检查](https://github.com/askusk521-cyber/xtbflow/blob/932b011d21bb6ae1f0fb23c8ceeefa8764566cf4/docs/evidence/qc_campaign_20261114/RESULTS_ZH.md)。
原生坐标/能量/力单位分别为 Bohr、Hartree、Hartree/Bohr；xtbflow 使用 Å 与
Hartree/Å，接口要显式转换并保留原子顺序、charge、multiplicity、环境与协议哈希。

现成编译程序降低的是新标签的实施门槛；本轮没有测量它相对其他引擎的速度优势。
同一种新增计算策略的比较应使用同一引擎，否则软件速度与数据策略效果无法分离。

## 5. 建议首先做的一个实验

**一个小分子构象来源，一个固定的局部修正模型，一个下游任务：在相同目标层级下，
以更少真实 QC 成本把非平衡起点修正到身份不变、达到力阈值的结构。**
优先候选是 GEOM-QM9 的可审计小子集；实际纳入数量与许可、原始字段、charge/spin
完整性先在 n2 清点。已优化点的近零力、邻域扰动与生成失败点分别成为选点池。

这里若旧标签是 GFN/xTB，新标签是 B3LYP，就必须称为**新增目标层级 E/F 的多保真增补**；
不能把它描述成“补回原数据库缺失的同层级力”。若选择 ANI-1ccx 则面临目标层级更贵、
GQC 当前资格不能直接覆盖 CC 真力的问题，因此不建议作为第一个真实补算任务。

建议先按已有完整标签遮蔽来调通 mask、分组与预算软件，再执行真实增补。遮蔽实验
使用的完整标签在获取前视为隐藏，不进入选点特征；其费用曲线只能称模拟预算。
真实来源整体缺力时，还应测试这种结构化缺失，不能只做随机遮蔽。

| 固定模型后的对照 | 目的 |
| --- | --- |
| 恢复原有 E/权重，零新增 QC | 排除字段在流水线中丢失造成的假缺口 |
| 旧点随机补目标 E/F | 简单增补基线 |
| 邻域新点随机补目标 E/F | 区分缺标签与缺结构覆盖 |
| 多样性或不确定度选点 | 常规主动学习基线 |
| 任务失败区域选点 | 检验下游相关选择是否有额外收益 |
| 相同点集＋部分监督/多保真/伪力/PIWSL | 单独检验“怎样用标签”，避免与选点同时变化 |

先比较旧点/邻域点与随机/任务相关选择的最小交叉，不同时引入所有模型和算法。
可以预设几个递增的调用上限档位，但真正横轴应是记录到的新增墙钟、CPU/GPU
资源及 E/F 操作种类；失败、重试、未结算调用都留在账本。一次 E/F 调用同时产出能量
和力时，能量也是新信息，不能当成免费补出的背景标签。训练、选点、伪标签生成、
数据生成与最终参考评测分别计费，再报告合计成本。

主要下游指标：固定起点上的优化成功率、身份/立体化学保留、达到固定力阈值的真实
QC 成本和失败率。相对能量、力误差、构象覆盖作为辅助指标。只达到力阈值时称
收敛结构；称极小点需要额外曲率/频率证据。母分子先分组，邻域点、来源重复、不同
层级和同一轨迹都继承同组；确认集独立冻结，不用于选点或模型/阈值选择。

“达到同等效果显著节省成本”仍是待验证假设。若随机补一点已经等效，就报告数据
需求与任务边界；若只有 E/F MAE 下降而优化没有获益，不宣称获得了新的下游能力。
统计单位是母分子，而不是大量相关构象；小试验只用于判断信号，不能证明普遍泛化。

## 6. 复核方法与未完成项

静态审计只用 Python 标准库，不 import 外部仓库，不读取真实数据。
在项目根目录执行（`SURVEY_WT` 为本报告所在独立工作树）：

```bash
SURVEY_WT=.codex-worktrees/data-completion-survey-20261115
python3 "$SURVEY_WT/docs/evidence/data_completion_survey_20261115/audit_sources.py" \
  --repos scratch/data-completion-survey-20261115/repos \
  --manifest "$SURVEY_WT/docs/evidence/data_completion_survey_20261115/clone_manifest.json" \
  --out scratch/data-completion-survey-20261115/static-audit-replay.json
python3 "$SURVEY_WT/scripts/validate_bootstrap.py"
git -C "$SURVEY_WT" diff --check
```

输出文件必须不存在。审计限制单文件 4 MiB、合计 32 MiB；Git 读取每次超时 15 秒。
本轮基线 bootstrap 515 文件通过，新增报告与证据在提交前按脚本刷新并验证清单。
当前完成状态为源码探索和运行环境核对，**未做论文复现、数据完整率统计或性能基准**。

第一次 n2 观察继承 SSH 登录工作目录，未记录 cwd，不运行远端项目源码；该原始
记录保留。随后另做显式工作目录和预算的只读复核，见
[probe_n2_runtime.sh](../evidence/data_completion_survey_20261115/probe_n2_runtime.sh)与
[独立复核输出](../evidence/data_completion_survey_20261115/n2_runtime_recheck.json)。
脚本 SHA-256 为 `b8c45f413eadd3e9cf0ba524f4efadb851cc7a8352d811df54c1640988a8ffde`。
它从项目目录经 SSH 在 n2 的登录 home 中激活 xtbflow-qc，再用已配置的 GQC
解释器读取 metadata 和库哈希；不需要远端 Git checkout，不 import 原生模块。
总墙钟上限 75 秒、Python 60 秒、库读取 2 GiB、0 QC/训练/GPU 作业、无自动重试。
本次复核读取 986,609,080 bytes，程序墙钟约 0.655 秒；这不是量化计算性能。

```bash
SURVEY_WT=.codex-worktrees/data-completion-survey-20261115
set -o noclobber
bash "$SURVEY_WT/docs/evidence/data_completion_survey_20261115/probe_n2_runtime.sh" \
  > scratch/data-completion-survey-20261115/n2-runtime-replay.json
```

该命令只生成新的本地 JSON，私有解释器/库路径不写入报告。第一次和复核阶段均为
成功的 metadata 观察；不存在新的科学计算失败或训练结果可报告。

仍需解决的工作按顺序为：实际数据版本/字段/许可审计；新增 E/F 目标协议；分组与
部分标签 adapter；外部示例兼容修复；冻结最小实验与独立评测预算；最后才是补算。
本轮不修改第三方实现、不新增训练预算，也不把未取得的 EKD 专用脚本写成已经复现。
GQC 历史限制、ReactionQM quarantine、已曝光测试池和所有旧失败证据继续有效。
