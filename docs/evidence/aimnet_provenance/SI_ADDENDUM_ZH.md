本报告全部结果为探索性；新增两篇补充材料与指定 rxn 权重哈希核查后，两模型关于 T1x 或派生结构的结论仍为 `UNRESOLVED`，不适用统计置信区间。

# 来源核查补充：已取得并阅读两篇 SI

本补充文件更新同目录 REPORT_ZH.md 的“SI 未取得”及“rxn 精确文件未关联”状态；原报告保留作为取证过程，不覆盖旧证据。

## 新增直接证据

1. AIMNet2 SI，Supplementary Table 1，PDF第2页，标题短引文：“Number of molecules and conformers in training and test datasets”。表格列出 ANI-1x、ANI-2x、Orbnet、Peptide dimers、ChEMBL 和六种 PubChem 采样来源，总计20,328,583。没有把 Transition1x 列为独立来源。这证明作者报告的来源分类，不证明每个上游来源均排除 T1x 派生结构或结构重叠。
   - 论文 DOI：https://doi.org/10.1039/D4SC08572H
   - 可重取 SI 的公开接口：https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12057637/supplementaryFiles
   - ZIP成员：`SC-016-D4SC08572H-s001.pdf`；已完整通过 unzip -t 并读取13页文本，非先前截断 ZIP。

2. AIMNet2-rxn SI，Figure S1，PDF第2页，短引文：“The recalculated barrier heights from the Transition-1x benchmark”。该图明确是通用 AIMNet2 和 MACE-OFF23 的反应势垒基准，不是 rxn 的训练来源证明。
   - 论文 DOI：https://doi.org/10.1126/sciadv.aea1557
   - SI接口：https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13644750/supplementaryFiles
   - ZIP成员：`sciadv.aea1557_sm.pdf`；完整通过 unzip -t，全文已读，包括Figs.S1–S6、Table S1、Notes S1/S2。

3. 同 rxn SI，Table S1，PDF第2页，短引文：“All systems are composed of CHNO atoms and calculated as neutral singlets”。该表提供4–62原子训练体系的规模分布；Notes S1/S2 讨论批量 NEB 算法及算力比较，没有提供逐结构来源或 T1x 排除审计。元素、电荷和规模重合不能推断数据泄漏。
   - URL同第2项，Table S1及Supplementary Notes 1/2。

4. 官方注册表 `models / aimnet2-rxn_0` 及 `aliases / aimnet2_rxn_0`，短引文：“file: aimnet2_rxn_0.pt”。官方SHA256为 `ce8bc2a4cc6fcb5ebaa1bc8897fcd75104e5a00fe86e9d5fc6a020d572174894`；对n2现有只读资产执行sha256sum得到完全相同结果。因此精确权重与官方注册条目的关联已核实，不再只是旧别名推测。
   - URL：https://github.com/isayevlab/aimnetcentral/blob/main/aimnet/calculators/model_registry.yaml
   - 本轮未下载或替换模型权重。

## 结论与未解决边界

- `aimnet2_wb97m_d3_0.pt`：`UNRESOLVED`。正文和SI的来源分类没有列出T1x，但没有足够证据排除结构或反应派生关系。
- `aimnet2_rxn_0.pt`：`UNRESOLVED`。正文说明完整RGD-1、约四十万额外枚举反应和主动学习；SI中的T1x出现于基准图，不能认作训练包含。精确权重已关联，但派生关系仍未核实。
- 未取得数据集页面的有效正文和Hugging Face模型卡：后者DNS解析失败；Figshare API为错误响应，kilthub页面为空。没有把空响应或挑战页当作资料。数据集说明这一交付要求仍未完整满足，整体任务不宣称完成。
- 获取ZIP分别限制为90MB以下；实际文件大小与哈希见公开来源清单。没有登录、接受许可、安装软件或下载大型训练集。

Coding-Agent: pi
pi-Version: 0.99.2
Model: gpt-6-astra
Reasoning-Effort: off
