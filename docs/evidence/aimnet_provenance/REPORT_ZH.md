本报告全部结果为探索性；当前对两个指定模型是否包含 Transition1x 或其派生结构的结论均为 `UNRESOLVED`，这是来源证据判断，不是带统计区间的数值估计。

# AIMNet2 训练来源核查（待补充 SI 的阶段性报告）

## 结论与边界

| 指定模型 | 当前结论 | 已证实内容 | 尚不能证明 |
|---|---|---|---|
| `aimnet2_wb97m_d3_0.pt` | `UNRESOLVED` | 注册表文件名和 SHA256 与本项目资产一致；论文明确列出 PubChem、ChEMBL、ANI-1x、ANI-2x、OrbNet 等来源 | 没有逐结构排除 T1x 或由 T1x 反应衍生结构的证据 |
| `aimnet2_rxn_0.pt` | `UNRESOLVED` | 官方文档保留该旧名称；论文说明使用完整 RGD-1、额外枚举反应和主动学习 | 未取得当前指定权重的训练结构清单及 T1x 派生关系排除审计 |

不能因为来源段落没有列出 T1x 就判定 `EXCLUDES_T1X`。在基准测试中使用 T1x 也不等于训练包含 T1x。不同 DFT 标签水平不意味着几何结构不重叠。

## 可核对的原文证据

所有下列短引文均不超过十五个英文词；中文描述区分来源陈述与本报告的推论。

1. 模型注册表 `models / aimnet2-wb97m-d3_0`：文件 `aimnet2_wb97m_d3_0.pt`，SHA256 为 `f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28`，与交接和 n2 实测一致。短引文：“file: aimnet2_wb97m_d3_0.pt”。
   URL: https://github.com/isayevlab/aimnetcentral/blob/main/aimnet/calculators/model_registry.yaml
   该证据把使用中的文件与公开注册名称关联，不单独证明训练来源。

2. AIMNet2 原论文，Methods → Dataset preparation。短引文：“selected neutral and charged molecules under 20 heavy atoms from PubChem”。同段描述 ChEMBL、互变异构/质子化、几何优化、扭转扫描和分子动力学。
   URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC12057637/
   DOI: https://doi.org/10.1039/D4SC08572H

3. 同论文、同段。短引文：“supplemented with systems from ANI-1x, ANI-2x and OrbNet datasets”。这是明确的补充数据来源，不是完整结构排除清单。
   URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC12057637/

4. 同论文，Data availability。短引文：“molecular structures in the training datasets used in this study are publicly available”。论文指向数据 DOI；本轮访问数据 API 返回 403，尚未读取完整数据说明。
   URL: https://doi.org/10.1184/R1/27629937.v2
   对应论文段落: https://pmc.ncbi.nlm.nih.gov/articles/PMC12057637/

5. AIMNet2-rxn 官方文档，Loading → Legacy alias。短引文：“aimnet2_rxn_0 … aimnet2_rxn_3 still resolve and remain supported”。文档将 Hugging Face 模型卡指定为来源详情的正式入口，该入口本轮访问超时。
   URL: https://github.com/isayevlab/aimnetcentral/blob/main/docs/models/aimnet2_rxn.md
   模型卡 URL: https://huggingface.co/isayevlab/aimnet2-rxn

6. AIMNet2-rxn 原论文，Materials and Methods → Dataset construction。短引文：“The entire RGD-1 dataset”。同段解释使用该数据作为第一代模型的初始反应覆盖，并从原 AIMNet2 取平衡及近平衡中性 CHNO 构象。
   URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC13644750/
   DOI: https://doi.org/10.1126/sciadv.aea1557

7. 同论文、同节。短引文：“a collection of ∼400,000 additional reactions”。作者进一步描述 b2f2/b3f3 键电子矩阵编辑、UFF/GFN2 弛豫、NEB、受限 MD 和四轮主动学习，最终为 4,685,165 个几何结构。该来源说明并未提供本项目可核验的 T1x 派生排除关系。
   URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC13644750/

8. 同论文，Introduction，讨论 Fig.1 与 Fig.S1 的段落。短引文：“we also benchmark these models on the Transition1x dataset”。此处指通用 AIMNet2 和 MACE-OFF23 基准，不能据此认定 rxn 的训练包含 T1x。
   URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC13644750/

9. 同论文，Results → AIMNet2-rxn，Fig.4 附近。短引文：“reactant and product molecules that are not represented in the training set”。该句限定图2和图4的验证体系，不能无依据扩大为“全部 T1x 及其派生结构都排除”。
   URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC13644750/

## 获取记录与未完成项

- 已读取两个原论文的 Europe PMC 全文 XML；官方 GitHub README、模型文档、注册表均已取得。原始副本保存在 n2 的本轮运行目录 `provenance/`。
- 全文接口：`https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12057637/fullTextXML` 和相应 `PMC13644750/fullTextXML`。
- SI 的 PMC / publisher / CDN 链接分别返回 404、403；Europe PMC supplementaryFiles 两次超时留下不完整 ZIP，另一次返回 500。HTML 挑战页和不完整 ZIP **没有当作 PDF 或已读 SI**。
- 尚需：取得并阅读两篇论文 SI、数据集说明和 rxn 模型卡；核对 rxn 文件的具体版本/哈希；有直接证据才改变结论。没有下载大型训练集来做隐含的额外计算。
- 未登录网站，未接受第三方许可，未安装包，单次下载设置了低于 100 MB 的上限。

本报告是可审计的阶段性结果，不宣称已经满足交接中“含 SI 和数据集说明”的全部来源核查要求。
