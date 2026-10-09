本报告全部结果为探索性；补读数据集 DOI 元数据及模型卡镜像后，两指定模型是否含 T1x 或派生结构仍为 `UNRESOLVED`；这是证据结论，不是统计估计。

# 数据集说明与模型卡取证补充

本文件补充 REPORT_ZH.md 和 SI_ADDENDUM_ZH.md，不覆盖早期获取失败记录。两篇正文和完整 SI、官方 GitHub 文档/注册表、数据 DOI 说明现均有可读证据；模型卡使用公开镜像，明确不冒充已直连官方源站。

## 数据集说明：可核对的官方 DOI 注册元数据

- DOI：https://doi.org/10.1184/R1/27629937.v2
- 可读取URL：https://api.datacite.org/dois/10.1184/R1/27629937.v2
- 位置：`data.attributes.descriptions[0]`，类型 Abstract；发布者 Carnegie Mellon University，题名 Training datasets for AIMNet2 machine-learned neural network potential。
- 短引文：“Each data file contains about 20M structures”。说明指出数据包含B97-3c或wB97M-def2-TZVPP计算的分子结构和性质，ORCA5.0.3，性质包括能量、力、电荷和电偶极/四极矩。
- 注册元数据是数据集说明，不是训练结构清单；其中没有T1x派生关系或重复结构审计，不能据此给出EXCLUDES_T1X。元数据指向约29GB数据集，本轮没有下载该训练集。

## 模型卡：镜像证据及其限制

1. 原始URL：https://huggingface.co/isayevlab/aimnet2-rxn/blob/main/README.md
   实际读取URL：https://hf-mirror.com/isayevlab/aimnet2-rxn/raw/main/README.md
   位置：Model Details → Training set。短引文：“combining the AIMNet2 base set with the RGD-1 reaction database”。该模型卡给出4,685,165个几何，并说明重新计算标签。与原论文的训练来源描述相容，但不能代替原论文更详细的额外枚举/主动学习说明，也没有证明T1x包含或排除。
2. 原始URL：https://huggingface.co/isayevlab/aimnet2-wb97m-d3/blob/main/README.md
   实际读取URL：https://hf-mirror.com/isayevlab/aimnet2-wb97m-d3/raw/main/README.md
   位置：Model Details → DFT functional。短引文：“DFT functional: wB97M-D3”。该卡没有逐来源数据表；来源判断仍以原论文SI Table1、正文Dataset preparation和官方注册表文件SHA为依据。
3. 官方Hugging Face域名仍DNS失败，镜像内容未能与源站独立逐字节核对，故模型卡仅作支持性证据。卡中有版本相关说明和格式转换信息，不能无依据将新safetensors发布与旧`.pt`资产的训练结构完全等同。两指定`.pt`的官方注册哈希与n2资产关联已在SI_ADDENDUM_ZH.md记录。

## 交付结论

| 指定文件 | 结论 | 核心理由 |
|---|---|---|
| aimnet2_wb97m_d3_0.pt | UNRESOLVED | 来源表列ANI/PubChem/ChEMBL/OrbNet等，未列T1x，但无派生结构排除证据 |
| aimnet2_rxn_0.pt | UNRESOLVED | 来源为原AIMNet2部分数据、完整RGD-1、额外枚举反应和主动学习；T1x在SI中为基准图，非训练包含证据 |

没有直接证据支持INCLUDES_T1X，也没有足够的结构级排除证据支持EXCLUDES_T1X。负责人若需要肯定排除结论，需要作者确认或另行批准的逐结构来源/反应派生审计；本交接只要求公开资料核查，不擅自扩展为下载训练集。未登录或接受许可，未安装模型卡示例中的软件，未下载大型模型/数据集。

Coding-Agent: pi
pi-Version: 0.99.2
Model: gpt-6-astra
Reasoning-Effort: off
