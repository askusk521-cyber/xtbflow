# Reaction-QM / RGD1 data gate

核查日期：2026-09-30。原始公开字节只存放在 n2 的
`/home/lhshen/.cache/xtbflow`，仓库只保留来源配置、哈希、审计摘要和可复现脚本。

Reaction-QM v2 的反应信息 CSV 和 B3LYP-D3/TZVP endpoint/TS HDF5 已在 n2
缓存并通过官方 MD5、文件大小和 SHA-256 校验。CSV 与 HDF5 均有 199,890 条
反应记录；HDF5 每条 TS 记录含 `smiles`、`EHG`、`charge`、`multiplicity`、
`atomic_numbers` 和 `coordinates`。TS 反应 SMILES 的 mapped graph 在 199,890
条中都能解析，且都是 neutral singlet；按 CHNOS 过滤后有 38,458 条，
161,432 条含 Cl/F/P/Si 等范围外元素，5,947 条含芳香键，直接隔离等待显式
芳香表示规则。

HDF5 坐标只保存原子序数和坐标，没有每个坐标行对应的 atom-map number。
195,720 条的 HDF5 原子序列与 mapped TS SMILES 顺序不同；即使元素多重集相同，
也不能把坐标行猜配到 map ID。按 CHNOS、neutral singlet、无芳香键、端点有键变更
和元素序列一致的机械候选只有 1,062 条；这些仍不能进入 confirmatory 或正式
几何训练，直到独立坐标-to-map 关系被核实。`ReactionQMLoader` 因此会对未核实
坐标映射保留明确失败原因，不自动重排。

RGD1 v6 的 atom-mapped index、DFT reaction info、`RandP_smiles.txt` 和
`RGD1_CHNO.h5` 已缓存并哈希固定。CSV/index 有 176,898 条唯一 channel，全部
为 CHNO，图形式电荷 proxy 为中性；缺少显式 charge 和 multiplicity。映射 CSV
中有 2,913 条芳香记录和 258 条无端点键变更。`RGD1_CHNO.h5` 有 176,992 条
几何/TS 记录，`RG`、`PG`、`TSG`、`Rsmiles`/`Psmiles`、能量、力、坐标、元素和
`_id` 字段在这 176,992 个组中完整出现，比 CSV 多 94 条；这些版本差异不能静默拼接。RGD1 只作为独立
cross-source validation 候选，不进入 Reaction-QM 训练。

来源固定版本、URL、文件大小、SHA-256 和字段映射见
[`configs/public_reaction_sources.yaml`](../configs/public_reaction_sources.yaml)。
审计入口是 [`scripts/audit_reaction_qm_sources.py`](../scripts/audit_reaction_qm_sources.py)，
审计输出见 [`evidence/reaction_qm_rgd1_source_audit_20260930.json`](evidence/reaction_qm_rgd1_source_audit_20260930.json)。
Reaction-QM 适配合同和 endpoint event provenance 在
[`src/xtbflow/data/reaction_qm.py`](../src/xtbflow/data/reaction_qm.py)。

当前 claim limits：endpoint `B_product - B_reactant` 只是
`derived_under_contract` 的图差分标签，不是电子密度运动、弯箭头机制或 IRC
证明；Reaction-QM 的母反应/反应族/独立反应物体系分组还需独立表；没有完成这些
分组与坐标映射审计前，不启动 confirmatory training。
