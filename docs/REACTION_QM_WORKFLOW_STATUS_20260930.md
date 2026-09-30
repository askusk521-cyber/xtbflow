# Reaction-QM / RGD1 workflow status

核查基准：2026-09-30。完成度按 `docs/PARALLEL_REACTION_DATA_WORKFLOW_PROMPT.md` 的阶段出口判断；“完成”表示有可复现证据，不表示可以跳过后续科学门槛。七个阶段同权时，当前整体约 **35–40%**：第一阶段基本完成，第二阶段已有严格适配合同，第三至第七阶段仍有数据准入或协议冻结缺口。

| 阶段 | 当前状态 | 证据与完成内容 | 未完成或阻塞 |
| --- | --- | --- | --- |
| 1. 官方数据源核查 | **约 90%** | n2 已缓存并校验 Reaction-QM v2 的 reaction-info、B3LYP endpoint/TS HDF5、common metadata、reactant enumeration，以及官方 B3LYP train/valid/test CSV；已记录 URL、版本、文件大小、MD5、SHA-256、字段、记录数、CHNOS、neutral singlet、坐标顺序和缺失字段。RGD1 v6 的 CSV、DFT info、RandP 映射、CHNO HDF5 也已审计。 | Reaction-QM Zenodo 页面当前没有显示可机器读取的 record license；Reaction-QM 代码仓库的 BSD-3-Clause 许可证不能替代数据资产许可证。IRC 9.9 GB 资产按提示词有意延后。 |
| 2. Reaction-QM 适配层 | **约 65–70%** | 已有 source config、hash-gated read-only loader、schema、endpoint event provenance、完整 payload/source revision/source asset hash、charge/multiplicity 奇偶校验、product/TS 输入防泄漏、坐标顺序 quarantine 和 14 个专项测试。 | `ReactionQMRecord` 还没有收敛为现有公共 `PublicRecord`/Track-B 准入合同的统一输出；官方数据没有 parent/family/independent-system/repeated-TS 或 coordinate-to-map 证据，因此没有任何真实 admitted row。跨来源 overlap audit 也未完成。 |
| 3. 第一批真实联合训练数据 | **0%（Reaction-QM）** | 官方 split CSV 已冻结为 159,913 train / 19,990 validation / 19,987 test，199,890 个 ID 全覆盖、无 ID 交叉、无跨 split 完整 reaction-SMILES 重复；该审计只证明 ID partition。 | 仍没有可用于 `development_*` 的 Reaction-QM manifest：1,062 条机械候选及其余记录都缺少可核实坐标映射或分组证据，不能猜测补齐。 |
| 4. n2 真实 independent/serial/joint 训练 | **约 10–15%** | 软件控制、unconstrained-event 对照要求、预算账本和 n2 环境已有；完整代码测试已在 n2 通过。 | 没有 Reaction-QM 样本级 paired event/TS manifest，因此没有真实数据训练、方法矩阵、物理验证或科学指标。 |
| 5. Reaction-QM 同构型 GFN2 校正势 | **0%（Reaction-QM）** | 同构型校正的 SPICE2 E/F 工具链和 256 配置开发 pilot 已完成，可复用实现。 | 尚未从准入的 Reaction-QM DFT 路径抽取构型；未下载或运行 IRC，也不能用 SPICE2 结果替代 Reaction-QM 结果。 |
| 6. RGD1 跨来源验证 | **约 35–40%** | 官方 RGD1 v6 资产、映射字段、176,898 CSV rows、176,992 HDF5 groups、94-row release mismatch 和 charge/multiplicity 缺失已固定。 | 未建立与 Reaction-QM 训练库的联合化学身份 overlap audit；未冻结 external validation protocol，也未运行 TS/geometry cross-source validation。 |
| 7. Transition1x 对外 benchmark 对齐 | **约 25–30%** | 已保留 Transition1x 几何基线和输入防泄漏边界。 | 尚未在模型选择前冻结 protocol 哈希、官方 split、输入视图、K、blind/best-of-K、后处理和计算预算；因此不能报告最终外部排行榜。 |

## 当前可公开的结论

- Reaction-QM v2 的官方 split 可以作为**来源 ID 分区**，不能被称为母反应或独立反应物体系分区。
- `common_reaction_info.csv` 的 89,636 个 GFN↔B3LYP 对是跨层级身份连接，不是母反应/反应族/重复 TS/坐标映射表。
- `reactant_combinations.txt` 的 500,001 行没有 reaction ID，不能生成 split 分组。
- HDF5 的坐标数组没有 atom-map ID。元素序列一致仍不能证明坐标行对应 map ID；因此当前 loader 对记录保留 quarantine。
- 当前唯一已准入的公开训练来源仍是独立的 SPICE2 OpenFF E/F development pilot；它不能替代 Reaction-QM 反应事件/TS 监督。

## 需要在 PR 中决定的事项

1. 在收到独立 coordinate-to-map 表和 parent/family/independent-system/repeated-TS 表之前，是否保持 Reaction-QM 全部 quarantine（建议保持）。
2. 如果只允许 geometry-only development lane，是否接受“来源坐标顺序”而不生成 mapped event/TS 训练标签；这需要单独的合同，不能把它升级为 certified mapping。
3. RGD1 是否只做一次冻结的 external test；若要做 validation，必须先冻结模型/阈值选择规则和联合 overlap audit。
4. 是否由数据提供方补充 Reaction-QM record license；在此之前仓库只保存哈希和配置，不再分发原始字节。

复现入口：

- source audit: `scripts/audit_reaction_qm_sources.py`
- grouping audit: `scripts/audit_reaction_qm_grouping.py`
- official split audit: `scripts/audit_reaction_qm_official_splits.py`
- reports: `docs/evidence/reaction_qm_rgd1_source_audit_20260930.json`, `docs/evidence/reaction_qm_grouping_audit_20260930.json`, `docs/evidence/reaction_qm_official_split_audit_20260930.json`
