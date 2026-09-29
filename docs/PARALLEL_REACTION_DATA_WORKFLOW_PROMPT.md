# Reaction-QM / RGD1 数据路线并行工作提示词

下面的内容可以直接复制到一个新对话中，作为 xtbflow 项目的并行工作提示词。

---

请作为 xtbflow 项目的“公开反应数据路线负责人”，在现有工作基础上继续推进 Reaction-QM／RGD1 数据适配与真实联合训练。所有真实数据下载、解析、训练和量化计算必须在 n2 服务器上通过 `ssh n2` 执行；每完成一个可审查阶段，就创建独立分支并提交 PR，保持远程仓库最新，但不要自动合并 PR。

项目仓库：

<https://github.com/askusk521-cyber/xtbflow>

当前已存在的主要 PR：

- #67：Transition1x 隔离开发联合训练与几何基线
- #68：n2 CP2K 校准、有限差分和截断能收敛重放
- #69：CP2K–GFN2 同构型参考桥接 pilot

当前事实边界：

1. Transition1x 已完成审计，但缺少显式 formal charge、multiplicity、可信事件标签和独立认证 atom mapping，因此只能用于 quarantine development 和公开几何基线。
2. CP2K adapter、解析力、同构型桥接已经在 n2 真实环境完成有限验证；600 Ry CP2K 协议仍为 provisional。
3. 不得把 Transition1x 的同序原子直接称为 certified atom mapping。
4. 不得猜填 charge、multiplicity、mapping、反应族或事件标签。
5. 不得把端点差分事件描述成真实电子密度运动或完整弯箭头机制。
6. 不得把不同数据源的反应物、TS、产物跨库硬拼。
7. 不得把原始公开数据、模型权重或大文件复制进 git；只保留配置、哈希、审计报告、必要的小型 manifest 和可复现脚本。
8. 每个科学结果都必须写清数据来源、版本、哈希、筛选规则、claim limits 和失败记录。

## 总目标

以 Reaction-QM 的 DFT 子集作为主要联合训练源，Transition1x 作为经典公开基准，RGD1 作为跨来源验证。先完成可审查的数据适配，再启动真实 independent／serial／joint 训练。不要一开始下载全部 IRC 或 GFN2 大型资产；优先获取 Reaction-QM B3LYP-RXN 的 DFT 反应几何和反应信息表，必要时再补充少量 B3LYP-IRC 路径构型。

## 第一阶段：官方数据源核查

1. 在 n2 上检查网络、磁盘、Python 环境和 Slurm。
2. 只使用 Reaction-QM 和 RGD1 的官方页面、官方仓库、官方 Zenodo／Figshare／下载脚本。
3. 核对：
   - 数据集版本和发布日期；
   - 下载 URL；
   - 文件大小和 SHA-256；
   - 许可证或再分发限制；
   - 反应记录格式；
   - reactant/product/TS 坐标单位；
   - formal charge；
   - multiplicity；
   - atom mapping；
   - 反应族或反应物体系 ID；
   - DFT 方法、基组、溶剂和路径验证字段。
4. 先不要下载完整 IRC 或 GFN2 大文件。优先下载：
   - Reaction-QM B3LYP-RXN DFT 几何文件；
   - 对应反应信息 CSV／JSON；
   - RGD1 官方索引、映射和 TS 记录。
5. 下载文件只放在 n2 cache，例如：
   `/home/lhshen/.cache/xtbflow/reaction-qm/...`
   不得放进 git。
6. 生成第一份审计报告，至少包括：
   - source URL；
   - source revision；
   - file size；
   - SHA-256；
   - license status；
   - observed record count；
   - missing-field counts；
   - CHNOS 过滤前后数量；
   - neutral closed-shell 过滤前后数量；
   - mapping／TS／charge／multiplicity 完整性；
   - 不能准入的具体原因。
7. 完成后创建 PR，例如：
   `docs: audit Reaction-QM and RGD1 public sources`

## 第二阶段：实现 Reaction-QM 适配层

基于现有 `TrackBRecord` 合同实现适配，不修改已有治理边界。

要求：

1. 新增 source config，记录 Reaction-QM 和 RGD1 的固定版本、URL、SHA-256、许可证和字段映射。
2. 新增只读 loader，进行：
   - 文件哈希校验；
   - schema 校验；
   - 坐标形状和单位校验；
   - 原子顺序校验；
   - charge／multiplicity 校验；
   - mapping 唯一性校验；
   - TS 与 reactant 原子数匹配；
   - DFT protocol 身份记录。
3. 根据同一条映射反应的 reactant／product 图，按冻结规则派生：
   `B_product - B_reactant`
   作为 event label。
4. 派生标签必须明确写入：
   - `evidence: derived_under_contract`；
   - 源记录 ID；
   - 生成规则版本；
   - 输入图哈希；
   - 输出事件标签哈希。
5. 不能把派生端点事件称为真实电子密度路径。
6. 对芳香性、显式氢、共振式、对称原子、键级变化不明确的记录直接 quarantine。
7. 不得把 product 或 TS 信息放入 reactant-only 输入视图。
8. 同一条 Reaction-QM 反应必须同时提供：
   - reactant graph；
   - reactant coordinates；
   - charge；
   - multiplicity；
   - mapped event label；
   - product graph；
   - TS coordinates；
   - reference protocol；
   - source record hash。
9. 反应族、母反应、独立反应物体系和重复 TS 必须保存，不能只按 CSV 行号划分。
10. 写适配器单元测试，至少覆盖：
    - 合法 CHNOS 中性闭壳层记录；
    - 缺 charge；
    - 缺 multiplicity；
    - mapping 重复；
    - reactant/product 原子数不一致；
    - 歧义键变化；
    - product 泄漏到 reactant view；
    - 同一母反应跨 split；
    - 同一 input fingerprint 跨 split。
11. 先不要声称所有数据已经可用于 confirmatory training。根据字段完整性分成：
    - quarantine；
    - development_train；
    - development_validation；
    - development_test；
    - confirmatory。

完成后创建 PR，例如：

`feat: add Reaction-QM Track-B adapter`

## 第三阶段：建立第一批真实联合训练数据

目标不是一开始整理 199,890 条，而是先完成一批可审计的小规模数据。

建议目标：

- 100–300 个独立反应物体系；
- 至少包含 train／validation／test；
- 母反应不跨 split；
- 初始只使用 CHNOS；
- 只使用 neutral closed-shell；
- 保留所有过滤失败记录；
- 记录每个来源反应的 source hash；
- 记录独立反应物体系、反应族和路径副本去重规则。

必须生成：

1. `reaction_qm_track_b.v1.jsonl` 或等价 manifest；
2. 数据准入审计报告；
3. split audit；
4. leakage audit；
5. event-label provenance report；
6. CHNOS／charge／multiplicity 分布；
7. TS 记录完整性统计；
8. 失败样本清单；
9. 可复现构建命令。

如果 Reaction-QM 的真实字段不足以直接满足 confirmatory，应先生成 development 数据，并明确说明：

- 输入是否是来源 reactant endpoint；
- 是否独立生成；
- 是否使用了 product-derived event label；
- 是否有 DFT TS／IRC 验证；
- 当前只能支持哪种科学主张。

完成后创建 PR，例如：

`feat: admit first Reaction-QM development manifest`

## 第四阶段：在 n2 上运行真实 independent／serial／joint 训练

使用现有联合流模型和控制协议，不能只复用 Transition1x quarantine runner。

至少比较：

1. strong rule baseline；
2. event-only／geometry-only independent；
3. serial independent；
4. joint bidirectional；
5. same-integrator one-way control；
6. both-off control。

所有方法必须使用：

- 相同训练数据；
- 相同母反应 split；
- 相同随机种子集合；
- 相同训练步数；
- 相同候选数；
- 相同后处理；
- 相同物理验证预算。

报告指标：

- event legality rate；
- event decoder rejection rate；
- geometry RMSD；
- pair-distance MAE；
- endpoint connectivity；
- TS refinement success；
- independent validation success；
- 去重后的有效通道数；
- 每个有效通道的计算成本；
- 失败类型；
- conservation residual；
- product leakage audit；
- family holdout 结果（只有具备可信 family label 时才报告）。

训练优先在 n2 Slurm 上运行。使用独立 job script、job ID、环境记录、git commit、配置哈希和输出哈希。没有 GPU 时使用 CPU，不得把 CPU 结果称为 GPU benchmark。

完成后创建 PR，例如：

`feat: run Reaction-QM joint flow development benchmark`

## 第五阶段：Reaction-QM 同构型 GFN2 校正势数据

从 Reaction-QM DFT 训练反应中抽取有限路径构型，在完全相同坐标上运行 GFN2：

\[
\Delta E(X)=E_{\mathrm{DFT}}(X)-E_{\mathrm{GFN2}}(X)
\]

要求：

1. DFT 与 GFN2 必须使用同一坐标；
2. 不得减去两个优化后几何的能量；
3. 记录 charge、multiplicity、protocol、coordinate hash；
4. 训练／测试按母反应划分；
5. 不得把同一反应的路径帧当成独立反应；
6. 先做小规模 CHNOS 反应区域 E/F pilot；
7. 与现有 SPICE 校正势结果分开报告；
8. 单独比较：
   - no correction；
   - endpoint-only correction；
   - flow-time physical guidance；
   - equal-budget terminal optimization。

完成后创建 PR，例如：

`feat: add Reaction-QM same-geometry correction pilot`

## 第六阶段：RGD1 跨来源验证

只有 Reaction-QM 适配和内部 split audit 完成后，才启动 RGD1。

要求：

1. 使用完整、官方、修正后的 RGD1 资产；
2. 先做字段和许可证审计；
3. 不把 RGD1 作为 Reaction-QM 的训练数据；
4. 只做 cross-source validation；
5. 报告不同来源的：
   - geometry error；
   - event legality；
   - TS refinement；
   - endpoint connectivity；
   - valid channel count；
   - failure modes。
6. 如果 RGD1 结构超出 CHNOS／闭壳层范围，按明确规则过滤，不临时扩展模型化学范围。

完成后创建 PR，例如：

`eval: add RGD1 cross-source validation`

## 第七阶段：Transition1x 对外 benchmark 对齐

保留已有 Transition1x 几何基线，但进一步明确：

1. 采用哪套官方 split；
2. 是否提供 product；
3. 是否提供 event；
4. 是否使用 reaction center；
5. 采样数；
6. 后处理预算；
7. 与 React-OT、TS-DFM、MolGEN、TransTS 等外部方法的输入条件是否一致。
8. 不得把 product-conditioned TS 结果和 product-free channel discovery 结果混在同一排行榜。
9. 不得把 oracle `ts_guess_true` 当作模型性能。
10. 最好报告：
    - one-shot；
    - best-of-K；
    - blind-selected；
    - physical validation success。

完成后创建 PR，例如：

`docs: align Transition1x benchmark protocol`

## 执行规则

- 每次开始先检查当前分支、工作树、远程 PR 和 n2 工作树状态。
- 所有真实计算优先通过 `ssh n2`。
- 运行前检查环境：
  - Python；
  - torch；
  - tblite；
  - ASE；
  - RDKit；
  - Slurm；
  - CP2K；
  - OMP 线程。
- 不覆盖已有证据文件；使用新日期和新 source commit。
- 失败是有效结果，保留失败报告。
- 不放宽阈值来制造通过。
- 不自动合并 PR。
- 每个 PR 必须包含：
  - 做了什么；
  - 为什么做；
  - n2 运行命令；
  - git commit；
  - 数据／配置哈希；
  - 测试结果；
  - 真实计算结果；
  - claim limits；
  - 未完成项。
- 若发现 Reaction-QM 或 RGD1 实际字段与描述不一致，暂停依赖该字段的训练，先更新审计报告，不得猜测补齐。
- 如果下载受限，先完成 source config、loader contract、测试夹具和失败审计，不要偷偷使用 Transition1x 或合成数据代替 Reaction-QM。
- 任何新科学结论都必须等待独立验证和物理验证，不得从模型 loss 下降直接推出化学发现。

## 最终判定问题

最终目标不是尽快制造一个训练曲线，而是完成第一张可信的真实联合数据表，并用它回答：

1. 守恒事件表示是否降低非法事件率？
2. 双向 event–geometry coupling 是否优于同预算串联？
3. flow 内物理引导是否优于等预算末端优化？
4. Reaction-QM 训练后是否能在 Transition1x 和 RGD1 上保持可解释的跨来源能力？
5. 哪些新竞争通道是联合模型提出、规则或串联方法容易漏掉，并且经过独立物理验证？
