# 工作流 C2：CP2K 参考桥接与 TS 验证

工作流 C2 对应 #43、#17 和 #19。它把三种不同层级的证据分开：

1. **计算器正确性**：当前代码是否能用冻结 CP2K 身份稳定产生可审计的 E/F；
2. **参考桥接**：在完全相同的构型、电荷和多重度上，CP2K 与已资格化 CPU GFN2 的差异是多少；
3. **路径验证**：候选是否经过 TS 优化、频率/模态检查以及端点连通性验证。

前两层不能替代第三层。E/F 成功、有限差分通过或 CP2K–GFN2 差异较小，都不能单独证明一个结构是过渡态。

## 冻结运行身份

物理协议由 `configs/calculators/cp2k/protocol_v0.1.yaml` 给出。运行时加载器会在计算前检查：

- CP2K 版本和源码 revision 与协议中的 build identity 一致；
- basis、pseudopotential 和 D3 参数文件真实存在；
- 每个体系显式提供电荷和多重度；
- 运行时路径不写入公共证据，只公开可执行文件名和 SHA-256。

任一检查失败时，计算在启动 CP2K 前失败关闭。

## Gate C2.1：当前代码重放

```bash
export XTBFlow_CP2K_EXECUTABLE=/path/to/cp2k.psmp
export PYTHONPATH=src:vendor/mechai_reusable
export OMP_NUM_THREADS=4

python scripts/cp2k_calibration_smoke.py \
  --threads 4 \
  --timeout 600 \
  --artifact-dir "$RUN_ROOT/cp2k-calibration" \
  --output "$RUN_ROOT/cp2k-calibration.json"
```

三个校准体系分别覆盖中性闭壳层、带电开壳层和含硫闭壳层。所有 input、output、stdout 和 stderr 必须持久化。该 gate 只证明当前 adapter/runtime 可执行，不授予正式参考标签。

## Gate C2.2：解析力的独立有限差分检查

```bash
python scripts/cp2k_force_fd_smoke.py \
  --output "$RUN_ROOT/cp2k-force-fd.json" \
  --ledger "$RUN_ROOT/cp2k-force-fd-ledger.json" \
  --artifact-dir "$RUN_ROOT/cp2k-force-fd-artifacts" \
  --threads 4 \
  --step 1e-3 \
  --threshold 5e-4
```

脚本对一个非平衡水分子构型执行三次有预算令牌的 CP2K 调用：基准 E/F、正位移能量和负位移能量，并比较

\[
F_{i\alpha}^{\mathrm{analytic}}
\quad\text{与}\quad
-\frac{E(R+h e_{i\alpha})-E(R-h e_{i\alpha})}{2h}.
\]

通过只验证一个体系、一个坐标分量和一个冻结运行时；它不等于基组/截断收敛或广泛化学精度验证。

## Gate C2.3：有界同构型参考桥接

冻结配置位于 `configs/experiments/cp2k_bridge.yaml`。首个 pilot 只包含四个预先声明的体系：

- 非平衡中性水；
- 非平衡中性 H₂S；
- 非平衡闭壳层 H₃O⁺；
- 有限显式水环境中的水二聚体。

它们分别检查中性、含硫、带电和有限水环境，不能代表完整目标域。

```bash
python scripts/run_reference_bridge.py \
  --config configs/experiments/cp2k_bridge.yaml \
  --output "$RUN_ROOT/reference-bridge.json" \
  --ledger-dir "$RUN_ROOT/reference-bridge-ledgers" \
  --artifact-dir "$RUN_ROOT/reference-bridge-artifacts"
```

运行器强制串行执行，并分别记录 CP2K 和 GFN2 calculator calls。每个比较要求两种后端具有相同的 `input_hash`，即原子顺序、坐标、电荷、多重度和环境合同完全一致。

允许汇总的主要量是力分量差：

- component MAE；
- component RMSE；
- component maximum absolute difference。

绝对能量只保留在单体系记录中。不同化学计量之间不得汇总绝对能量差，也不得把四个小体系的均值写成广泛精度结论。

## Gate C2.4：截断与协议收敛

当前 600 Ry 协议仍标记为 provisional。收敛脚本使用同一非平衡水构型，按预先声明的 cutoff 梯度逐次调用 CP2K；每次底层 E/F 调用都必须先消耗持久化 ledger token，并保留独立 artifact。

```bash
python scripts/cp2k_convergence_smoke.py \
  --output "$RUN_ROOT/cp2k-convergence/report.json" \
  --ledger "$RUN_ROOT/cp2k-convergence/ledger.json" \
  --artifact-dir "$RUN_ROOT/cp2k-convergence/artifacts" \
  --cp2k-executable "$XTBFlow_CP2K_EXECUTABLE" \
  --cp2k-data-dir "$XTBFlow_CP2K_DATA_DIR" \
  --threads 4 \
  --cutoffs 500 600 700 \
  --energy-tolerance 1e-5 \
  --force-tolerance 1e-4

python scripts/publish_cp2k_convergence_evidence.py \
  --private-report "$RUN_ROOT/cp2k-convergence/report.json" \
  --ledger "$RUN_ROOT/cp2k-convergence/ledger.json" \
  --artifact-root "$RUN_ROOT/cp2k-convergence/artifacts" \
  --output docs/evidence/cp2k_convergence_<commit>.json
```

退出码 `2` 表示真实计算完成但冻结阈值未通过；它是有效负结果，不得删除或放宽阈值后冒充通过。只有所有 cutoff 成功且相对于最高 cutoff 的能量、力差同时通过时，gate 才标记为 `pass`。无论正负，单一水构型的结果都不等于基组、SCF、广泛化学精度或 TS 路径资格。

### 2026-09-29 实测结果（source commit `fcb8049`）

500、600、700 Ry 三次 CP2K E/F 均正常结束，持久化 ledger 恰好结算 3 次调用，无剩余 reservation。相对于 700 Ry：

- 500 Ry：`|ΔE| = 1.1019861841887746e-4 Ha`，`max |ΔF| = 8.653055924661562e-5 Ha/Å`；
- 600 Ry：`|ΔE| = 3.398007183719187e-5 Ha`，`max |ΔF| = 4.176294735423053e-5 Ha/Å`。

力阈值 `1e-4 Ha/Å` 通过，但 600→700 Ry 的绝对能量差超过预注册阈值 `1e-5 Ha`，因此本 gate 为 **fail**，600 Ry 协议继续保持 provisional，不升级为正式参考标签。路径清理后的机器可读证据见 `docs/evidence/cp2k_convergence_fcb8049.json`；该负结果不否定 adapter 的实机可执行性，也不构成 TS、连通性或广泛化学精度结论。

## Gate C2.5：真实 TS/模态/连通性

#19 的软件验证器已经能够记录候选身份、预算、模式证据和端点证据，但物理准入仍要求至少一个真实成功通道和一个失败对照：

1. 使用独立参考协议进行 TS 优化；
2. 计算 Hessian/频率并确认恰好一个相关虚频；
3. 检查虚频模态与目标反应事件的一致性；
4. 运行 IRC 或等价双向路径；
5. 根据端点重新推导观察到的事件，而不是复用目标标签；
6. 保留所有 calculator calls、失败、重试和原始 artifact。

在此 gate 完成前，`path_status` 必须保持 `not_requested`、`not_run` 或 `not_validated`，不得写成 `validated`。

### 低成本候选准入预检

为避免在明显不处于鞍点邻域的候选上直接消耗完整 dimer 预算，带有
`preflight` 配置的开发试验先使用独立持久化账本完成最多 3 次 GFN2 E/F
调用：一次声明几何的能量/力，以及沿声明内部模态正、负位移的两次计算。
预检记录刚体投影后的力范数和方向曲率
`v^T H v`；只有曲率符号和力阈值同时通过，runner 才创建正式 dimer
预算。预检失败是有效负结果，正式 dimer ledger 不得产生。

NH₃ 反转开发候选冻结在
`configs/validation/gfn2_dimer_nh3_pilot_v0.1.json`。该筛选仅用于低成本
候选准入，不是完整 Hessian、相关虚频或端点连通验证；即使后续 dimer
收敛，也不能直接改变 #19 的 `path_status`。

```bash
python scripts/run_gfn2_dimer_pilot.py \
  --config configs/validation/gfn2_dimer_nh3_pilot_v0.1.json \
  --output "$RUN_ROOT/nh3-dimer/report.json" \
  --ledger "$RUN_ROOT/nh3-dimer/dimer-ledger.json" \
  --preflight-ledger "$RUN_ROOT/nh3-dimer/preflight-ledger.json" \
  --artifact-dir "$RUN_ROOT/nh3-dimer/artifacts"
```

### 2026-09-29 NH₃ 内部 Hessian 与双向端点实测（source commit `409e906`）

冻结的平面驻点在本次独立阶段重新计算得到梯度范数
`3.275187997245202e-7 Ha/Å`。6 维内部 Cartesian Hessian 的特征值为：

```text
[-11.5076920, 11.5713878, 11.5714837,
  48.3812130, 115.9848947, 115.9852735] eV/Å²
```

仅第一个模态低于 `-0.05 eV/Å²`，观察到的负模态数恰为 1；该模态与此前
dimer 模态的绝对重叠为 `1.0`。Hessian 阶段使用 13 次调用，其中 1 次为冻结
驻点 E/F，12 次为 6 个内部基方向的中央差分。

沿独立 Hessian 负模态正负各位移 0.2 Å 后，两侧 BFGS 各使用 7 次调用并收敛。
最终 N 到有序 H₃ 平面的带符号距离分别为
`+0.36683693584122057 Å` 与 `-0.36683693584121835 Å`；两侧梯度范数约为
`1.17e-6 Ha/Å`，能量均比平面驻点低约 `0.0097372 Ha`。全部 ledger 均已结算，
无 pending reservation，总调用数为 27。

因此本轮的 `development_path_status` 为 `validated`，但
`reference_path_status` 被明确保留为 `not_validated`。原因是全部证据仍来自
有限气相 GFN2 模型，内部 Hessian 只报告能量/长度平方特征值而非谐振频率，且
NH₃ 反转的 `observed_bond_event` 为空。机器可读证据见
`docs/evidence/gfn2_nh3_path_409e906.json`。此前 H₃ 正曲率候选继续作为有效失败
控制，不得删除或重标记。

### NH₃ 独立内部 Hessian 与双向端点开发门槛

`configs/validation/gfn2_nh3_path_v0.1.json` 将通过的 dimer 驻点绑定到父证据
SHA-256，并把后续证据拆成三个独立阶段：

1. 在冻结几何重新计算一次 E/F；
2. 从 6 个刚体自由度的正交补构造 NH₃ 的 6 维内部 Cartesian 基，以中央差分
   `2 × 6` 次调用计算完整内部 Hessian；
3. 仅在恰好一个显著负模态且该模态与 dimer 方向绝对重叠不低于 0.95 时，沿
   正负方向分别位移并使用独立 BFGS 账本松弛端点。

对 NH₃，双向端点使用 N 原子到有序 H₃ 平面的带符号距离区分两个金字塔极小值；
两侧必须符号相反、绝对距离不低于 0.1 Å、能量低于平面驻点且梯度通过门槛。
由于该过程不改变键连接，`observed_bond_event` 必须保持为空，不能将其包装成
键变化反应。即使全部开发 gate 通过，`reference_path_status` 仍保持
`not_validated`，直到独立 CP2K 或批准的参考协议完成对应证据。

```bash
python scripts/run_gfn2_path_pilot.py \
  --config configs/validation/gfn2_nh3_path_v0.1.json \
  --output "$RUN_ROOT/nh3-path/report.json" \
  --ledger-dir "$RUN_ROOT/nh3-path/ledgers" \
  --artifact-dir "$RUN_ROOT/nh3-path/artifacts"
```

### 2026-09-29 NH₃ 驻点收紧重放（source commit `89cfed9`）

新的 v0.2 配置从 `2771281` 的最终几何出发，但使用新协议身份、新账本和更严格
门槛；旧的 fail 记录保持不变。3-call 预检再次确认负曲率
`-11.56015900736597 eV/Å²`。随后 dimer 搜索在 1 个优化步、19 次调用后收敛，
最终完整梯度范数为 `3.275187997245202e-7 Ha/Å`，dimer 曲率为
`-11.507484459044218 eV/Å²`，通过预注册的 `5e-4 Ha/Å` 开发门槛。
预检与 dimer 合计 22 次调用，77 次未使用额度被释放，两个 ledger 均无 pending
reservation。

该 **pass 只授予驻点/dimer driver gate**。`full_ts_validation` 仍为 `false`：尚未
独立计算完整内部 Hessian、确认恰好一个负模态，也未完成沿该模态的双向端点
松弛。机器可读记录见
`docs/evidence/gfn2_dimer_nh3_refine_89cfed9.json`。它不能被写成 CP2K 参考 TS、
反应机理或键变化事件证据。

### 2026-09-29 NH₃ 反转候选预检与 dimer 重放（source commit `2771281`）

平面 NH₃ 反转候选先通过独立的 3-call GFN2 预检：刚体投影后的力范数为
`0.07576791037668802 Ha/Å`，声明反转模态方向曲率为
`-8.156935714550215 eV/Å²`。因此 runner 才创建正式 dimer 预算。

后续 dimer 搜索在 2 个优化步、31 次 GFN2 调用后由 ASE 报告收敛，最终
dimer 曲率为 `-11.560159007366892 eV/Å²`。但完整梯度范数为
`0.0011264100063743865 Ha/Å`，高于预注册门槛 `0.001 Ha/Å`，故本轮保留为
**fail**。预检与 dimer 合计 34 次调用，两个 ledger 均无 pending reservation；
未使用的 97 次 dimer 额度被释放。

该结果说明 NH₃ 候选处于明确负曲率区域，但尚未达到冻结的驻点门槛。不得
放宽阈值或把 ASE 的 dimer 收敛标志改写成 TS 验证通过。下一步必须使用新的
配置与新账本进一步收紧驻点，然后独立计算内部 Hessian，并沿唯一负模态执行
双向端点松弛。机器可读、无主机绝对路径的记录见
`docs/evidence/gfn2_dimer_nh3_2771281.json`；这仍是 GFN2 开发证据，不是 CP2K
参考 TS 准入。

### 2026-09-29 首次 GFN2 dimer 开发试验（source commit `b755757`）

单个线性 H₃ 双重态候选在冻结的 96-call、并发 1、重试 0 预算下运行 ASE dimer 驱动。ledger 恰好结算 96 次 GFN2 调用，无 pending reservation；运行因预算耗尽而以 **fail** 结束。轨迹显示最低模态逐步转成近似整体平移 `[0.576, 0.579, 0.578]`，结构随后偏离声明的对称候选。该结果暴露了驱动未剔除平移/转动自由度，而不是参考 TS 证据。

路径清理后的负结果见 `docs/evidence/gfn2_dimer_pilot_b755757.json`。该记录不得删除或重新标记为通过；后续驱动必须在新提交、新 ledger 和新 artifact 目录中加入刚体模态投影后重放。即使后续 dimer 搜索收敛，仍须单独完成 Hessian/相关虚频和双向端点验证，才能改变 `path_status`。

### 2026-09-29 刚体投影后重放（source commit `ae3da3c`）

修复版正确识别线性 H₃ 的 5 个刚体自由度；预算耗尽时的最终模态为 `[0.596, 0.186, -0.782]`，分量和约为 `-2.36e-9`，不再是整体平移。新 ledger 再次恰好结算 96 次调用、无 pending reservation，并保留 11 个优化步和 JSONL 轨迹。

该重放仍为 **fail**：起始对称构型的外侧 H 实际力约为 `±0.08054 Ha/Å`，声明反应模态曲率为正 `18.35 eV/Å²`；搜索过程中曲率升至 `101.19 eV/Å²`。因此问题从“驱动刚体污染”收敛为“该 H₃/GFN2 初猜不在负曲率 TS 区域”。不得通过增加预算把它改写成成功；后续应改用经过低成本力/曲率筛选的候选。机器可读证据见 `docs/evidence/gfn2_dimer_pilot_ae3da3c.json`。

## 证据发布规则

公共证据可以包含：

- 物理协议和运行脚本的 SHA-256；
- CP2K 版本、revision、可执行文件 SHA-256；
- 体系输入、input hash、数值结果和预算计数；
- 原始 artifact 的文件名和内容哈希。

公共证据不得包含私有主机绝对路径、凭据或未经许可的大型原始输出。完整输入/输出保留在受控运行目录中，公共记录只保存经过清理的有界摘要。
