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
